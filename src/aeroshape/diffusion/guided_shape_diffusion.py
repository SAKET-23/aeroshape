"""Physics-guided DDIM sampler and training loss (Plan/04_models/diffusion.md, derivations 6.3-6.6)."""
from dataclasses import dataclass

import torch

from aeroshape.pinn.stress_integrator import surface_forces


def diffusion_loss(denoiser, schedule, z0, cond, cond_drop, generator):
    """eps-prediction MSE with classifier-free condition dropout; generator is a CPU generator."""
    B = z0.shape[0]
    t = torch.randint(1, schedule.T + 1, (B,), generator=generator).to(z0.device)
    eps = torch.randn(z0.shape, generator=generator).to(z0)
    drop = (torch.rand(B, generator=generator) < cond_drop).to(z0.device)
    return ((denoiser(schedule.q_sample(z0, t, eps), t, cond, null_mask=drop) - eps) ** 2).mean()


def project_thickness(nurbs, alpha, t_min, iters=50):
    """Gauss-Newton projection (thickness is linear in alpha) until thickness_at_ctrl >= t_min."""
    J = nurbs.thickness_at_ctrl(torch.eye(nurbs.D, dtype=alpha.dtype, device=alpha.device)).T  # [k, D]
    P = torch.linalg.pinv(J.cpu().double()).to(alpha.device, alpha.dtype)
    a = alpha.detach().clone()
    for _ in range(iters):
        viol = torch.relu(t_min - nurbs.thickness_at_ctrl(a))
        if viol.max() < 1e-7:
            break
        a = a + viol @ P.T
    return a


@dataclass
class SampleResult:
    alpha: torch.Tensor
    z_trajectory: torch.Tensor | None
    J_history: torch.Tensor  # [n_steps, B]; NaN on unguided steps


class GuidedSampler:
    def __init__(self, denoiser, schedule, nurbs, pinn, norm, cfg=None, physics_fn=None):
        """norm = (mean, std) of free coordinates; cfg: optional {w_L, mu}.
        physics_fn(alpha, cond[B,4]) -> (CD, CL); default = surface_forces with the frozen pinn."""
        self.dn, self.sch, self.nurbs, self.pinn = denoiser, schedule, nurbs, pinn
        self.mean, self.std = norm
        self.w_L, self.mu = (cfg or {}).get("w_L", 1.0), (cfg or {}).get("mu", 10.0)
        self.physics_fn = physics_fn or self._pinn_forces

    def _pinn_forces(self, alpha, cond):
        pc = torch.stack([cond[:, 2], cond[:, 3].sin(), cond[:, 3].cos()], 1)
        return surface_forces(self.pinn, self.nurbs, alpha, pc)

    def to_alpha(self, z):
        return self.nurbs.free_to_full(z * self.std + self.mean)

    def objective(self, alpha, cond, t_min):
        """Derivations (6.5); cond = [CL_ref, CD_ref, log Re, AoA_rad]."""
        CD, CL = self.physics_fn(alpha, cond)
        short = torch.relu(t_min - self.nurbs.thickness_at_ctrl(alpha))
        return 0.5 * (CD - cond[:, 1]) ** 2 + self.w_L * 0.5 * (CL - cond[:, 0]) ** 2 + self.mu * (short ** 2).sum(1)

    def guidance_grad(self, z0, cond, t_min):
        """(dJ/dz0_hat, J) with z0_hat a leaf: the step is applied to z0_hat, so the gradient must live in z0_hat space.
        (Taking it w.r.t. z_t puts the denoiser Jacobian dz0_hat/dz_t in between; it is not positive definite and climbed J, WP17 round 2.)"""
        with torch.enable_grad():
            z0 = z0.detach().requires_grad_(True)
            J = self.objective(self.to_alpha(z0), cond, t_min)
            g, = torch.autograd.grad(J.sum(), z0)
        return g, J.detach()

    def _eps(self, z, t, cond, w):
        B = z.shape[0]
        e = self.dn(torch.cat([z, z]), torch.cat([t, t]), torch.cat([cond, cond]),
                    null_mask=torch.cat([torch.zeros(B, dtype=torch.bool), torch.ones(B, dtype=torch.bool)]).to(z.device))
        return (1 + w) * e[:B] - w * e[B:]

    def sample(self, cond, t_min, n_steps=50, cfg_scale=2.0, guide_scale=1.0, guide_clip=1.0, guide_start=0.2,
               generator=None, record=False):
        dev, B = cond.device, cond.shape[0]
        z = torch.randn(B, self.nurbs.n_free, generator=generator).to(dev, cond.dtype)  # CPU noise: device-independent
        ts = self.sch.ddim_timesteps(n_steps)
        gb = self.sch.gamma_bar.to(dev, z.dtype)
        traj, J_hist = [], torch.full((n_steps, B), float("nan"), device=dev)
        for i in range(n_steps):
            t, tp = ts[i], ts[i + 1]
            guide = guide_scale > 0 and i >= guide_start * n_steps
            with torch.no_grad():
                eps = self._eps(z, t.expand(B).to(dev), cond, cfg_scale)
                z0 = ((z - (1 - gb[t]).sqrt() * eps) / gb[t].sqrt()).clamp(-4, 4)  # clipped clean estimate
            if guide:
                g, J = self.guidance_grad(z0, cond, t_min)
                g = g * (guide_clip / (g.norm(dim=1, keepdim=True) + 1e-12)).clamp(max=1.0)
                z0, J_hist[i] = z0 - guide_scale * g, J
            z = gb[tp].sqrt() * z0.detach() + (1 - gb[tp]).sqrt() * eps.detach()
            if record:
                traj.append(z)
        alpha = project_thickness(self.nurbs, self.to_alpha(z), t_min)
        return SampleResult(alpha, torch.stack(traj) if record else None, J_hist)
