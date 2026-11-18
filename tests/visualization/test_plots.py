import torch

from aeroshape.core.types import ForceHistory
from aeroshape.visualization import plots


def test_plot_functions_write_files(tmp_path):
    s = torch.arange(0, 100, 10)
    plots.plot_force_history(ForceHistory(s, 2 + 0 * s.float(), 0 * s.float()), tmp_path / "a.png", "t", ref=2.04)
    plots.plot_field(torch.rand(20, 40), tmp_path / "b.png", "w")
    plots.plot_convergence([10, 20, 40], [2.5, 2.3, 2.2], tmp_path / "c.png", ref=2.04)
    assert all((tmp_path / n).stat().st_size > 0 for n in ("a.png", "b.png", "c.png"))


def test_write_table1_md(tmp_path):
    rows = [dict(re=20, report=2.04, literature="Dennis and Chang 2.045", cd=2.1, error_pct=2.9, passed=False)]
    plots.write_table1_md(rows, tmp_path / "t.md")
    t = (tmp_path / "t.md").read_text()
    assert "| 20 | 2.04 |" in t and "2.100" in t and "FAIL" in t


def test_adjoint_plots(tmp_path):
    ga, gf = torch.rand(6), torch.rand(6)
    plots.plot_gradient_check(ga, gf, [1, 2, 4, 5], tmp_path / "g.png", err=0.01)
    F = torch.rand(9, 20, 40)
    plots.plot_adjoint_field(F, torch.zeros(20, 40, dtype=torch.bool), tmp_path / "f.png")
    assert (tmp_path / "g.png").stat().st_size > 0 and (tmp_path / "f.png").stat().st_size > 0
