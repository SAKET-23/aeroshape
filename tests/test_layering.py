import ast
from pathlib import Path

SRC = Path(__file__).parents[1] / "src" / "aeroshape"
# layer -> aeroshape subpackages it may import (itself always allowed)
ALLOWED = {
    "core": set(),
    "geometry": {"core", "lbm.lattice"},  # exception: pure D2Q9 constants only
    "lbm": {"core", "geometry"},
    "pinn": {"core", "geometry"},
    "diffusion": {"core", "geometry", "pinn"},
    "optimization": {"core", "geometry", "lbm", "pinn"},
    "visualization": {"core", "geometry", "lbm", "pinn", "diffusion", "optimization"},
}


def imported_layers(path):
    out = set()
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            mods = [n.module] + [f"{n.module}.{a.name}" for a in n.names]
        elif isinstance(n, ast.Import):
            mods = [a.name for a in n.names]
        else:
            continue
        for m in mods:
            parts = m.split(".")
            if parts[0] == "aeroshape" and len(parts) > 1:
                out.add(".".join(parts[1:3]) if parts[1:3] == ["lbm", "lattice"] else parts[1])
    return out


def test_layering():
    bad = []
    for layer, ok in ALLOWED.items():
        for f in (SRC / layer).rglob("*.py"):
            deps = {d if layer == "geometry" or d != "lbm.lattice" else "lbm" for d in imported_layers(f)}
            for dep in deps - ok - {layer}:
                bad.append(f"{f.relative_to(SRC)} imports {dep}")
    assert not bad, bad
