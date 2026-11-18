from pathlib import Path

import yaml

CONFIGS = Path(__file__).resolve().parents[3] / "configs"


class Config:
    """Nested attribute-access view of a dict; `to_dict()` gives the dict back."""

    def __init__(self, d: dict):
        self.__dict__["_d"] = d
        for k, v in d.items():
            self.__dict__[k] = Config(v) if isinstance(v, dict) else v

    def __setattr__(self, k, v):
        raise AttributeError("Config is frozen")

    def to_dict(self) -> dict:
        return self._d

    def __repr__(self):
        return f"Config({self._d})"


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _load(p) -> dict:
    return yaml.safe_load(Path(p).read_text()) or {}


def _set(d: dict, key: str, val) -> None:
    *path, last = key.split(".")
    for p in path:
        if not isinstance(d.get(p), dict):
            raise KeyError(f"unknown config key: {key}")
        d = d[p]
    if last not in d:
        raise KeyError(f"unknown config key: {key}")
    d[last] = val


def load_config(case: str | None, tier: str, overrides: list[str]) -> Config:
    """base -> tier -> case -> overrides ('a.b=value', value YAML-parsed; unknown keys raise)."""
    d = _merge(_load(CONFIGS / "base.yaml"), _load(CONFIGS / "tiers" / f"{tier}.yaml"))
    if case:
        p = Path(case)
        d = _merge(d, _load(p if p.exists() else CONFIGS / p.name))
    for o in overrides:
        key, sep, val = o.partition("=")
        if not sep:
            raise ValueError(f"override must be key=value: {o}")
        _set(d, key.strip(), yaml.safe_load(val))
    return Config(d)


def dump_config(cfg: Config) -> str:
    return yaml.safe_dump(cfg.to_dict(), sort_keys=False)
