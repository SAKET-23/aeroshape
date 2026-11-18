"""Compare two smoke summary.json files: python scripts/compare_summary.py CANDIDATE REFERENCE
Only the fields in TOL are compared (relative tolerance, e2e_mac_mini.md pass criteria); every stage in the candidate must be ok.
Exit 1 on any mismatch."""
import json
import math
import sys

TOL = {  # "stage.field": rtol
    "2_lbm_forward.cd": 1e-4,
    "5_pinn_train.mlp_final_loss": 5e-2,
    "5_pinn_train.transformer_final_loss": 5e-2,
}


def compare(cand: dict, ref: dict, tol: dict = TOL) -> list[str]:
    bad = [f"{k}: status {v['status']}" for k, v in cand["stages"].items() if v["status"] != "ok"]
    for key, rtol in tol.items():
        stage, field = key.split(".")
        try:
            a, b = cand["stages"][stage][field], ref["stages"][stage][field]
        except KeyError:
            bad.append(f"{key}: missing in candidate or reference")
            continue
        if not (math.isfinite(a) and abs(a - b) <= rtol * abs(b)):
            bad.append(f"{key}: {a:.6g} vs reference {b:.6g} (rtol {rtol:g})")
    return bad


if __name__ == "__main__":
    c, r = (json.load(open(p)) for p in sys.argv[1:3])
    bad = compare(c, r)
    print("\n".join(bad) if bad else f"OK: {len(TOL)} fields within tolerance, all stages ok")
    sys.exit(1 if bad else 0)
