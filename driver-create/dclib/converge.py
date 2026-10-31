"""Convergence loop: check every character and kart view, repair the failing ones,
re-check, and repeat until the whole set is PASS (or the round budget is spent).

This is the part the raw `imgai` CLI does not do: `imgai roster build` generates
once and then blocks a non-PASS set for a human. driver-create keeps working on the
failing views automatically, with a fresh base seed per round so the validated
attempt ladder explores new candidates instead of redrawing the same ones.

The hero images are never touched (imgai protects them); only the turnaround views
are regenerated, and every attempt is logged to repair-log.jsonl by imgai.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dclib import imgai as I

# A prime offset per round, far away from the ladder's own seed offsets
# (0, 1009, 2018, 3027, 4036) so each round explores genuinely new candidates.
ROUND_SEED_STEP = 100003


def base_seed(d: Path) -> int:
    spec = I.load_spec(d)
    return int(spec.get("character", {}).get("seed", 1))


def _repair(d: Path, name: str, root: Path, model: str, preset: str, seed: int, max_attempts: int, verbose: bool) -> None:
    for kind in ("character", "vehicle"):
        fail = I.failing_views(d)[kind]
        if not fail:
            continue
        if verbose:
            print(f"  repair {kind}: {', '.join(fail)} (seed {seed})", flush=True)
        I.run_imgai([kind, "repair", "--character", name, "--root", str(root),
                     "--model", model, "--preset", preset, "--seed", str(seed),
                     "--max-attempts", str(max_attempts)])


def converge(name: str, root: Path, *, model: str, preset: str, max_rounds: int,
             max_attempts: int = 6, verbose: bool = True) -> str:
    """Repair failing views round by round until the set is PASS or rounds are spent.

    Returns the final overall status (PASS / REVIEW_REQUIRED / FAIL).
    """
    d = I.set_dir(name, root)
    if not (d / "design-spec.json").is_file():
        raise FileNotFoundError(f"{d} has no design-spec.json; run the create/build stage first")
    seed0 = base_seed(d)

    for rnd in range(1, max_rounds + 1):
        fail = I.failing_views(d)
        if not fail["character"] and not fail["vehicle"]:
            if verbose:
                print(f"  round {rnd}: no failing views", flush=True)
            break
        if verbose:
            print(f"  round {rnd}: failing -> character {fail['character'] or '-'} | kart {fail['vehicle'] or '-'}", flush=True)
        _repair(d, name, root, model, preset, seed0 + rnd * ROUND_SEED_STEP, max_attempts, verbose)
        status = I.status_of(d)
        if verbose:
            I.print_status(name, status)
        if status == I.PASS:
            break
    return I.status_of(d)


if __name__ == "__main__":
    import argparse
    from dclib import imgai as I
    p = argparse.ArgumentParser(description="Converge one driver set to PASS")
    p.add_argument("name")
    p.add_argument("--root", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--preset", default=None)
    p.add_argument("--max-rounds", type=int, default=None)
    p.add_argument("--max-attempts", type=int, default=None)
    a = p.parse_args()
    root = Path(a.root).expanduser() if a.root else Path(I.defaults().get("default_root", "~/Pictures/AI-Generated/drivers")).expanduser()
    df = I.defaults()
    st = converge(a.name, root, model=a.model or df.get("model", "moss"), preset=a.preset or df.get("preset", "balanced"),
                  max_rounds=a.max_rounds or df.get("max_rounds", 4), max_attempts=a.max_attempts or df.get("max_attempts", 6))
    I.print_status(a.name, st)
    raise SystemExit(0 if st == I.PASS else 1)
