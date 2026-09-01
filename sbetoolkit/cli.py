"""CLI: regenerate the README figure from the marketplace simulator."""

from __future__ import annotations

import argparse
from pathlib import Path

from sbetoolkit.marketplace import MarketplaceConfig, MarketplaceSimulator
from sbetoolkit.plots import plot_naive_vs_switchback


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="docs/naive_vs_switchback.png")
    p.add_argument("--reps", type=int, default=80)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)

    sim = MarketplaceSimulator(
        MarketplaceConfig(
            n_regions=8,
            n_periods=24,
            riders_per_cell=40,
            drivers_per_cell=18,
            p_request_control=0.70,
            p_request_treat=0.90,
            seed=args.seed,
        )
    )
    comparison = sim.compare_estimators(n_reps=args.reps, seed=args.seed)
    out = Path(args.out)
    plot_naive_vs_switchback(comparison, path=out)
    print(f"wrote {out}")
    print(comparison[["naive_ate", "switchback_ate", "truth_ate"]].agg(["mean", "std"]).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
