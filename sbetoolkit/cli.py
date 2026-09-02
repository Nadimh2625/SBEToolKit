"""CLI: README figures — bias chart or null Type I calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

from sbetoolkit.calibration import empirical_power_check, type_i_null_check
from sbetoolkit.marketplace import MarketplaceConfig, MarketplaceSimulator, summarize_estimators
from sbetoolkit.plots import plot_empirical_power, plot_naive_vs_switchback, plot_type_i


def _chart(args: argparse.Namespace) -> int:
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
    summary = summarize_estimators(comparison)
    csv_path = out.with_suffix(".csv")
    summary.to_csv(csv_path, index=False)
    print(f"wrote {out}")
    print(f"wrote {csv_path}")
    print(summary.to_string(index=False))
    return 0


def _null(args: argparse.Namespace) -> int:
    result = type_i_null_check(n_reps=args.reps, seed=args.seed, alpha=args.alpha)
    out = Path(args.out)
    plot_type_i(result, path=out)
    summary = result.summary()
    csv_path = out.with_suffix(".csv")
    summary.to_csv(csv_path, index=False)
    print(f"wrote {out}")
    print(f"wrote {csv_path}")
    print(summary.to_string(index=False))
    print(
        f"n_blocks={result.n_blocks}  "
        f"mean_rides={result.n_rides_per_rep:.0f}  "
        f"n_reps={result.n_reps}"
    )
    return 0


def _power(args: argparse.Namespace) -> int:
    result = empirical_power_check(
        n_reps=args.reps,
        seed=args.seed,
        alpha=args.alpha,
    )
    out = Path(args.out)
    plot_empirical_power(result, path=out)
    summary = result.summary()
    csv_path = out.with_suffix(".csv")
    summary.to_csv(csv_path, index=False)
    print(f"wrote {out}")
    print(f"wrote {csv_path}")
    print(summary.to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["chart", "null", "power"], default="chart")
    p.add_argument("--out", default=None)
    p.add_argument("--reps", type=int, default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args(argv)
    if args.mode == "chart":
        args.out = args.out or "docs/naive_vs_switchback.png"
        args.reps = 500 if args.reps is None else args.reps
        return _chart(args)
    if args.mode == "null":
        args.out = args.out or "docs/type_i_null.png"
        args.reps = 1000 if args.reps is None else args.reps
        args.seed = 11 if args.seed == 7 else args.seed
        return _null(args)
    args.out = args.out or "docs/empirical_power.png"
    args.reps = 500 if args.reps is None else args.reps
    args.seed = 13 if args.seed == 7 else args.seed
    return _power(args)


if __name__ == "__main__":
    raise SystemExit(main())
