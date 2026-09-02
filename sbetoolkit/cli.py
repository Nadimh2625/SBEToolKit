"""CLI: README figures — bias chart or null Type I calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

from sbetoolkit.calibration import empirical_power_check, type_i_null_check
from sbetoolkit.interference import diagnose_spatial_spillover
from sbetoolkit.marketplace import MarketplaceConfig, MarketplaceSimulator, summarize_estimators
from sbetoolkit.plots import plot_empirical_power, plot_naive_vs_switchback, plot_type_i
from sbetoolkit.randomization import spatial_buffer_cost


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


def _spillover(args: argparse.Namespace) -> int:
    # Slack control / tight treated so leftover drivers exist to leak.
    leaky = MarketplaceSimulator(
        MarketplaceConfig(
            n_regions=6,
            n_periods=16,
            riders_per_cell=40,
            drivers_per_cell=30,
            p_request_control=0.50,
            p_request_treat=0.95,
            driver_leakage=1.0,
            seed=args.seed,
        )
    )
    report = diagnose_spatial_spillover(
        leaky, n_reps=args.reps, seed=args.seed, cluster_size=3
    )
    cost = spatial_buffer_cost(
        n_regions=6,
        n_periods=16,
        cluster_sizes=(1, 2, 3),
        n_draws=200,
        seed=args.seed,
    )
    out = Path(args.out)
    csv_path = out.with_suffix(".csv")
    report.to_csv(csv_path, index=False)
    cost_path = csv_path.with_name(csv_path.stem + "_keep.csv")
    cost.to_csv(cost_path, index=False)
    print(f"wrote {csv_path}")
    print(report.to_string(index=False))
    print(f"wrote {cost_path}")
    print(cost.to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["chart", "null", "power", "spillover"], default="chart")
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
    if args.mode == "spillover":
        args.out = args.out or "docs/spatial_spillover.csv"
        args.reps = 200 if args.reps is None else args.reps
        args.seed = 3 if args.seed == 7 else args.seed
        return _spillover(args)
    args.out = args.out or "docs/empirical_power.png"
    args.reps = 500 if args.reps is None else args.reps
    args.seed = 13 if args.seed == 7 else args.seed
    return _power(args)


if __name__ == "__main__":
    raise SystemExit(main())
