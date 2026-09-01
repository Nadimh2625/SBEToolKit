# SwitchbackExpToolkit

Rider-level A/B tests in a two-sided market estimate the wrong thing. Treated units steal scarce supply from control (or spill extra supply onto them), so the contrast is not the effect of launching the policy. **Time-region switchbacks** put the whole local market on one policy at a time and recover the global ATE.

This library is the stack you actually need to run that experiment: power, randomization, CUPED, alpha spending, and an interference diagnostic with a simulated marketplace whose true effect is known.

## The chart

Eighty simulated launches of a demand-side treatment in a **supply-constrained** market. Ground truth is the difference in match rate between an all-treat world and an all-control world.

![Naive A/B vs switchback vs ground truth](docs/naive_vs_switchback.png)

| Estimator | Mean estimated lift in match rate |
| --- | ---: |
| Naive rider A/B | **+0.111** |
| Time-region switchback | **+0.003** |
| Ground truth | **+0.005** |

Naive A/B is precise and wrong: treated riders crowd out control riders for the same drivers, so the within-cell gap looks like an 11pp win. The launch does almost nothing for total matches — the market was already saturated. Switchback is noisier and unbiased.

That is the whole interview. The rest of the README is how to compute it.

## Install

```bash
pip install -e ".[dev]"
python -m pytest
python -m sbetoolkit.cli --out docs/naive_vs_switchback.png
```

## Marketplace interference

```python
from sbetoolkit import MarketplaceConfig, MarketplaceSimulator, diagnose_interference

sim = MarketplaceSimulator(MarketplaceConfig(
    n_regions=8, n_periods=24,
    riders_per_cell=40, drivers_per_cell=18,
    p_request_control=0.70, p_request_treat=0.90,
    seed=7,
))
print(sim.ground_truth())
print(diagnose_interference(sim).direction)
# → naive A/B overstates the global effect
```

The diagnostic splits the bias into **control crowding-out** (control match rate in a mixed market vs all-control) and **treated dilution** (treated match rate in a mixed market vs all-treat). Flip the story by setting `extra_drivers_if_treated > 0` with no demand lift: rider A/B never applies a cell-level supply incentive, so it **understates** the launch.

## Switchback randomization

```python
from sbetoolkit import assign_switchback

design = assign_switchback(
    regions=["sf", "nyc", "chi", "aus"],
    periods=range(28),
    design="balanced",   # or independent, blocked_random
    washout=1,           # drop re-equilibration periods after each switch
    seed=0,
)
design.table.head()
design.n_switches()
```

Blocks are `(region, period)`. Washout flags are on the assignment table; analysis should use `design.analysis_table`.

## Power / sample size

Variance of the switchback contrast is `w'Γw` with AR(1) outcomes, not the iid two-sample formula. Positive serial correlation **inflates** variance for long treated/control runs and **shrinks** it for alternating switches (neighbor pairing).

```python
from sbetoolkit import switchback_power, switchback_sample_size

needed = switchback_sample_size(
    delta=0.02, sigma=0.08, n_regions=8,
    rho_ar1=0.4, washout=1, power=0.8,
)
print(needed.n_periods, needed.power, needed.se)

print(switchback_power(0.02, 0.08, n_regions=8, n_periods=28, rho_ar1=0.4))
```

## CUPED

```python
from sbetoolkit.cuped import cuped_theta, difference_in_means

fit = cuped_theta(y, x_pre)          # θ = Cov(Y,X)/Var(X)
est = difference_in_means(y, treat, x=x_pre)
print(fit.variance_reduction, est["ate"], est["se"])
```

Fit `θ` on a locked pre-period if you will peek sequentially. Do not use the previous *experimental* period as `X` — that is post-treatment.

## Sequential tests (Lan–DeMets)

Boundaries solve the Brownian-motion crossing probabilities so total Type I error is `α`, not `α × n_looks`.

```python
from sbetoolkit import SequentialMonitor, spending_boundaries

spending_boundaries([0.25, 0.5, 0.75, 1.0], family="obrien_fleming")
# first look is strict; last look is near the usual 1.96

mon = SequentialMonitor(alpha=0.05, family="obrien_fleming",
                        planned_info=[0.5, 1.0])
mon.update(z=1.1, info_fraction=0.5)   # does not reject
mon.update(z=2.3, info_fraction=1.0)
```

Families: `obrien_fleming`, `pocock`, `kim_deMets` (power spending `α t^ρ`).

## Reproduce the chart

```python
from sbetoolkit import MarketplaceConfig, MarketplaceSimulator
from sbetoolkit.plots import plot_naive_vs_switchback

sim = MarketplaceSimulator(MarketplaceConfig(seed=7))
df = sim.compare_estimators(n_reps=80, seed=7)
plot_naive_vs_switchback(df, path="docs/naive_vs_switchback.png")
```

`compare_estimators` draws naive rider splits and switchback assignments on the same class of market primitives and compares both to a Monte Carlo global ATE.
