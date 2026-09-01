# SwitchbackExpToolkit

A coupon test on a shopping site can treat each person as their own little world. The same test on a rideshare or delivery network cannot: giving a discount to some riders lengthens the wait for everyone else still looking for a car. Ordinary A/B tests then measure *who won the scramble*, not *what happens if you launch the policy*, and they often report fake precision besides — every ride looks like a new data point even though you only flipped the city-hour a few hundred times.

This library is the toolkit for that setting. It turns a policy on and off for a whole city and time window at once ([switchback experiments](https://arxiv.org/abs/2009.00148)), plans how many of those windows you need, uses last week’s numbers to cut noise ([CUPED](https://doi.org/10.1145/2487575.2487651)), lets you peek at results without inflating false alarms ([alpha spending](https://doi.org/10.1093/biomet/70.3.659)), and checks the whole pipeline on a fake marketplace where the true effect is known. Uncertainty is counted at the city-hour you randomized, not at the ride ([clustered standard errors](https://doi.org/10.1093/qje/qjac038)).

**References.** Marketplace interference: [Blake & Coey (2014)](https://doi.org/10.1145/2566486.2567967); causal effects when people affect each other: [Hudgens & Halloran (2008)](https://doi.org/10.1198/016214508000000292). Switchback design and analysis: [Bojinov, Simchi-Levi & Zhao (2023)](https://doi.org/10.1287/mnsc.2022.4583). CUPED: [Deng, Xu, Kohavi & Walker (2013)](https://doi.org/10.1145/2487575.2487651). Sequential testing: [Lan & DeMets (1983)](https://doi.org/10.1093/biomet/70.3.659). When to cluster: [Abadie, Athey, Imbens & Wooldridge (2023)](https://doi.org/10.1093/qje/qjac038).

A 5% test analyzed at the ride level rejects a true-zero marketplace experiment **35% of the time**. That is what happens when you port ordinary product A/B tooling into a two-sided market: you randomized 500 time-region blocks and the SE pretends you have 41,000 independent rides. Cluster at the block and the same test rejects 4.7%. The point estimate is wrong for the same reason. Treated units steal scarce supply from control, so a naive rider split shows an 11pp match-rate win when launching the policy moves total matches by half a point. **Time-region switchbacks** with block-level SEs recover the global ATE and the nominal Type I rate.

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

## Proof the test is calibrated (true effect = 0)

A thousand independent switchbacks of a **null** policy: treated and control riders request at the same rate, no extra drivers. There is nothing to detect. A 5% test should light up about fifty times.

Each run randomizes **500** time-region blocks and records ~**41,000** rides. If you feed the rides to an iid t-test, the SE thinks those are 41,000 independent experiments. They are not. Treatment was assigned to the block.

![Null Type I calibration](docs/type_i_null.png)

| Estimator | n used in the SE | Reject H₀ (α = 0.05) |
| --- | ---: | ---: |
| iid SE at ride level | 40,997 rides | **35.1%** |
| Cluster-robust SE at the block | 500 blocks | **4.7%** |
| Difference in means on blocks | 500 blocks | **4.6%** |

35% false positives is the usual mistake when experimentation tooling is ported from a normal product context to a marketplace: you analyze at the individual-ride level after you randomized at the block. Cluster at the randomization unit, or aggregate to one row per block. Both land on the nominal 5%.

```bash
python -m sbetoolkit.cli --mode null --reps 1000 --seed 11
```

```python
from sbetoolkit import type_i_null_check

result = type_i_null_check(n_reps=1000, seed=11)
print(result.summary())
# iid 35.1%   cluster 4.7%   block 4.6%
```

Raw rates: `docs/type_i_null.csv`.

## Install

```bash
pip install -e ".[dev]"
python -m pytest
python -m sbetoolkit.cli --mode chart --out docs/naive_vs_switchback.png
python -m sbetoolkit.cli --mode null --reps 1000 --seed 11 --out docs/type_i_null.png
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
