# SwitchbackExpToolkit

Normal A/B testing breaks in a rideshare or delivery marketplace, and this library is the set of tools that makes testing work there instead.

The problem, in plain terms is if you're testing a functionality such as button color on a shopping site, each user is basically in their own bubble. What happens to one person doesn't change what happens to anyone else. So you can split users in half, compare and be done.

A ridesharing app isn't like that. There's a limited number of drivers on the road. If you give half the riders a discount and they start requesting more rides, they take drivers away from the other half. So the discount group looks great and the no-discount group looks worse, but you haven't created a single extra ride. You just moved rides from one group to the other.

And essentially your test says "+11% wins!" so of course you launch it to everyone, and then you see total rides go up by half a percent. The test just simply measured who won a fight over a fixed pool, not whether the feature is worth shipping.

Also there is another issue about confidence; for example think of polling~ if you want to know what a city thinks, and you ask 500 different people, you will get 500 real opinions. If instead you ask 500 people but they're all sitting in the same room agreeing with each other, you don't have 500 opinions. You have a handful, repeated.

That's your rides. All the rides in an area f.e Brooklyn at 3pm are stuck together, because they're all fighting over the same drivers that hour. If drivers are scarce that hour, everybody in that hour has a bad time together. So those rides aren't separate pieces of information. They're one hour's worth of information.

To prove our test is calibrated we run an experiment where the feature does absolutely nothing, a thousand times over. A properly calibrated test should falsely declare victory about 5% of the time. The ride-level version declares victory 35% of the time so we would be shipping fake wins in a third of your experiments and have no idea.

This library is the toolkit for that setting. It turns a policy on and off for a whole city and time window at once ([switchback experiments](https://arxiv.org/abs/2009.00148)), plans how many of those windows you need, uses last week’s numbers to cut noise ([CUPED](https://doi.org/10.1145/2487575.2487651)), lets you peek at results without inflating false alarms ([alpha spending](https://doi.org/10.1093/biomet/70.3.659)), and checks the whole pipeline on a fake marketplace where the true effect is known. Uncertainty is counted at the city-hour you randomized, not at the ride ([clustered standard errors](https://doi.org/10.1093/qje/qjac038)).

**References.** Marketplace interference: [Blake & Coey (2014)](https://doi.org/10.1145/2566486.2567967); causal effects when people affect each other: [Hudgens & Halloran (2008)](https://doi.org/10.1198/016214508000000292). Switchback design and analysis: [Bojinov, Simchi-Levi & Zhao (2023)](https://doi.org/10.1287/mnsc.2022.4583). CUPED: [Deng, Xu, Kohavi & Walker (2013)](https://doi.org/10.1145/2487575.2487651). Sequential testing: [Lan & DeMets (1983)](https://doi.org/10.1093/biomet/70.3.659). When to cluster: [Abadie, Athey, Imbens & Wooldridge (2023)](https://doi.org/10.1093/qje/qjac038).

## The chart

Five hundred simulated launches of a demand-side treatment in a **supply-constrained** market. Ground truth is the difference in match rate between an all-treat world and an all-control world. Every estimate ships a 95% interval; coverage is how often that interval actually contains the truth.

![Naive A/B vs switchback vs ground truth](docs/naive_vs_switchback.png)

| Estimator | Mean | Bias | RMSE | 95% coverage |
| --- | ---: | ---: | ---: | ---: |
| Naive rider A/B | **+0.112** | **+0.107** | **0.107** | **0%** |
| Time-region switchback | **+0.005** | **+0.000** | **0.017** | **96%** |
| Ground truth | **+0.005** | — | — | — |

Naive A/B is precise and wrong: treated riders crowd out control riders for the same drivers, so the within-cell gap looks like an 11pp win. Its intervals are tight around that wrong number, so they almost never contain the launch effect. Switchback is noisier; the interval does the job a PM reads it for.

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

## Proof the power formula is calibrated (true effect ≠ 0)

The Type I check asks whether you cry wolf too often. This one asks whether you miss a real wolf. `switchback_sample_size` said **8 regions × 22 periods** (176 blocks) for 80% power at δ = 0.05, σ = 0.12, ρ = 0.4. Five hundred simulated switchbacks at *exactly that size*, with that effect present:

![Empirical power vs w′Γw prediction](docs/empirical_power.png)

| | Power |
| --- | ---: |
| Predicted (`w′Γw`) | **81.5%** |
| Detected (formula SE) | **81.2%** |
| Detected (Welch SE on analysis_table) | **79.4%** |
| Monte Carlo SD of ATE / formula SE | **0.98** |

If the variance formula were wrong you would see ~55% here: the test would be sized for a different experiment than the one you run. 81% vs 80% is the formula and the simulator agreeing.

```bash
python -m sbetoolkit.cli --mode power --reps 500 --seed 13
```

```python
from sbetoolkit import empirical_power_check, switchback_sample_size

needed = switchback_sample_size(0.05, 0.12, n_regions=8, rho_ar1=0.4, power=0.8)
print(needed.n_periods, needed.power)
print(empirical_power_check(n_reps=500, seed=13).summary())
```

Raw rates: `docs/empirical_power.csv`.

## Install

```bash
pip install -e ".[dev]"
python -m pytest
python -m sbetoolkit.cli --mode chart --out docs/naive_vs_switchback.png
python -m sbetoolkit.cli --mode null --reps 1000 --seed 11 --out docs/type_i_null.png
python -m sbetoolkit.cli --mode power --reps 500 --seed 13 --out docs/empirical_power.png
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
from sbetoolkit import assign_switchback, estimate_switchback

design = assign_switchback(
    regions=["sf", "nyc", "chi", "aus"],
    periods=range(28),
    design="balanced",   # or independent, blocked_random
    washout=1,           # drop re-equilibration periods after each switch
    seed=0,
)
design.analysis_table  # estimation sample; washout rows are gone
estimate_switchback(design, outcomes)  # joins on analysis_table, not table
```

Blocks are `(region, period)`. `design.table` is the full calendar (needed to simulate the world, including re-equilibration). Every estimator goes through `design.analysis_table`. Poison the washout rows and the ATE does not move.

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
from sbetoolkit import MarketplaceConfig, MarketplaceSimulator, summarize_estimators
from sbetoolkit.plots import plot_naive_vs_switchback

sim = MarketplaceSimulator(MarketplaceConfig(seed=7))
df = sim.compare_estimators(n_reps=500, seed=7)
print(summarize_estimators(df))
plot_naive_vs_switchback(df, path="docs/naive_vs_switchback.png")
```

`compare_estimators` draws naive rider splits and switchback assignments on the same class of market primitives and compares both to a Monte Carlo global ATE. Coverage is the share of 95% intervals that contain that ATE.
