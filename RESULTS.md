# Results: deep hedging under transaction costs

This report is generated from `results/full`. Monetary quantities are
dollars per option on one share. Lower entropic risk is better.

## Protocol and provenance

This is a fresh consolidated rerun, separate from recorded exploratory outputs.
Both architectures use one uniform validation-based stopping protocol, rather
than the original dense model's two-stage schedule. Exact metrics can differ
from the historical notebook because optimization is hardware-sensitive.

Training paths: 50,000; validation: 10,000;
fresh final paths: 30,000 (seed 909). Both
architectures report seeds [123, 456, 789, 1011]. The financial rule widths are
selected on validation data: reset 0.075, boundary
0.050 shares. No seed is selected by final performance.
Risk aversion is 1 per dollar; stock transaction cost is
0.10%. Opening and closing stock trades are charged.

## Final sample: every strategy

| strategy | risk | mean_pnl | std_pnl | pnl_5th_percentile | mean_cost | mean_gross_pnl |
| --- | --- | --- | --- | --- | --- | --- |
| no_hedge | 16.47316 | 0.01743 | 3.36574 | -7.15225 | 0.00000 | 0.01743 |
| half_share | 3.10118 | -0.08391 | 1.68613 | -3.30909 | 0.10000 | 0.01608 |
| delta | 0.35593 | -0.24488 | 0.44832 | -1.02030 | 0.24075 | -0.00413 |
| reset_band | 0.34489 | -0.21832 | 0.48261 | -1.03734 | 0.21441 | -0.00391 |
| boundary_band | 0.31844 | -0.18753 | 0.50403 | -1.00852 | 0.18595 | -0.00157 |
| dense_123 | 0.32796 | -0.20717 | 0.48730 | -0.97922 | 0.20477 | -0.00239 |
| dense_456 | 0.32783 | -0.20683 | 0.48937 | -0.97217 | 0.20447 | -0.00235 |
| dense_789 | 0.32683 | -0.20795 | 0.48491 | -0.96788 | 0.20551 | -0.00244 |
| dense_1011 | 0.32760 | -0.20736 | 0.48507 | -0.97826 | 0.20491 | -0.00245 |
| learned_band_123 | 0.31796 | -0.19253 | 0.50016 | -0.97490 | 0.19014 | -0.00239 |
| learned_band_456 | 0.31695 | -0.19474 | 0.49423 | -0.96385 | 0.19245 | -0.00229 |
| learned_band_789 | 0.31655 | -0.18855 | 0.50681 | -0.97505 | 0.18632 | -0.00223 |
| learned_band_1011 | 0.31884 | -0.18894 | 0.51022 | -0.98367 | 0.18676 | -0.00218 |

Average P&L, standard deviation and the 5th percentile describe different parts
of the outcome distribution; none alone is the training objective. The 5th
percentile is not a loss bound. Gross P&L excludes only stock transaction costs.

## Variation across training seeds

| architecture | mean | std | min | max |
| --- | --- | --- | --- | --- |
| dense | 0.32755 | 0.00050 | 0.32683 | 0.32796 |
| learned_band | 0.31758 | 0.00103 | 0.31655 | 0.31884 |

These are means and ranges of separate model scores. They are not pooled
outcomes, an ensemble, or independent market samples. Every model shares the
same training and evaluation paths, while initialization and batch order vary.

## Paired comparisons against the strong boundary rule

Difference = neural risk minus boundary-rule risk. Negative favors the neural
policy. Individual approximate 95% percentile intervals use
2,000 paired bootstrap resamples of the final paths.

| strategy | reference | risk_difference | ci_lower | ci_upper |
| --- | --- | --- | --- | --- |
| dense_123 | boundary_band | 0.00952 | 0.00638 | 0.01234 |
| dense_456 | boundary_band | 0.00939 | 0.00619 | 0.01222 |
| dense_789 | boundary_band | 0.00839 | 0.00528 | 0.01124 |
| dense_1011 | boundary_band | 0.00916 | 0.00605 | 0.01201 |
| learned_band_123 | boundary_band | -0.00048 | -0.00289 | 0.00186 |
| learned_band_456 | boundary_band | -0.00149 | -0.00381 | 0.00073 |
| learned_band_789 | boundary_band | -0.00189 | -0.00425 | 0.00032 |
| learned_band_1011 | boundary_band | 0.00040 | -0.00202 | 0.00266 |

For the learned-band architecture, 0 interval(s) favor the neural policy,
0 favor the boundary rule, and 4 include zero. These are
pointwise intervals with no multiple-comparison adjustment. They condition on
each trained model and the simulator; they do not measure uncertainty about
market realism or the training dataset.

## Volatility misspecification

| actual_volatility | delta | reset_band | boundary_band | dense | learned_band |
| --- | --- | --- | --- | --- | --- |
| 0.10000 | -0.86917 | -0.88524 | -0.88302 | -0.87687 | -0.88392 |
| 0.20000 | 0.35277 | 0.33986 | 0.31319 | 0.32366 | 0.31285 |
| 0.30000 | 2.03282 | 2.02539 | 1.82813 | 1.75531 | 1.79182 |
| 0.40000 | 4.99767 | 4.99341 | 4.32034 | 3.90810 | 4.21234 |

Neural columns average separate seed risk scores. Actual volatility changes;
assumed volatility, option premium, network parameters, costs and selected band
widths remain fixed. Scenarios share underlying random shocks and are not
independent. Compare strategies within each row. A high-volatility loss can
reflect an insufficient original premium, which stock hedging cannot repair
in expectation under zero drift. Extreme-loss-sensitive entropic estimates
can need larger simulation samples in stressed scenarios.

## Checkpoint selection

| architecture | seed | best_epoch | epochs_run | validation_risk |
| --- | --- | --- | --- | --- |
| dense | 123 | 97 | 100 | 0.32961 |
| dense | 456 | 99 | 100 | 0.32958 |
| dense | 789 | 98 | 100 | 0.32840 |
| dense | 1011 | 98 | 100 | 0.32909 |
| learned_band | 123 | 97 | 100 | 0.31861 |
| learned_band | 456 | 99 | 100 | 0.31753 |
| learned_band | 789 | 95 | 100 | 0.31808 |
| learned_band | 1011 | 86 | 96 | 0.31990 |

## Interpretation

The initial dense network was competitive with a reset-to-delta band but was
more expensive than a stronger boundary rule. An explicit learned no-trade
region largely closed the validation gap. The tables above show whether that
translated into a resolved final-sample advantage for each seed. A model that
matches a simple rule is not evidence of universal superiority.

The result demonstrates finance-aware architecture design, cost-aware training,
and the importance of strong baselines. It does not demonstrate tradable alpha,
live performance, or an optimal hedging policy.

## Limits and implementation conventions

Prices are simulated geometric Brownian motion. There are no jumps, stochastic
volatility, market impact, changing liquidity, financing costs or lot sizes.
Stress scenarios have a different constant volatility, not time-varying
volatility. Band widths come from a finite validation grid. Financial bands
open at delta; neural policies choose their opening holding, so initial-position
conventions are an additional difference. All models are constrained to
nonnegative holdings of at most one share.

Development used exploratory results before this final protocol was fixed.
The final seed is new; the project was not preregistered. Repeated seeds assess
optimizer variability on one training dataset. Hardware/framework changes can
alter results. These limits bound the interpretation of confidence intervals.

## Figures

Figures are in the run's `figures/` folder: risk and costs, training histories,
P&L distributions, volatility stresses, and learned boundaries. Representative
policy plots use the first declared seed (123), not the best seed.

## References

- Buehler et al., [Deep Hedging](https://arxiv.org/abs/1802.03042).
- Imaki et al., [No-Transaction Band Network](https://arxiv.org/abs/2103.01775).
