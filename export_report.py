"""Export figures and a Markdown report from an existing completed run."""

import argparse
import json
from pathlib import Path

import pandas as pd

from hedging import Config
from reporting import make_figures


def markdown_table(frame, digits=5):
    def fmt(value):
        return f"{value:.{digits}f}" if isinstance(value, float) else str(value)
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *rows])


def export_report(run_dir, destination):
    run_dir, destination = Path(run_dir), Path(destination)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    config = Config(**json.loads((run_dir / "config.json").read_text()))
    make_figures(run_dir, config)
    table = pd.read_csv(run_dir / "final_metrics.csv")
    intervals = pd.read_csv(run_dir / "paired_intervals.csv")
    seeds = pd.read_csv(run_dir / "seed_summary.csv")
    training = pd.read_csv(run_dir / "training_summary.csv")
    widths = json.loads((run_dir / "selected_bands.json").read_text())
    stress = pd.read_csv(run_dir / "stress_metrics.csv")
    # Mean of separate seed scores, not a pooled P&L distribution or ensemble.
    stress["group"] = stress.strategy.str.replace(r"_\d+$", "", regex=True)
    stress_summary = stress.groupby(["actual_volatility", "group"]).risk.mean().unstack()
    stress_summary = stress_summary[["delta", "reset_band", "boundary_band",
                                     "dense", "learned_band"]].reset_index()
    learned_intervals = intervals[intervals.strategy.str.startswith("learned_band_")]
    favor = int((learned_intervals.ci_upper < 0).sum())
    against = int((learned_intervals.ci_lower > 0).sum())
    unresolved = len(learned_intervals) - favor - against
    prefix = "" if manifest["profile"] == "full" else (
        "> **QUICK INTEGRATION CHECK: these are not research results.**\n\n")
    text = f"""# Results: deep hedging under transaction costs

{prefix}This report is generated from `{run_dir.as_posix()}`. Monetary quantities are
dollars per option on one share. Lower entropic risk is better.

## Protocol and provenance

This is a fresh consolidated rerun, separate from recorded exploratory outputs.
Both architectures use one uniform validation-based stopping protocol, rather
than the original dense model's two-stage schedule. Exact metrics can differ
from the historical notebook because optimization is hardware-sensitive.

Training paths: {config.train_size:,}; validation: {config.validation_size:,};
fresh final paths: {config.final_size:,} (seed {config.final_seed}). Both
architectures report seeds {list(config.seeds)}. The financial rule widths are
selected on validation data: reset {widths['reset_band']:.3f}, boundary
{widths['boundary_band']:.3f} shares. No seed is selected by final performance.
Risk aversion is {config.risk_aversion:g} per dollar; stock transaction cost is
{config.cost_rate:.2%}. Opening and closing stock trades are charged.

## Final sample: every strategy

{markdown_table(table)}

Average P&L, standard deviation and the 5th percentile describe different parts
of the outcome distribution; none alone is the training objective. The 5th
percentile is not a loss bound. Gross P&L excludes only stock transaction costs.

## Variation across training seeds

{markdown_table(seeds)}

These are means and ranges of separate model scores. They are not pooled
outcomes, an ensemble, or independent market samples. Every model shares the
same training and evaluation paths, while initialization and batch order vary.

## Paired comparisons against the strong boundary rule

Difference = neural risk minus boundary-rule risk. Negative favors the neural
policy. Individual approximate 95% percentile intervals use
{config.bootstrap_samples:,} paired bootstrap resamples of the final paths.

{markdown_table(intervals)}

For the learned-band architecture, {favor} interval(s) favor the neural policy,
{against} favor the boundary rule, and {unresolved} include zero. These are
pointwise intervals with no multiple-comparison adjustment. They condition on
each trained model and the simulator; they do not measure uncertainty about
market realism or the training dataset.

## Volatility misspecification

{markdown_table(stress_summary)}

Neural columns average separate seed risk scores. Actual volatility changes;
assumed volatility, option premium, network parameters, costs and selected band
widths remain fixed. Scenarios share underlying random shocks and are not
independent. Compare strategies within each row. A high-volatility loss can
reflect an insufficient original premium, which stock hedging cannot repair
in expectation under zero drift. Extreme-loss-sensitive entropic estimates
can need larger simulation samples in stressed scenarios.

## Checkpoint selection

{markdown_table(training[['architecture', 'seed', 'best_epoch', 'epochs_run', 'validation_risk']])}

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
policy plots use the first declared seed ({config.seeds[0]}), not the best seed.

## References

- Buehler et al., [Deep Hedging](https://arxiv.org/abs/1802.03042).
- Imaki et al., [No-Transaction Band Network](https://arxiv.org/abs/2103.01775).
"""
    destination.write_text(text)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    destination = args.report or args.run_dir / "REPORT.md"
    print(export_report(args.run_dir, destination))
