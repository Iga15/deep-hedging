"""Create figures from saved results; this never retrains or selects a model."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras

from hedging import call_delta


def make_figures(run_dir, config):
    run_dir = Path(run_dir)
    folder = run_dir / "figures"
    folder.mkdir(exist_ok=True)
    table = pd.read_csv(run_dir / "final_metrics.csv").set_index("strategy")
    neural = [name for name in table.index
              if name.startswith(("dense_", "learned_band_"))]
    names = ["delta", "reset_band", "boundary_band", *neural]
    colors = ["#888888" if n == "delta" else "#dd9645" if "band" in n and
              not n.startswith("learned") else "#2878a5" if n.startswith("dense")
              else "#34875f" for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, title in zip(axes, ["risk", "mean_cost"],
                                  ["Entropic risk (lower is better)",
                                   "Mean trading cost ($ per option)"]):
        ax.barh(names, table.loc[names, metric], color=colors)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Fresh confirmation sample: every training seed is shown")
    fig.tight_layout()
    fig.savefig(folder / "risk_and_costs.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    training = pd.read_csv(run_dir / "training_summary.csv")
    for _, run in training.iterrows():
        label = run.architecture
        history = pd.read_csv(run_dir / "models" /
                              f"{label}_seed_{run.seed}_history.csv")
        ax.plot(history.epoch, history.validation_risk,
                color="#2878a5" if label == "dense" else "#34875f", alpha=.65)
    ax.plot([], [], color="#2878a5", label="Dense network (all seeds)")
    ax.plot([], [], color="#34875f", label="Learned band (all seeds)")
    ax.set(xlabel="Epoch", ylabel="Validation entropic risk",
           title="Training progress; checkpoint selection uses validation only")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(folder / "training.png", dpi=160)
    plt.close(fig)

    outcomes = np.load(run_dir / "final_outcomes.npz")
    representative = config.seeds[0]  # Predeclared; never select using test performance.
    names = ["boundary_band", f"dense_{representative}", f"learned_band_{representative}"]
    all_pnl = np.concatenate([outcomes[n] for n in names])
    bins = np.linspace(all_pnl.min(), all_pnl.max(), 70)
    fig, ax = plt.subplots(figsize=(9, 4))
    for name in names:
        ax.hist(outcomes[name], bins=bins, density=True, histtype="step",
                 linewidth=1.6, label=name)
    ax.axvline(0, color="black", linestyle="--", alpha=.5)
    ax.set(xlabel="Final P&L ($ per option)", ylabel="Density",
           title=f"Outcome distributions (neural seed {representative}, predeclared)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "pnl_distribution.png", dpi=160)
    plt.close(fig)

    stress = pd.read_csv(run_dir / "stress_metrics.csv")
    fig, ax = plt.subplots(figsize=(9, 4))
    for name in ["delta", "reset_band", "boundary_band"]:
        rows = stress[stress.strategy == name]
        ax.plot(rows.actual_volatility * 100, rows.risk, marker="o", label=name)
    for architecture, color in [("dense", "#2878a5"), ("learned_band", "#34875f")]:
        rows = stress[stress.strategy.str.startswith(architecture + "_")]
        agg = rows.groupby("actual_volatility").risk.agg(["mean", "min", "max"])
        x = agg.index.to_numpy() * 100
        ax.plot(x, agg["mean"], marker="o", label=f"{architecture}: mean across seeds",
                color=color)
        ax.fill_between(x, agg["min"], agg["max"], alpha=.15, color=color)
    ax.set(xlabel="Actual annualized volatility (%)", ylabel="Entropic risk",
           title="Volatility misspecification: premium and strategy assumptions fixed")
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(folder / "volatility_stress.png", dpi=160)
    plt.close(fig)

    model = keras.models.load_model(
        run_dir / "models" / f"learned_band_seed_{representative}.keras")
    inspector = keras.Model(model.input,
                             [model.get_layer("lower_boundary").output,
                              model.get_layer("upper_boundary").output])
    import json
    widths = json.loads((run_dir / "selected_bands.json").read_text())
    grid = np.linspace(85, 115, 200)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, fraction in zip(axes, [.5, .25, .1]):
        state = np.column_stack([np.log(grid / config.strike),
                                 np.full_like(grid, fraction), np.zeros_like(grid)])
        lower, upper = inspector(state.astype(np.float32), training=False)
        delta = call_delta(grid, config.strike, config.volatility,
                           config.maturity * fraction)
        ax.fill_between(grid, lower.numpy().ravel(), upper.numpy().ravel(),
                         alpha=.3, color="#34875f", label="Learned no-trade region")
        ax.plot(grid, delta, "k--", label="Delta")
        ax.plot(grid, np.clip(delta - widths["boundary_band"], 0, 1),
                color="#dd9645", label="Fixed-band boundaries")
        ax.plot(grid, np.clip(delta + widths["boundary_band"], 0, 1), color="#dd9645")
        ax.set(title=f"{fraction:.0%} of time remaining", xlabel="Stock price ($)")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Stock holding (shares)")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Hypothetical state inspection; some states are rare in training")
    fig.tight_layout()
    fig.savefig(folder / "learned_boundaries.png", dpi=160)
    plt.close(fig)
    return sorted(folder.glob("*.png"))
