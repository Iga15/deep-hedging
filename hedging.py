"""Deep hedging of one cash-settled European call, with proportional costs.

All monetary quantities are dollars per option on one share. Interest and
dividends are zero. This module contains no data downloads or trading APIs.
"""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
import platform
import time

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm
import tensorflow as tf
from tensorflow import keras


@dataclass(frozen=True)
class Config:
    initial_price: float = 100.0
    strike: float = 100.0
    volatility: float = 0.20
    annual_return: float = 0.0
    days: int = 20
    trading_days: int = 252
    cost_rate: float = 0.001
    risk_aversion: float = 1.0
    train_size: int = 50_000
    validation_size: int = 10_000
    final_size: int = 30_000
    stress_size: int = 20_000
    batch_size: int = 1024
    max_epochs: int = 100
    patience: int = 10
    min_improvement: float = 0.0001
    learning_rate: float = 0.001
    seeds: tuple = (123, 456, 789, 1011)
    train_seed: int = 101
    validation_seed: int = 202
    final_seed: int = 909
    stress_seed: int = 10001
    bootstrap_seed: int = 11001
    bootstrap_samples: int = 2000
    band_candidates: tuple = (0.0, 0.025, 0.05, 0.075, 0.10,
                              0.125, 0.15, 0.20, 0.30)
    stress_volatilities: tuple = (0.10, 0.20, 0.30, 0.40)

    @property
    def maturity(self):
        return self.days / self.trading_days

    @property
    def premium(self):
        return float(call_price(self.initial_price, self.strike,
                                self.volatility, self.maturity))

    def quick(self):
        """Integration check only; not enough training for research conclusions."""
        return replace(self, train_size=2048, validation_size=512,
                       final_size=1024, stress_size=512, max_epochs=3,
                       patience=3, seeds=(123,), bootstrap_samples=100)


def simulate_paths(n_paths, seed, config, volatility=None):
    sigma = config.volatility if volatility is None else volatility
    if n_paths <= 0 or config.days <= 0 or sigma <= 0:
        raise ValueError("Positive path count, days and volatility are required.")
    rng = np.random.default_rng(seed)
    dt = 1 / config.trading_days
    shocks = rng.standard_normal((n_paths, config.days))
    returns = ((config.annual_return - sigma**2 / 2) * dt
               + sigma * np.sqrt(dt) * shocks)
    paths = np.empty((n_paths, config.days + 1))
    paths[:, 0] = config.initial_price
    paths[:, 1:] = config.initial_price * np.exp(np.cumsum(returns, axis=1))
    return paths


def call_price(stock, strike, volatility, tau):
    """Black--Scholes call price, with zero interest/dividends and tau > 0."""
    d1 = (np.log(np.asarray(stock) / strike) + volatility**2 * tau / 2) / (
        volatility * np.sqrt(tau))
    return stock * norm.cdf(d1) - strike * norm.cdf(d1 - volatility * np.sqrt(tau))


def call_delta(stock, strike, volatility, tau):
    d1 = (np.log(np.asarray(stock) / strike) + volatility**2 * tau / 2) / (
        volatility * np.sqrt(tau))
    return norm.cdf(d1)


def delta_targets(paths, config):
    tau = np.arange(config.days, 0, -1) / config.trading_days
    return call_delta(paths[:, :-1], config.strike, config.volatility, tau)


def band_holdings(deltas, width, boundary=False):
    """Both financial rules establish delta at day zero, as in the source notebook."""
    if width < 0:
        raise ValueError("Band width must be nonnegative.")
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.ndim != 2 or deltas.shape[1] == 0:
        raise ValueError("Expected [paths, pre-expiry times] delta targets.")
    holdings = np.empty_like(deltas)
    holdings[:, 0] = deltas[:, 0]
    for day in range(1, deltas.shape[1]):
        previous = holdings[:, day - 1]
        target = deltas[:, day]
        if boundary:
            holdings[:, day] = np.clip(previous, np.maximum(0, target - width),
                                       np.minimum(1, target + width))
        else:
            holdings[:, day] = np.where(np.abs(target - previous) > width,
                                        target, previous)
    return holdings


def account(paths, holdings, config):
    """Self-financing P&L and costs, including initial purchase and liquidation.

    With zero interest the cash ledger equals premium + stock gains - payoff
    - costs. No capital injection or free stock purchases are assumed.
    """
    paths = np.asarray(paths, dtype=np.float64)
    holdings = np.asarray(holdings, dtype=np.float64)
    if paths.ndim != 2 or holdings.shape != (len(paths), paths.shape[1] - 1):
        raise ValueError("Holdings must have one entry per pre-expiry price.")
    if not np.all(np.isfinite(paths)) or not np.all(np.isfinite(holdings)):
        raise ValueError("Prices and holdings must be finite.")
    trades = np.diff(holdings, axis=1, prepend=np.zeros((len(paths), 1)))
    turnover = (np.sum(np.abs(trades) * paths[:, :-1], axis=1)
                + np.abs(holdings[:, -1]) * paths[:, -1])
    costs = config.cost_rate * turnover
    gains = np.sum(holdings * np.diff(paths, axis=1), axis=1)
    payoff = np.maximum(paths[:, -1] - config.strike, 0)
    gross = config.premium + gains - payoff
    return {"pnl": gross - costs, "gross_pnl": gross,
            "costs": costs, "turnover": turnover}


def entropic_risk(pnl, gamma=1.0):
    if gamma <= 0:
        raise ValueError("Risk aversion must be positive.")
    pnl = np.asarray(pnl, dtype=np.float64).ravel()
    if not len(pnl) or not np.all(np.isfinite(pnl)):
        raise ValueError("P&L must be nonempty and finite.")
    return float((logsumexp(-gamma * pnl) - np.log(len(pnl))) / gamma)


def tensor_risk(pnl, gamma):
    n = tf.cast(tf.size(pnl), pnl.dtype)
    return (tf.reduce_logsumexp(-gamma * pnl) - tf.math.log(n)) / gamma


def build_policy(architecture, seed):
    keras.utils.set_random_seed(seed)
    state = keras.Input(shape=(3,), name="market_state")
    if architecture == "dense":
        x = keras.layers.Dense(32, activation="tanh")(state)
        x = keras.layers.Dense(32, activation="tanh")(x)
        target = keras.layers.Dense(1, activation="sigmoid",
                                    name="target_holding")(x)
    elif architecture == "learned_band":
        x = keras.layers.Dense(32, activation="tanh")(state[:, :2])
        x = keras.layers.Dense(32, activation="tanh")(x)
        endpoints = keras.layers.Dense(2, activation="sigmoid",
                                       name="predicted_endpoints")(x)
        a, b = endpoints[:, :1], endpoints[:, 1:2]
        lower = keras.layers.Minimum(name="lower_boundary")([a, b])
        upper = keras.layers.Maximum(name="upper_boundary")([a, b])
        above = keras.layers.Maximum()([state[:, 2:3], lower])
        target = keras.layers.Minimum(name="target_holding")([above, upper])
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    return keras.Model(state, target, name=f"{architecture}_seed_{seed}")


def policy_rollout(paths, policy, config, training=False):
    prices = tf.convert_to_tensor(paths, dtype=tf.float32)
    batch = tf.shape(prices)[0]
    cash = tf.fill([batch], tf.cast(config.premium, tf.float32))
    shares = tf.zeros([batch], dtype=tf.float32)
    history = []
    for day in range(config.days):
        price = prices[:, day]
        state = tf.stack([
            tf.math.log(price / config.strike),
            tf.fill([batch], tf.cast((config.days - day) / config.days, tf.float32)),
            shares,
        ], axis=1)
        target = policy(state, training=training)[:, 0]
        trade = target - shares
        cash = cash - trade * price - config.cost_rate * tf.abs(trade) * price
        shares = target
        history.append(shares)
    final = prices[:, -1]
    cash = cash + shares * final - config.cost_rate * tf.abs(shares) * final
    pnl = cash - tf.maximum(final - config.strike, 0.0)
    return pnl, tf.stack(history, axis=1)


def train_policy(architecture, seed, train_paths, validation_paths, config, folder):
    model = build_policy(architecture, seed)
    optimizer = keras.optimizers.Adam(config.learning_rate, clipnorm=1.0)
    data = (tf.data.Dataset.from_tensor_slices(train_paths.astype(np.float32))
            .shuffle(len(train_paths), seed=seed + 10000,
                     reshuffle_each_iteration=True)
            .batch(config.batch_size).prefetch(tf.data.AUTOTUNE))
    spec = [tf.TensorSpec((None, config.days + 1), tf.float32)]

    @tf.function(input_signature=spec)
    def step(batch):
        with tf.GradientTape() as tape:
            pnl, _ = policy_rollout(batch, model, config, training=True)
            loss = tf.debugging.check_numerics(
                tensor_risk(pnl, config.risk_aversion), "Nonfinite training loss")
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    @tf.function(input_signature=spec)
    def validate(batch):
        pnl, _ = policy_rollout(batch, model, config)
        return tensor_risk(pnl, config.risk_aversion)

    val = tf.convert_to_tensor(validation_paths, tf.float32)
    best = float(validate(val).numpy())
    weights = model.get_weights()
    best_epoch, stale = 0, 0
    records = [{"epoch": 0, "validation_risk": best}]
    started = time.monotonic()
    for epoch in range(1, config.max_epochs + 1):
        for batch in data:
            step(batch)
        score = float(validate(val).numpy())
        if not np.isfinite(score):
            raise FloatingPointError("Nonfinite validation score")
        records.append({"epoch": epoch, "validation_risk": score})
        if score < best - config.min_improvement:
            best, weights, best_epoch, stale = score, model.get_weights(), epoch, 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"{architecture} seed={seed} epoch={epoch:03d} "
                  f"validation={score:.5f}", flush=True)
        if stale >= config.patience:
            break
    model.set_weights(weights)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / f"{architecture}_seed_{seed}.keras"
    model.save(checkpoint)
    pd.DataFrame(records).to_csv(folder / f"{architecture}_seed_{seed}_history.csv",
                                 index=False)
    meta = {"architecture": architecture, "seed": seed, "best_epoch": best_epoch,
            "epochs_run": epoch, "validation_risk": best,
            "seconds": time.monotonic() - started, "checkpoint": checkpoint.name}
    print(f"Saved {checkpoint.name}: epoch {best_epoch}, risk {best:.5f}", flush=True)
    return model, meta


def select_bands(paths, config):
    deltas = delta_targets(paths, config)
    rows = []
    for boundary in (False, True):
        name = "boundary_band" if boundary else "reset_band"
        for width in config.band_candidates:
            values = account(paths, band_holdings(deltas, width, boundary), config)
            rows.append({"strategy": name, "width": width,
                         "validation_risk": entropic_risk(values["pnl"],
                                                           config.risk_aversion)})
    table = pd.DataFrame(rows)
    selected = table.loc[table.groupby("strategy")["validation_risk"].idxmin()]
    return table, dict(zip(selected.strategy, selected.width))


def metrics(name, values, config):
    pnl = values["pnl"]
    return {"strategy": name, "risk": entropic_risk(pnl, config.risk_aversion),
            "mean_pnl": np.mean(pnl), "std_pnl": np.std(pnl),
            "pnl_5th_percentile": np.percentile(pnl, 5),
            "mean_cost": np.mean(values["costs"]),
            "mean_gross_pnl": np.mean(values["gross_pnl"])}


def evaluate_all(paths, models, widths, config):
    deltas = delta_targets(paths, config)
    fixed = {"no_hedge": np.zeros_like(deltas),
             "half_share": np.full_like(deltas, 0.5), "delta": deltas,
             "reset_band": band_holdings(deltas, widths["reset_band"]),
             "boundary_band": band_holdings(deltas, widths["boundary_band"], True)}
    results = {name: account(paths, holds, config) for name, holds in fixed.items()}
    for name, model in models.items():
        # Evaluate in batches to bound memory even when final samples are large.
        holdings = []
        for start in range(0, len(paths), 4096):
            _, h = policy_rollout(paths[start:start + 4096].astype(np.float32),
                                  model, config)
            holdings.append(h.numpy())
        # Score all strategies in float64 with the same accounting implementation.
        results[name] = account(paths, np.concatenate(holdings), config)
    table = pd.DataFrame([metrics(name, values, config)
                          for name, values in results.items()])
    return table, results


def paired_bootstrap(a, b, gamma, samples=2000, seed=11001):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 2:
        raise ValueError("Expected equal-length one-dimensional paired outcomes.")
    rng = np.random.default_rng(seed)
    differences = np.empty(samples)
    for i in range(samples):
        indices = rng.integers(0, len(a), len(a))
        differences[i] = entropic_risk(a[indices], gamma) - entropic_risk(b[indices], gamma)
    lower, upper = np.percentile(differences, [2.5, 97.5])
    return {"risk_difference": entropic_risk(a, gamma) - entropic_risk(b, gamma),
            "ci_lower": lower, "ci_upper": upper}


def run_experiment(config, output_dir, profile="full"):
    """Frozen experiment: select on validation, then score every seed on fresh paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise FileExistsError(f"Completed run already exists: {out}. Choose a new folder.")
    if list((out / "models").glob("*.keras")):
        raise FileExistsError(f"Checkpoints already exist: {out}. Choose a new folder.")
    (out / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    train = simulate_paths(config.train_size, config.train_seed, config)
    validation = simulate_paths(config.validation_size, config.validation_seed, config)
    search, widths = select_bands(validation, config)
    search.to_csv(out / "band_search.csv", index=False)
    (out / "selected_bands.json").write_text(json.dumps(widths, indent=2) + "\n")
    print(f"Validation-selected widths: {widths}", flush=True)
    models, training = {}, []
    for architecture in ("dense", "learned_band"):
        for seed in config.seeds:
            name = f"{architecture}_{seed}"
            models[name], meta = train_policy(architecture, seed, train, validation,
                                              config, out / "models")
            training.append(meta)
            pd.DataFrame(training).to_csv(out / "training_summary.csv", index=False)
    # Final paths are generated only after training and validation selection finish.
    final = simulate_paths(config.final_size, config.final_seed, config)
    table, values = evaluate_all(final, models, widths, config)
    table.to_csv(out / "final_metrics.csv", index=False)
    np.savez_compressed(out / "final_outcomes.npz",
                        **{name: v["pnl"] for name, v in values.items()})
    intervals = []
    for name in models:
        interval = paired_bootstrap(values[name]["pnl"], values["boundary_band"]["pnl"],
                                    config.risk_aversion, config.bootstrap_samples,
                                    config.bootstrap_seed)
        intervals.append({"strategy": name, "reference": "boundary_band", **interval})
    pd.DataFrame(intervals).to_csv(out / "paired_intervals.csv", index=False)
    stress_tables = []
    for actual_vol in config.stress_volatilities:
        stress = simulate_paths(config.stress_size, config.stress_seed,
                                config, volatility=actual_vol)
        stress_table, _ = evaluate_all(stress, models, widths, config)
        stress_table.insert(0, "actual_volatility", actual_vol)
        stress_tables.append(stress_table)
    pd.concat(stress_tables).to_csv(out / "stress_metrics.csv", index=False)
    summary = table[table.strategy.isin(models)].copy()
    summary["architecture"] = summary.strategy.str.rsplit("_", n=1).str[0]
    summary.groupby("architecture").risk.agg(["mean", "std", "min", "max"]).to_csv(
        out / "seed_summary.csv")
    manifest = {"profile": profile, "python": platform.python_version(),
                "tensorflow": tf.__version__, "keras": keras.__version__,
                "numpy": np.__version__, "pandas": pd.__version__,
                "final_seed": config.final_seed, "models": list(models),
                "status": "complete", "note": "Simulation study; all seeds reported. "
                "Quick profile is an integration check, not a research result."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(table.to_string(index=False), flush=True)
    return table
