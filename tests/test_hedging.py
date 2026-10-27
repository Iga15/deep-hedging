"""Financial and integration checks; no model-performance assertions."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
from tensorflow import keras

from hedging import (Config, account, band_holdings, build_policy, call_delta,
                     entropic_risk, paired_bootstrap, policy_rollout, simulate_paths,
                     tensor_risk)


class FinancialChecks(unittest.TestCase):
    def setUp(self):
        self.config = Config()

    def test_cash_ledger_equivalence(self):
        paths = np.array([[100., 110., 105., 130.]])
        # Use the actual model premium; the tutorial's $5 example shifts this by
        # the difference between $5 and that premium.
        for holdings, stock_gain in [([0, 0, 0], 0),
                                     ([.5, .5, .5], 15),
                                     ([.5, .65, .55], 15.5)]:
            c = replace(self.config, cost_rate=0)
            result = account(paths, np.array([holdings]), c)
            self.assertAlmostEqual(result["pnl"][0], c.premium + stock_gain - 30)

    def test_costs_include_both_endpoints(self):
        paths = np.array([[100., 110., 105., 130.]])
        result = account(paths, np.array([[.5, .5, .5]]), self.config)
        self.assertAlmostEqual(result["costs"][0], .001 * (50 + 65))
        self.assertAlmostEqual(result["pnl"][0], result["gross_pnl"][0] - .115)

    def test_market_and_risk(self):
        a = simulate_paths(16, 12, self.config)
        np.testing.assert_array_equal(a, simulate_paths(16, 12, self.config))
        self.assertTrue(np.all(a > 0))
        np.testing.assert_array_equal(a[:, 0], 100)
        self.assertAlmostEqual(self.config.premium, 2.24749, places=4)
        delta = call_delta(100., 100., .2, 20 / 252)
        self.assertTrue(.5 < delta < .52)
        self.assertAlmostEqual(entropic_risk(np.array([-.2, -.2])), .2)
        x = np.array([-3., -1., 0., 2.])
        self.assertAlmostEqual(entropic_risk(x + 5), entropic_risk(x) - 5)
        self.assertAlmostEqual(float(tensor_risk(x.astype(np.float32), 1)),
                               entropic_risk(x), places=5)

    def test_bands_and_zero_width(self):
        targets = np.array([[.5, .65, .58]])
        np.testing.assert_allclose(band_holdings(targets, .075, True),
                                   [[.5, .575, .575]])
        for boundary in [False, True]:
            np.testing.assert_array_equal(band_holdings(targets, 0, boundary), targets)

    def test_tensor_accounting_and_no_lookahead(self):
        paths = simulate_paths(8, 17, self.config).astype(np.float32)
        for architecture in ["dense", "learned_band"]:
            model = build_policy(architecture, 123)
            pnl, holdings = policy_rollout(paths, model, self.config)
            expected = account(paths, holdings.numpy(), self.config)["pnl"]
            np.testing.assert_allclose(pnl.numpy(), expected, atol=1e-4, rtol=0)
            changed = paths.copy()
            changed[:, 11:] *= 1.1
            _, other_holdings = policy_rollout(changed, model, self.config)
            np.testing.assert_array_equal(holdings.numpy()[:, :11],
                                          other_holdings.numpy()[:, :11])
            self.assertTrue(np.all((holdings.numpy() >= 0) & (holdings.numpy() <= 1)))

    def test_learned_band_inaction_and_serialization(self):
        model = build_policy("learned_band", 456)
        inspector = keras.Model(model.input, [model.get_layer("lower_boundary").output,
                                               model.get_layer("upper_boundary").output])
        state = np.array([[0., .5, 0.]], dtype=np.float32)
        lower, upper = inspector(state)
        state[0, 2] = (lower.numpy()[0, 0] + upper.numpy()[0, 0]) / 2
        np.testing.assert_allclose(model(state).numpy()[0, 0], state[0, 2], atol=1e-7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.keras"
            model.save(path)
            restored = keras.models.load_model(path)
            np.testing.assert_allclose(model(state).numpy(), restored(state).numpy())

    def test_paired_bootstrap_respects_identical_outcomes(self):
        x = np.array([-2., -1., 0., 1., 2.])
        result = paired_bootstrap(x, x, 1, samples=30, seed=123)
        for value in result.values():
            self.assertEqual(value, 0)


if __name__ == "__main__":
    unittest.main()
