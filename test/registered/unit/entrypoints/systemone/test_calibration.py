"""Unit tests for System One reads and calibration: read plans, combining reads, fitting, config."""

import json
import math
import random
import unittest

import msgspec

from sglang.srt.entrypoints.systemone.calibration import (
    BUILTIN_MAX_CHOICE_ORDERS,
    RAW_READS,
    ReadSetup,
    apply_calibration,
    choice_order_count,
    choice_orders,
    combine_reads,
    decode_config,
    identity_read,
    kendall_distance,
    label_variants,
    load_config,
    log_normalize,
    plan_reads,
    read_log_probabilities,
    reads_fingerprint,
)
from sglang.srt.entrypoints.systemone.calibration_fit import (
    auroc,
    choose_calibration,
    cross_validated_nll,
    decidable_share,
    expected_calibration_error,
    fit_platt,
    fit_temperature,
    fit_vector,
    nll,
    quadratic_weighted_kappa,
    ranked_probability_score,
    selective_accuracy,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

LABEL_FREE = ReadSetup(
    choice_orders="williams",
    choice_max_orders=4,
    choice_name_variants=True,
    noul_orders=2,
    noul_case_variants=True,
)


class TestReadPlans(CustomTestCase):
    def test_kendall_distance_counts_discordant_pairs(self):
        rng = random.Random(0)
        for _ in range(200):
            n = rng.randint(1, 8)
            a, b = list(range(n)), list(range(n))
            rng.shuffle(a)
            rng.shuffle(b)
            brute = sum(
                (a.index(x) - a.index(y)) * (b.index(x) - b.index(y)) < 0
                for x in range(n)
                for y in range(x + 1, n)
            )
            self.assertEqual(kendall_distance(a, b), brute)

    def test_full_designs_balance_positions_and_williams_neighbors(self):
        for n in range(1, 9):
            for scheme in ("rotations", "williams"):
                with self.subTest(n=n, scheme=scheme):
                    orders = choice_orders(scheme, n, n)
                    self.assertEqual(orders[0], tuple(range(n)))
                    self.assertEqual(len(set(orders)), n)
                    # Every option in every position once.
                    for column in zip(*orders):
                        self.assertEqual(sorted(column), list(range(n)))
                    pairs = {(o[i], o[i + 1]) for o in orders for i in range(n - 1)}
                    if scheme == "williams" and n % 2 == 0:
                        self.assertEqual(len(pairs), n * (n - 1))
                    if scheme == "rotations" and n > 2:
                        # Each option always has the same neighbors.
                        self.assertEqual(len(pairs), n)

    def test_partial_sets_are_the_most_discordant_and_nested(self):
        # The most discordant second order: a half shift, or the full reversal.
        self.assertEqual(choice_orders("rotations", 8, 2)[1], (4, 5, 6, 7, 0, 1, 2, 3))
        self.assertEqual(choice_orders("williams", 6, 2)[1], (5, 4, 3, 2, 1, 0))
        for scheme in ("rotations", "williams"):
            eight = choice_orders(scheme, 77, 8)
            for k in (1, 2, 4):
                self.assertEqual(choice_orders(scheme, 77, k), eight[:k])
            nearest = [
                min(kendall_distance(o, c) for c in eight[:i])
                for i, o in enumerate(eight)
                if i
            ]
            # Greedy growth: each added order is at least as far as any later one.
            self.assertEqual(nearest, sorted(nearest, reverse=True))

    def test_order_counts_follow_the_max(self):
        def setup(max_orders):
            return msgspec.structs.replace(LABEL_FREE, choice_max_orders=max_orders)

        self.assertEqual(choice_order_count(setup("all"), 5), 5)
        self.assertEqual(choice_order_count(setup(20), 3), 3)
        self.assertEqual(choice_order_count(setup(4), 77), 4)
        reads = plan_reads("choice", ["A", "B", "C"], setup("all"))
        self.assertEqual(
            [r.order for r in reads], list(choice_orders("williams", 3, 3))
        )
        self.assertTrue(all(r.labels == ("A", "B", "C") for r in reads))

    def test_raw_reads_are_the_single_decisions_read(self):
        for kind, labels in (
            ("choice", ["A", "B", "C"]),
            ("yes_no", ["yes", "no"]),
            ("score", ["0", "1"]),
        ):
            with self.subTest(kind):
                self.assertEqual(
                    plan_reads(kind, labels, RAW_READS), [identity_read(labels)]
                )
                self.assertEqual(label_variants(kind, labels[0], "name", RAW_READS), [])

    def test_noul_orders_swap_the_labels_shown(self):
        reads = plan_reads("yes_no", ["yes", "no"], LABEL_FREE)
        self.assertEqual(reads[0], identity_read(["yes", "no"]))
        self.assertEqual((reads[1].order, reads[1].labels), ((1, 0), ("no", "yes")))

    def test_score_levels_are_never_permuted(self):
        reads = plan_reads("score", ["0", "1", "2"], LABEL_FREE)
        self.assertEqual(reads, [identity_read(["0", "1", "2"])])

    def test_label_variants(self):
        self.assertEqual(
            label_variants("yes_no", "yes", "yes", LABEL_FREE), ["Yes", "YES"]
        )
        self.assertEqual(
            label_variants("choice", "A", "billing", LABEL_FREE), ["billing", "Billing"]
        )
        # A name equal to its label adds nothing.
        self.assertEqual(label_variants("choice", "A", "A", LABEL_FREE), [])
        self.assertEqual(label_variants("score", "0", "0", LABEL_FREE), [])


class TestCombining(CustomTestCase):
    def test_variants_add_up_within_an_option(self):
        log_q = read_log_probabilities(
            [[math.log(0.1), math.log(0.3)], [math.log(0.2)]]
        )
        self.assertAlmostEqual(math.exp(log_q[0]), 0.4 / 0.6)
        self.assertAlmostEqual(math.exp(log_q[1]), 0.2 / 0.6)

    def test_reads_combine_by_geometric_mean(self):
        a = log_normalize([math.log(0.8), math.log(0.2)])
        b = log_normalize([math.log(0.2), math.log(0.8)])
        # Opposite reads cancel to uniform, as an order bias should.
        self.assertTrue(
            all(abs(math.exp(v) - 0.5) < 1e-12 for v in combine_reads([a, b]))
        )
        c = log_normalize([math.log(0.9), math.log(0.1)])
        combined = combine_reads([a, c])
        expected = [math.sqrt(0.8 * 0.9), math.sqrt(0.2 * 0.1)]
        self.assertAlmostEqual(math.exp(combined[0]), expected[0] / sum(expected))

    def test_one_zero_read_does_not_veto_an_option(self):
        zero = [0.0, -math.inf]
        other = log_normalize([math.log(0.1), math.log(0.9)])
        combined = combine_reads([zero, other])
        self.assertTrue(all(math.isfinite(v) for v in combined))
        self.assertGreater(math.exp(combined[1]), 0.0)


class TestFitting(CustomTestCase):
    def _synthetic(self, temperature, n=2000, seed=0):
        """Rows whose true probabilities are the logits divided by temperature."""
        rng = random.Random(seed)
        log_q, gold = [], []
        for _ in range(n):
            logits = [rng.gauss(0, 3) for _ in range(4)]
            true = log_normalize([v / temperature for v in logits])
            weights = [math.exp(v) for v in true]
            gold.append(rng.choices(range(4), weights)[0])
            log_q.append(log_normalize(logits))
        return log_q, gold

    def test_temperature_fit_recovers_the_true_temperature(self):
        log_q, gold = self._synthetic(temperature=2.5)
        fitted = fit_temperature(log_q, gold)
        self.assertAlmostEqual(fitted["temperature"], 2.5, delta=0.3)
        before, after = cross_validated_nll(log_q, gold, fit_temperature, folds=5)
        self.assertLess(after, before)

    def test_platt_fit_recovers_scale_and_offset(self):
        rng = random.Random(1)
        log_q, gold = [], []
        for _ in range(4000):
            z = rng.gauss(0, 2)
            p = 1 / (1 + math.exp(-(0.5 * z + 0.7)))
            gold.append(0 if rng.random() < p else 1)
            log_q.append(log_normalize([z, 0.0]))
        params = fit_platt(log_q, gold)
        self.assertAlmostEqual(params["a"], 0.5, delta=0.1)
        self.assertAlmostEqual(params["b"], 0.7, delta=0.15)

    def test_platt_fit_converges_from_saturated_predictions(self):
        # Every answer says yes with high confidence while half are no: undamped
        # Newton steps from there diverged to huge parameters.
        rng = random.Random(2)
        log_q, gold = [], []
        for _ in range(300):
            z = rng.uniform(2.0, 7.0)
            log_q.append(log_normalize([z, 0.0]))
            gold.append(rng.randrange(2))
        params = fit_platt(log_q, gold)
        self.assertLess(abs(params["a"]), 1.0)
        fitted = [apply_calibration(params, row) for row in log_q]
        self.assertLess(nll(fitted, gold), math.log(2) + 0.01)
        self.assertLess(nll(fitted, gold), nll(log_q, gold))

    def test_vector_fit_removes_a_per_option_bias(self):
        # The model overrates option 0 by a constant; a temperature cannot undo it.
        rng = random.Random(3)
        log_q, gold = [], []
        for _ in range(1500):
            logits = [rng.gauss(0, 2) for _ in range(4)]
            weights = [math.exp(v) for v in log_normalize(logits)]
            gold.append(rng.choices(range(4), weights)[0])
            log_q.append(log_normalize([logits[0] + 2] + logits[1:]))
        vector = fit_vector(log_q, gold)
        self.assertLess(vector["bias"][0], min(vector["bias"][1:]) - 1.0)
        calibration, report = choose_calibration("choice", log_q, gold, folds=5)
        self.assertEqual(calibration["type"], "vector")
        self.assertLess(report["vector"], report["temperature"])

    def test_nothing_is_chosen_when_answers_are_already_calibrated(self):
        log_q, gold = self._synthetic(temperature=1.0, n=600, seed=4)
        calibration, report = choose_calibration("choice", log_q, gold, folds=5)
        self.assertIsNone(calibration)
        self.assertEqual(min(report, key=report.get), "none")

    def test_apply_calibration(self):
        log_q = log_normalize([math.log(0.9), math.log(0.1)])
        hot = apply_calibration({"type": "temperature", "temperature": 1e6}, log_q)
        self.assertAlmostEqual(math.exp(hot[0]), 0.5, places=4)
        identity = apply_calibration({"type": "platt", "a": 1.0, "b": 0.0}, log_q)
        self.assertAlmostEqual(math.exp(identity[0]), 0.9)
        shifted = apply_calibration(
            {"type": "vector", "scale": [1.0, 1.0], "bias": [0.0, math.log(9)]}, log_q
        )
        self.assertAlmostEqual(math.exp(shifted[0]), 0.5)
        for calibration, row in (
            ({"type": "platt", "a": 1.0, "b": 0.0}, [0.0, -1.0, -2.0]),
            ({"type": "vector", "scale": [1.0], "bias": [0.0]}, log_q),
        ):
            with self.assertRaises(ValueError):
                apply_calibration(calibration, row)


class TestMetrics(CustomTestCase):
    def test_hand_computed_values(self):
        self.assertAlmostEqual(
            expected_calibration_error([0.9, 0.9, 0.1], [True, False, False], bins=10),
            (abs(1.8 - 1) + abs(0.1 - 0)) / 3,
        )
        self.assertEqual(auroc([0.9, 0.8, 0.1, 0.2], [True, True, False, False]), 1.0)
        self.assertEqual(auroc([0.5, 0.5], [True, False]), 0.5)
        perfect = [[0.0, -math.inf, -math.inf]]
        self.assertEqual(ranked_probability_score(perfect, [0]), 0.0)
        self.assertEqual(quadratic_weighted_kappa([0, 1, 2], [0, 1, 2], 3), 1.0)
        self.assertEqual(
            selective_accuracy([0.9, 0.8, 0.2], [True, False, False], 0.34), 1.0
        )
        self.assertAlmostEqual(
            decidable_share([0.9, 0.8, 0.7, 0.6], [True, True, False, True], 0.05), 0.5
        )


class TestConfig(CustomTestCase):
    def _config(self, **overrides):
        reads = {**msgspec.structs.asdict(LABEL_FREE), "choice_max_orders": 4}
        body = {"default_reads": reads, "max_choice_orders": 8}
        body.update(overrides)
        return json.dumps(body).encode()

    def test_config_is_explicit_and_bounded(self):
        self.assertEqual(decode_config(self._config()).default_reads, LABEL_FREE)
        builtin = load_config(None)
        self.assertEqual(builtin.default_reads, RAW_READS)
        self.assertEqual(builtin.max_choice_orders, BUILTIN_MAX_CHOICE_ORDERS)
        body = json.loads(self._config())
        del body["default_reads"]["noul_orders"]
        reads = msgspec.structs.asdict(LABEL_FREE)
        cases = {
            "missing field": json.dumps(body).encode(),
            "unknown field": self._config(fitted=None),
            "orders above 2": self._config(default_reads={**reads, "noul_orders": 3}),
            "unknown scheme": self._config(
                default_reads={**reads, "choice_orders": "random"}
            ),
        }
        for name, data in cases.items():
            with self.subTest(name), self.assertRaises(msgspec.ValidationError):
                decode_config(data)
        # Default reads must fit the cap, so clients that cannot choose are never refused.
        for default_max in ("all", 9):
            with self.subTest(default_max), self.assertRaises(ValueError):
                decode_config(
                    self._config(
                        default_reads={**reads, "choice_max_orders": default_max}
                    )
                )

    def test_fingerprints_cover_what_a_calibration_depends_on(self):
        def fingerprint(kind, options=4, model="m", **fields):
            setup = msgspec.structs.replace(LABEL_FREE, **fields)
            return reads_fingerprint(kind, options, setup, model, "r", 1)

        base = fingerprint("choice")
        self.assertNotEqual(base, fingerprint("choice", choice_max_orders=2))
        self.assertNotEqual(base, fingerprint("choice", choice_orders="rotations"))
        self.assertNotEqual(base, fingerprint("choice", options=5))
        self.assertNotEqual(base, fingerprint("choice", model="other"))
        # The same reads however the setup spells them.
        self.assertEqual(base, fingerprint("choice", choice_max_orders="all"))
        self.assertEqual(base, fingerprint("choice", choice_max_orders=20))
        self.assertEqual(
            fingerprint("choice", choice_max_orders=1),
            fingerprint("choice", choice_max_orders=1, choice_orders="rotations"),
        )
        # Choice orders do not touch noul reads, nor noul settings choice reads.
        self.assertEqual(
            fingerprint("yes_no", options=2),
            fingerprint("yes_no", options=2, choice_max_orders=2),
        )
        self.assertNotEqual(
            fingerprint("yes_no", options=2),
            fingerprint("yes_no", options=2, noul_orders=1),
        )
        self.assertEqual(base, fingerprint("choice", noul_orders=1))


if __name__ == "__main__":
    unittest.main()
