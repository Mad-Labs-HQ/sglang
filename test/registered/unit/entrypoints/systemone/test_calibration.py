"""Unit tests for System One reads and calibration: read plans, combining reads, fitting, config."""

import json
import math
import random
import unittest

import msgspec

from sglang.srt.entrypoints.systemone.calibration import (
    BUILTIN_MAX_CHOICE_ROTATIONS,
    RAW_READS,
    ReadSetup,
    apply_calibration,
    combine_reads,
    decode_config,
    identity_read,
    label_variants,
    load_config,
    log_normalize,
    plan_reads,
    read_log_probabilities,
    reads_fingerprint,
    rotation_offsets,
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
    choice_rotations=4,
    choice_name_variants=True,
    noul_orders=2,
    noul_case_variants=True,
)


class TestReadPlans(CustomTestCase):
    def test_rotations_are_evenly_spaced_and_nested(self):
        self.assertEqual(rotation_offsets(4, 4), [0, 1, 2, 3])
        self.assertEqual(rotation_offsets(3, 8), [0, 1, 2])
        for n in (5, 6, 26, 77, 255):
            for k in (1, 2, 4):
                with self.subTest(n=n, k=k):
                    self.assertLessEqual(
                        set(rotation_offsets(n, k)), set(rotation_offsets(n, 2 * k))
                    )

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

    def test_choice_rotations_keep_labels_by_position(self):
        setup = msgspec.structs.replace(LABEL_FREE, choice_rotations=2)
        reads = plan_reads("choice", ["A", "B", "C"], setup)
        self.assertEqual([r.order for r in reads], [(0, 1, 2), (1, 2, 0)])
        self.assertTrue(all(r.labels == ("A", "B", "C") for r in reads))

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
        body = {
            "default_reads": msgspec.structs.asdict(LABEL_FREE),
            "max_choice_rotations": 8,
        }
        body.update(overrides)
        return json.dumps(body).encode()

    def test_config_is_explicit_and_bounded(self):
        self.assertEqual(decode_config(self._config()).default_reads, LABEL_FREE)
        builtin = load_config(None)
        self.assertEqual(builtin.default_reads, RAW_READS)
        self.assertEqual(builtin.max_choice_rotations, BUILTIN_MAX_CHOICE_ROTATIONS)
        body = json.loads(self._config())
        del body["default_reads"]["noul_orders"]
        cases = {
            "missing field": json.dumps(body).encode(),
            "unknown field": self._config(fitted=None),
            "orders above 2": self._config(
                default_reads={**msgspec.structs.asdict(LABEL_FREE), "noul_orders": 3}
            ),
        }
        for name, data in cases.items():
            with self.subTest(name), self.assertRaises(msgspec.ValidationError):
                decode_config(data)
        with self.assertRaises(ValueError):
            decode_config(self._config(max_choice_rotations=2))

    def test_fingerprints_cover_what_a_calibration_depends_on(self):
        def fingerprint(kind, setup=LABEL_FREE, model="m", revision="r", version=1):
            return reads_fingerprint(kind, setup, model, revision, version)

        more_rotations = msgspec.structs.replace(LABEL_FREE, choice_rotations=2)
        one_order = msgspec.structs.replace(LABEL_FREE, noul_orders=1)
        self.assertNotEqual(
            fingerprint("choice"), fingerprint("choice", more_rotations)
        )
        self.assertEqual(fingerprint("yes_no"), fingerprint("yes_no", more_rotations))
        self.assertNotEqual(fingerprint("yes_no"), fingerprint("yes_no", one_order))
        self.assertEqual(fingerprint("score"), fingerprint("score", one_order))
        for other in (
            fingerprint("choice", model="other"),
            fingerprint("choice", revision="other"),
            fingerprint("choice", version=2),
        ):
            self.assertNotEqual(fingerprint("choice"), other)


if __name__ == "__main__":
    unittest.main()
