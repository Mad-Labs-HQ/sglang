"""Unit tests for System One calibration: read plans, combining reads, priors, fitting, config."""

import json
import math
import random
import unittest

import msgspec

from sglang.srt.entrypoints.systemone.calibration import (
    BatchPriorOff,
    BatchPriorOn,
    BatchPriors,
    FittedProfiles,
    LabelFreeConfig,
    PlattParams,
    Profile,
    TemperatureParams,
    apply_params,
    check_fitted_matches,
    combine_reads,
    decode_config,
    encode_config,
    find_profile,
    identity_read,
    label_variants,
    log_normalize,
    plan_reads,
    question_signature,
    read_log_probabilities,
    reads_fingerprint,
    rotation_offsets,
)
from sglang.srt.entrypoints.systemone.calibration_fit import (
    auroc,
    cross_validated_nll,
    decidable_share,
    expected_calibration_error,
    fit_platt,
    fit_temperature,
    quadratic_weighted_kappa,
    ranked_probability_score,
    selective_accuracy,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _label_free(**overrides):
    fields = dict(
        choice_rotations=4,
        choice_name_variants=True,
        noul_orders=2,
        noul_case_variants=True,
        batch_prior=BatchPriorOff(),
    )
    fields.update(overrides)
    return LabelFreeConfig(**fields)


def _config_json(**overrides):
    body = {
        "default_mode": "label_free",
        "label_free": {
            "choice_rotations": 4,
            "choice_name_variants": False,
            "noul_orders": 2,
            "noul_case_variants": True,
            "batch_prior": {"type": "off"},
        },
        "fitted": None,
    }
    body.update(overrides)
    return json.dumps(body).encode()


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

    def test_choice_rotations_keep_labels_by_position(self):
        reads = plan_reads("choice", ["A", "B", "C"], _label_free(choice_rotations=2))
        self.assertEqual([r.order for r in reads], [(0, 1, 2), (1, 2, 0)])
        self.assertTrue(all(r.labels == ("A", "B", "C") for r in reads))

    def test_noul_orders_swap_the_labels_shown(self):
        reads = plan_reads("yes_no", ["yes", "no"], _label_free())
        self.assertEqual(reads[0], identity_read(["yes", "no"]))
        self.assertEqual((reads[1].order, reads[1].labels), ((1, 0), ("no", "yes")))
        one = plan_reads("yes_no", ["yes", "no"], _label_free(noul_orders=1))
        self.assertEqual(one, [identity_read(["yes", "no"])])

    def test_score_levels_are_never_permuted(self):
        reads = plan_reads("score", ["0", "1", "2"], _label_free(choice_rotations=8))
        self.assertEqual(reads, [identity_read(["0", "1", "2"])])

    def test_label_variants(self):
        config = _label_free()
        self.assertEqual(label_variants("yes_no", "yes", "yes", config), ["Yes", "YES"])
        self.assertEqual(
            label_variants("choice", "A", "billing", config), ["billing", "Billing"]
        )
        # A name equal to its label adds nothing.
        self.assertEqual(label_variants("choice", "A", "A", config), [])
        off = _label_free(noul_case_variants=False, choice_name_variants=False)
        self.assertEqual(label_variants("yes_no", "yes", "yes", off), [])
        self.assertEqual(label_variants("choice", "A", "billing", off), [])
        self.assertEqual(label_variants("score", "0", "0", config), [])


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


class TestBatchPriors(CustomTestCase):
    def test_prior_applies_after_min_count_and_evicts_oldest(self):
        priors = BatchPriors(BatchPriorOn(strength=1.0, min_count=3, max_keys=2))
        skewed = log_normalize([math.log(0.9), math.log(0.1)])
        self.assertEqual(priors.apply("q", skewed), list(skewed))
        self.assertEqual(priors.apply("q", skewed), list(skewed))
        # The third answer equals the prior, so dividing it out gives uniform.
        third = priors.apply("q", skewed)
        self.assertAlmostEqual(math.exp(third[0]), 0.5)
        priors.apply("a", skewed)
        priors.apply("b", skewed)
        # "q" was least recently used and is gone, so it starts over.
        self.assertEqual(priors.apply("q", skewed), list(skewed))

    def test_strength_scales_the_prior(self):
        priors = BatchPriors(BatchPriorOn(strength=0.5, min_count=1, max_keys=8))
        log_q = log_normalize([math.log(0.8), math.log(0.2)])
        out = priors.apply("q", log_q)
        expected = [0.8 / math.sqrt(0.8), 0.2 / math.sqrt(0.2)]
        self.assertAlmostEqual(math.exp(out[0]), expected[0] / sum(expected))


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
        self.assertAlmostEqual(fitted.temperature, 2.5, delta=0.3)
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
        self.assertAlmostEqual(params.a, 0.5, delta=0.1)
        self.assertAlmostEqual(params.b, 0.7, delta=0.15)

    def test_apply_params(self):
        log_q = log_normalize([math.log(0.9), math.log(0.1)])
        hot = apply_params(TemperatureParams(temperature=1e6), log_q)
        self.assertAlmostEqual(math.exp(hot[0]), 0.5, places=4)
        identity = apply_params(PlattParams(a=1.0, b=0.0), log_q)
        self.assertAlmostEqual(math.exp(identity[0]), 0.9)
        with self.assertRaises(ValueError):
            apply_params(PlattParams(a=1.0, b=0.0), [0.0, -1.0, -2.0])


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
    def test_config_round_trips_and_forbids_unknown_fields(self):
        config = decode_config(_config_json())
        self.assertEqual(decode_config(encode_config(config)), config)
        body = json.loads(_config_json())
        body["label_free"]["rotations"] = 2
        with self.assertRaises(msgspec.ValidationError):
            decode_config(json.dumps(body).encode())

    def test_every_field_is_required(self):
        body = json.loads(_config_json())
        del body["fitted"]
        with self.assertRaises(msgspec.ValidationError):
            decode_config(json.dumps(body).encode())

    def test_invalid_values_are_refused(self):
        for field, value in (("choice_rotations", 0), ("noul_orders", 3)):
            with self.subTest(field):
                body = json.loads(_config_json())
                body["label_free"][field] = value
                with self.assertRaises(msgspec.ValidationError):
                    decode_config(json.dumps(body).encode())
        with self.assertRaises(ValueError):
            decode_config(_config_json(default_mode="fitted"))

    def _fitted(self, **overrides):
        fields = dict(
            model="m",
            model_revision="r",
            prompt_format_version=1,
            base="label_free",
            reads_fingerprint=reads_fingerprint("label_free", _label_free()),
            profiles=[],
        )
        fields.update(overrides)
        return FittedProfiles(**fields)

    def test_fitted_profiles_must_match_the_server(self):
        check_fitted_matches(self._fitted(), _label_free(), "m", "r", 1)
        # The batch prior is not part of the reads.
        check_fitted_matches(
            self._fitted(),
            _label_free(batch_prior=BatchPriorOn(strength=1, min_count=1, max_keys=1)),
            "m",
            "r",
            1,
        )
        for field, args in (
            ("model", ("other", "r", 1)),
            ("model_revision", ("m", "other", 1)),
            ("prompt_format_version", ("m", "r", 2)),
        ):
            with self.subTest(field), self.assertRaisesRegex(ValueError, field):
                check_fitted_matches(self._fitted(), _label_free(), *args)
        with self.assertRaisesRegex(ValueError, "reads_fingerprint"):
            check_fitted_matches(
                self._fitted(), _label_free(choice_rotations=2), "m", "r", 1
            )

    def test_profile_lookup_prefers_the_most_specific_key(self):
        signature = question_signature("choice", "Q", ["a", "b"], [None, None])
        profiles = [
            Profile(
                key=key,
                params=TemperatureParams(temperature=t),
                rows=100,
                oof_nll_base=1.0,
                oof_nll_fitted=0.9,
            )
            for key, t in (
                ("type:choice", 1.0),
                ("bucket:choice:2", 2.0),
                (f"signature:{signature}", 3.0),
            )
        ]
        fitted = self._fitted(profiles=profiles)
        self.assertEqual(
            find_profile(fitted, "choice", 2, signature).params.temperature, 3.0
        )
        self.assertEqual(
            find_profile(fitted, "choice", 2, "other").params.temperature, 2.0
        )
        self.assertEqual(
            find_profile(fitted, "choice", 3, "other").params.temperature, 1.0
        )
        self.assertIsNone(find_profile(fitted, "score", 3, "other"))


if __name__ == "__main__":
    unittest.main()
