"""/v1/systemone on a chat template that always reasons, raw and with label-free calibration."""

import json
import math
import os
import tempfile
import unittest

import requests
from transformers import AutoTokenizer

from sglang.srt.entrypoints.systemone.calibration import (
    combine_reads,
    read_log_probabilities,
)
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=150, stage="base-b", runner_config="1-gpu-small")

MODEL = "IFM/K2-Horizon-0.9B"

LABEL_FREE = {
    "default_mode": "raw",
    "label_free": {
        "choice_rotations": 3,
        "choice_name_variants": False,
        "noul_orders": 2,
        "noul_case_variants": True,
        "batch_prior": {"type": "off"},
    },
    "fitted": None,
}

QUESTIONS = {
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {"billing": "Payments", "technical": "Bugs", "sales": None},
    },
    "urgent": {"type": "noul", "instructions": "The customer needs an answer today."},
    "mood": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["Calm", "Frustrated", "Very angry"],
    },
}
STATE = "My Stripe integration has been failing for 3 days and I'm losing sales."


class TestSystemOneK2Horizon(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.config_dir = tempfile.TemporaryDirectory()
        config_path = os.path.join(cls.config_dir.name, "label_free.json")
        with open(config_path, "w") as f:
            json.dump(LABEL_FREE, f)
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        # No --reasoning-parser: the template's reasoning markers are detected.
        cls.process = popen_launch_server(
            MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            # The checkpoint's dtype metadata says float32; K2 Horizon serves in BF16.
            other_args=[
                "--dtype",
                "bfloat16",
                "--decision-calibration-config",
                config_path,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)
        if hasattr(cls, "config_dir"):
            cls.config_dir.cleanup()

    def _systemone(self, **extensions):
        response = requests.post(
            self.base_url + "/v1/systemone",
            json={
                "state": STATE,
                "model": "jev-latest",
                "questions": QUESTIONS,
                **extensions,
            },
            timeout=120,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["answers"]

    def test_raw_answers_have_the_published_shapes(self):
        answers = self._systemone()
        self.assertEqual(list(answers), list(QUESTIONS))
        for question_id, answer in answers.items():
            with self.subTest(question_id):
                self.assertEqual(answer["x_calibration"], "raw")
                if answer["type"] == "noul":
                    self.assertTrue(0 <= answer["noul"] <= 1)
                else:
                    self.assertAlmostEqual(
                        sum(answer["probabilities"].values()), 1.0, places=5
                    )
                    self.assertTrue(0 <= answer["confidence"] <= 1)
        self.assertIn(answers["team"]["choice"], QUESTIONS["team"]["criteria"])

    def test_answers_are_read_after_an_empty_reasoning_block(self):
        response = requests.post(
            self.base_url + "/v1/decisions",
            json={
                "input": STATE,
                "questions": [
                    {"id": "q", "type": "yes_no", "question": "Is this urgent?"}
                ],
                "return_prompt_token_ids": True,
            },
            timeout=120,
        )
        self.assertEqual(response.status_code, 200, response.text)
        answer = response.json()["answers"]["q"]
        prompt = self.tokenizer.decode(answer["prompt_token_ids"])
        self.assertTrue(
            prompt.endswith("assistant\n<ifm|think>\n</ifm|think>\n"), prompt[-80:]
        )

    def test_label_free_answers_equal_their_reads_combined(self):
        answers = self._systemone(x_calibration="label_free", x_return_reads=True)
        self.assertEqual(len(answers["team"]["x_reads"]), 3)
        self.assertEqual(len(answers["urgent"]["x_reads"]), 2)
        self.assertEqual(len(answers["mood"]["x_reads"]), 1)
        # Case variants carry most of the yes and no mass on this model.
        self.assertGreater(answers["urgent"]["x_label_mass"], 0.5)
        for question_id, answer in answers.items():
            with self.subTest(question_id):
                self.assertEqual(answer["x_calibration"], "label_free")
                names = answer["x_reads"][0]["order"]
                expected = [
                    math.exp(v)
                    for v in combine_reads(
                        [
                            read_log_probabilities([read["logprobs"][n] for n in names])
                            for read in answer["x_reads"]
                        ]
                    )
                ]
                served = (
                    [answer["noul"]]
                    if answer["type"] == "noul"
                    else list(answer["probabilities"].values())
                )
                for got, want in zip(served, expected):
                    self.assertAlmostEqual(got, want, places=9)


if __name__ == "__main__":
    unittest.main(verbosity=3)
