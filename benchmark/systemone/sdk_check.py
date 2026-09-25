"""Ask a server through the official TypeSafe Python SDK, as a System One client would.

python benchmark/systemone/sdk_check.py --url http://127.0.0.1:30000
"""

import argparse

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    # A server started without --api-key accepts any key.
    client = TypeSafeClient(base_url=args.url, api_key="unused")
    result = client.system_one(
        "I've been trying to connect my Stripe account for 3 days and the "
        "integration keeps failing.",
        {
            "team": Choice(
                instructions="Which team should handle this ticket?",
                criteria={
                    "billing": None,
                    "technical": "Bugs or integration problems",
                    "sales": None,
                },
            ),
            "urgent": Noul(instructions="The customer needs an answer today."),
            "frustration": Score(
                instructions="How frustrated is the customer?",
                criteria=["Calm", "Frustrated but civil", "Very angry"],
            ),
        },
    )
    raw = result.raw_http_response.json()
    for name, answer in result.answers.items():
        print(name, answer, "x_calibration:", raw["answers"][name]["x_calibration"])
    assert result.answers["team"].choice in ("billing", "technical", "sales")
    assert 0 <= result.answers["urgent"].noul <= 1
    assert 0 <= result.answers["frustration"].score <= 2


if __name__ == "__main__":
    main()
