"""Compile an empirical simulator action prior from a MIMIC transition table."""

import argparse
import json

from osler_jepa.action_prior import EmpiricalActionPrior


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cohort")
    parser.add_argument("--output", default="dka_action_prior.json")
    args = parser.parse_args()
    prior = EmpiricalActionPrior.from_parquet(args.cohort)
    prior.save(args.output)
    print(json.dumps(prior.payload, indent=2))
    print(f"saved action prior: {args.output}")


if __name__ == "__main__":
    main()
