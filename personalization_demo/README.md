# Osler Factual Forecast Demo

This research-only demo serves 12 serialized patient-state forecast artifacts.
Those exact target/horizon cells have byte-verified model, preprocessing,
population-anchor, metadata, and conformal files. Other belief-derived values
are illustrative model-internal state summaries; they are not validated
artifacts and never receive calibrated intervals.

It calls the real predict-update belief builders:

- `aki_renal_belief.py`
- `cardiovascular_belief.py`
- `electrolyte_belief.py`
- `respiratory_belief.py`
- `endocrine_belief.py`

Each output card displays its tier. Illustrative values are computed from the
submitted trajectory, but that computation does not make them validated.

## What This Demo Serves

- `validated_artifact`: one of the 12 exact serialized target/horizon cells;
  point forecasts and calibrated intervals come from that artifact.
- `illustrative`: a transparent belief-derived research illustration with no
  calibrated interval and no promotion authority.
- `unsupported`: no authorized forecast or interval.

Reported interval-width reduction is held-out cohort aggregate evidence only.
Across the 120 validation runs it was approximately 1.1% to 34.5%, with an
approximately 11.8% median. It is not a real-time claim about an individual
case.

## Safety Boundary

This demo is research-only:

- no clinical claim
- no causal claim
- no counterfactual claim
- no treatment recommendation
- no diagnosis
- no automated prescribing or drug-ranking changes
- no direct identifiers persisted; submitted API data is not stored

The model health endpoint (`/api/health/models`) performs strict manifest
verification and loads all 12 artifacts. It returns a failing status if any
artifact is missing, modified, or unloadable.

## Run Locally

From the repository root:

```bash
pip install -r personalization_demo/requirements.txt
waitress-serve --host=0.0.0.0 --port=7860 demo.demo_app:app
```

Then open `http://localhost:7860`.

## Hugging Face Spaces

Use the Docker SDK. Deploy from the repository root so the app can import the
belief modules next to this package.

The included `Dockerfile` exposes port `7860`, which is the default expected by
Hugging Face Spaces Docker apps.

