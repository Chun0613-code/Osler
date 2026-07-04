# Osler Personalized Belief Demo

This is a research demo for Osler's validated patient-specific belief layer.

It calls the real predict-update belief builders:

- `aki_renal_belief.py`
- `cardiovascular_belief.py`
- `electrolyte_belief.py`
- `respiratory_belief.py`
- `endocrine_belief.py`

The green personalized value in the UI is not a fixed illustrative gain. It is
computed from the submitted patient trajectory through those belief builders.

## What This Demo Serves

- Population forecast: persistence baseline from the latest observed value.
- Personalized forecast: belief-derived adjustment from the patient's own
  trajectory.
- Interval: conservative validation-scale demo band.

The audit ridge coefficients are not serialized as production artifacts yet, so
this app does not claim to serve the full nested router. It demonstrates the
validated personalization layer and keeps that boundary explicit.

## Safety Boundary

This demo is research-only:

- no clinical claim
- no causal claim
- no counterfactual claim
- no treatment recommendation
- no patient identifiers persisted

## Run Locally

From the repository root:

```bash
pip install -r personalization_demo/requirements.txt
uvicorn personalization_demo.app:app --host 0.0.0.0 --port 7860
```

Then open `http://localhost:7860`.

## Hugging Face Spaces

Use the Docker SDK. Deploy from the repository root so the app can import the
belief modules next to this package.

The included `Dockerfile` exposes port `7860`, which is the default expected by
Hugging Face Spaces Docker apps.

