# Oslian demo deployment

The Flask service hosts both the browser demo and the API:

- Browser demo: `https://demo.oslian.com/`
- Health check: `https://demo.oslian.com/healthz`
- API: `https://demo.oslian.com/api/*`
- Mobile app: set `EXPO_PUBLIC_API_BASE=https://demo.oslian.com`

## 1. Deploy the backend

The repository includes `render.yaml` and a production `Dockerfile`.

1. In Render, create a Blueprint from this GitHub repository. The Blueprint is
   pinned to the `feature/mobile-demo` branch.
2. Add the environment variables from `demo/.env.example` in the Render
   dashboard. Keep all secret values out of GitHub.
3. Set `PHOTON_OAUTH_REDIRECT` to the final public origin, for example
   `https://demo.oslian.com`.
4. Set `CORS_ORIGINS` only when the Expo web build is hosted on another origin.
   Use a comma-separated allowlist such as `https://app.example.com`.
5. Confirm that `/healthz` returns HTTP 200 before configuring DNS.

The Gunicorn command intentionally uses one worker because this demo stores
analysis, OAuth, and prescription state in memory. A production system should
move that state to Redis or a database before increasing worker count.
The Blueprint uses Render's free web-service plan for the demo, so the first
request after an idle period can take longer while the service wakes up.

## 2. Connect the Cloudflare domain

1. Add the domain to Cloudflare and use Cloudflare's assigned nameservers.
2. In Render, add the custom domain `demo.oslian.com`.
3. In Cloudflare DNS, create the CNAME Render provides with proxy status set to
   **DNS only**. Remove conflicting `AAAA` records for that hostname.
4. Wait until Render verifies the domain and issues its certificate. Then enable
   the Cloudflare proxy and switch SSL/TLS mode to `Full (strict)`.
5. Enable `Always Use HTTPS`.
6. In Photon/Neutron, allowlist the exact public OAuth origin used by
   `PHOTON_OAUTH_REDIRECT`.

Do not cache `/api/*`, `/healthz`, or OAuth callback responses in Cloudflare.

## 3. Point the mobile app at production

```bash
cd mobile
EXPO_PUBLIC_API_BASE=https://demo.oslian.com npx expo start
```

For a distributable app build, set the same variable in the EAS build profile.
Only the public API URL belongs in `EXPO_PUBLIC_*`; Photon and LLM credentials
must remain server-side or in the device's secure settings.

## 4. Smoke test

```bash
curl -fsS https://demo.oslian.com/healthz
curl -fsS https://demo.oslian.com/api/cases
```

Then analyze a preset case in the browser and in the mobile app, open Reasoning,
Chat, and Rx, and send only a Photon sandbox prescription.
