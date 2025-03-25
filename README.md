# Avatar Generator

## Deploy on Vast.ai

```
python3 deploy_vastai.py <api_key>
```

To include PostHog analytics, run with `--posthog-key`

```
python3 deploy_vastai.py <api_key> --posthog-key <posthog_key>
```

## Local Setup - Server

```
cd webui
docker compose build
docker compose up -d
```

API available at http://localhost:8000

## Local Setup - Demo

```
cd demo
python3 -m pip install -r requirements.txt
python3 -m streamlit run demo.py
```

UI available at http://localhost:8501