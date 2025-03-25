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

### Documentation

Docs available at `/docs` endpoint 

![test](https://github.com/user-attachments/assets/4f46df67-3c47-478a-a2c5-6a1f577b660d)

## Local Setup - Demo

```
cd demo
python3 -m pip install -r requirements.txt
python3 -m streamlit run demo.py
```

UI available at http://localhost:8501