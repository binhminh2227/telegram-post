# tg-worker-cloudrun

FastAPI service (Cloud Run) – giống Cloudflare Worker:
- /setup: thêm kênh + nhiều target bot/channel
- /status: xem cấu hình/log
- /delete: xoá kênh
- /post-new: VPS gửi bài (JSON hoặc multipart) → đăng Telegram (1 post duy nhất, album nếu ≥2 media)

## Deploy (không Dockerfile)
```bash
gcloud auth login
gcloud config set project <PROJECT_ID>
gcloud services enable run.googleapis.com cloudbuild.googleapis.com

gcloud run deploy tg-worker \
  --source . \
  --region asia-southeast1 \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_ENTRYPOINT="uvicorn app:app --host 0.0.0.0 --port \$PORT" \
  --set-env-vars VPS_BASE=http://<VPS_IP>:8080,VPS_BEARER=<BEAR>,CALLBACK_BEARER=<CALLBACK_BEARER> \
  --set-env-vars ENABLE_LLM=false \
  --memory 512Mi --cpu 1 --concurrency 40 --min-instances 0 --max-instances 10
