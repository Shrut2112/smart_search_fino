# Smart Search Fino — Deployment Guide

This project consists of 5 services that must run simultaneously for a complete production deployment:
1. **Frontend**: React application bundled and served by Nginx.
2. **API Backend**: FastAPI running with Gunicorn (`api:app`) to answer chat requests.
3. **Admin Dashboard**: Streamlit app for monitoring (`admin_dashboard/Home.py`).
4. **File Watcher**: Persistent Python process monitoring the `data/pdfs/` folder for new documents.
5. **Web Watcher**: Python process polling `fino.bank.in` via Firecrawl.

---

## 1. Local Production Simulation (Docker Compose)

The easiest way to run the entire cluster locally is using `docker-compose`.

```bash
# Start all 5 services in detached mode
docker-compose up --build -d

# Check live logs for any service
docker-compose logs -f api-backend
docker-compose logs -f file-watcher
```

- **User UI**: `http://localhost:5173`
- **API Server**: `http://localhost:8000`
- **Admin Dashboard**: `http://localhost:8502`

You can test processing PDFs by manually placing a file in the bind-mounted `data/pdfs/` folder on your host machine.

---

## 2. Free-Tier Cloud Deployments (Railway / Render)

Both Railway and Render support multi-service repositories. The trick is to point multiple services to the same Git repository but override the **Start Command** for each service.

### Railway (Recommended)
Railway is ideal because is offers TCP networking, background task limits, and persistent volumes.
1. Connect your GitHub repository to Railway.
2. Add a `PostgreSQL` plugin for your database (if not using Supabase).
3. Create a **Shared Volume** called `data-volume` via the Railway dashboard.
4. Deploy the repository **4 times**, overriding the command for each:
   - **Service 1 (API)**: Start command `gunicorn api:app --workers 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT`
   - **Service 2 (Admin UI)**: Start command `streamlit run admin_dashboard/Home.py --server.port $PORT`
   - **Service 3 (File Watcher)**: Start command `python watchers/watcher.py`. Mount `data-volume` to `/app/data/`.
   - **Service 4 (Web Watcher)**: Start command `python watchers/web_watcher.py`
5. Map the `data-volume` to all services so they share the same persistence.

### Render
Render has "Web Services" and "Background Workers".
- Deploy the **API** and **Admin UI** as a "Web Service" (requires selecting Python environment or Docker).
- Deploy the **Watchers** as "Background Workers" so Render does not force them to sleep if there are no incoming HTTP HTTP requests.

---

## 3. Enterprise AWS Deployment (ECS Fargate)

When transitioning to company credentials, the standard approach is a serverless cluster on AWS ECS Fargate.

### Architecture Overview:
- **Registry**: Push your single Docker image to AWS ECR (`aws_account_id.dkr.ecr.region.amazonaws.com/fino-search`).
- **Cluster**: Create an AWS ECS Cluster using Fargate launch type.
- **Task Definitions**: Create 4 distinct task definitions, all using the same ECR image but overriding the `Command`.
- **EFS (Elastic File System)**: Create an EFS drive and mount it to `/app/data/` for all containers. This ensures the watchdog service and the API service can read/write to the exact same PDF queues.
- **ALB (Application Load Balancer)**: Put an ALB in front of the API and Admin Dashboard to handle HTTPS termination and route internet traffic.

### Basic Steps:
1. Build and push image:
   ```bash
   aws ecr get-login-password | docker login --username AWS --password-stdin YOUR_URL
   docker build -t fino-app .
   docker tag fino-app:latest YOUR_URL/fino-app:latest
   docker push YOUR_URL/fino-app:latest
   ```
2. Setup **AWS EFS** (Elastic File System).
3. Define ECS Tasks, setting EFS mounts correctly.
4. Deploy Tasks into an ECS Service to ensure they always restart on failure.
