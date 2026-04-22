FROM python:3.12-slim

# 1. Create a non-root user (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH="/home/user/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# 2. Install system dependencies (Root required for apt-get)
USER root

# Force apt to be "polite" to the proxy and disable pipelining
RUN echo "Acquire::http::Pipeline-Depth 0;" > /etc/apt/apt.conf.d/99fixproxy && \
    echo "Acquire::http::No-Cache true;" >> /etc/apt/apt.conf.d/99fixproxy && \
    echo "Acquire::BrokenProxy true;" >> /etc/apt/apt.conf.d/99fixproxy

RUN apt-get update -o Acquire::http::No-Cache=True && \
    apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    libpq-dev \
    gcc \
    git \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# 3. Pre-create directories and set ownership to 'user'
RUN mkdir -p logs data/pdfs data/processed data/failed data/web_snapshots && \
    chown -R user:user $HOME/app

# Switch back to non-root user
USER user

# 4. Install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org && \
    pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# 5. Copy application files
COPY --chown=user . .

# 6. Expose the Hugging Face default port
EXPOSE 7860

# 7. Optimized command (Reduced workers for stability)
CMD ["gunicorn", "main:app", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:7860", "--timeout", "120"]