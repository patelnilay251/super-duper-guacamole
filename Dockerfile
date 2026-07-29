# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Pillow needs zlib/libjpeg at runtime; the slim image already carries them, so
# nothing to apt-install. Keeping it that way is what holds the image near 200MB.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ditherwall/ ./ditherwall/
COPY web/ ./web/

# Bake the photo corpus into the image. Fetching at boot would make every cold
# start wait on ~30 HTTP round trips, and a free instance cold-starts often.
ARG PHOTO_COUNT=32
RUN python3 -c "import sys; sys.path.insert(0, 'ditherwall'); \
    from fetch import fetch; print('baked', len(fetch($PHOTO_COUNT)), 'photos')"

# Conservative defaults for a small shared instance. Override per environment.
ENV HOST=0.0.0.0 \
    PORT=8000 \
    DW_PHOTOS=32 \
    DW_WORKERS=2 \
    DW_MAX_EDGE=480 \
    DW_CACHE_DEPTH=3

RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python3 -c "import urllib.request,os; \
        urllib.request.urlopen(f\"http://127.0.0.1:{os.environ['PORT']}/healthz\", timeout=4)"

CMD ["python3", "web/server.py"]
