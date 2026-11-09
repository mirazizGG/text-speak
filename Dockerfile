ARG BASE_IMAGE=kamera-nazorati:accuracy-base-20261024
FROM ${BASE_IMAGE}
USER root
WORKDIR /srv/text-speak
COPY requirements.txt requirements.lock ./
RUN --mount=type=bind,source=offline/wheels,target=/wheels,readonly \
    python -m pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.lock \
    && python -m pip check \
    && python -c "import imageio_ffmpeg; print('FFmpeg:', imageio_ffmpeg.get_ffmpeg_version())"
RUN groupadd --gid 10001 tts \
    && useradd --uid 10001 --gid 10001 --home-dir /data --no-create-home --shell /usr/sbin/nologin tts \
    && mkdir -p /data/voices /data/jobs /data/tmp /data/huggingface /srv/text-speak \
    && chown -R 10001:10001 /data /srv/text-speak
COPY api.py engine.py jobs.py job_service.py inference_pool.py trt_backend.py flashinfer_backend.py ./
COPY static ./static
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/srv/text-speak HOME=/data TMPDIR=/data/tmp \
    HF_HOME=/data/huggingface TTS_VOICES_DIR=/data/voices TTS_JOBS_DIR=/data/jobs \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
    TORCHINDUCTOR_CACHE_DIR=/data/torchinductor NUMBA_CACHE_DIR=/data/numba XDG_CACHE_HOME=/data/cache
USER 10001:10001
EXPOSE 8070
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8070/readyz', timeout=4)"
ENTRYPOINT []
CMD ["python", "-m", "uvicorn", "api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8070", "--workers", "1", "--timeout-keep-alive", "5", "--timeout-graceful-shutdown", "30", "--no-access-log"]
