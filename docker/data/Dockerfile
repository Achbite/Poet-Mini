FROM python:3.11-slim

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONIOENCODING=UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/train

RUN pip install --no-cache-dir pyyaml

COPY . /workspace/train

RUN chmod +x /workspace/train/docker/data/entrypoint.sh \
    /workspace/train/build_data.sh \
    /workspace/train/build.sh \
    /workspace/train/run.sh


ENTRYPOINT ["/workspace/train/docker/data/entrypoint.sh"]
CMD ["--config", "configs/data.yaml"]
