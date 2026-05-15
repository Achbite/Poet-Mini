FROM python:3.11-slim

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONIOENCODING=UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /workspace/train

COPY requirements.txt /tmp/requirements.txt
ARG TORCH_VERSION=2.7.1+cu128
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -m pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} torch==${TORCH_VERSION}

COPY . /workspace/train

RUN chmod +x /workspace/train/docker/data/entrypoint.sh \

    /workspace/train/build_data.sh \
    /workspace/train/build.sh \
    /workspace/train/run.sh


ENTRYPOINT ["/workspace/train/docker/data/entrypoint.sh"]
CMD ["--config", "configs/data.yaml"]
