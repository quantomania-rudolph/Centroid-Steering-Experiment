# CUDA PyTorch base for Vast.ai / RunPod H100 instances.
# Build locally: docker build -t jlens-fit .
# Or use Vast's PyTorch template and run run_fit.sh instead.

FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

WORKDIR /workspace/jlens-rental

RUN apt-get update && apt-get install -y --no-install-recommends git openssh-client \
    && rm -rf /var/lib/apt/lists/*

COPY fit_config.yaml fit_on_gpu.py run_fit.sh ./
RUN chmod +x run_fit.sh

# Prompts + secrets uploaded at runtime (not baked into image).
ENV RENTAL_REMOTE_DIR=/workspace/jlens-rental

ENTRYPOINT ["bash", "run_fit.sh"]
