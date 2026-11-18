# syntax=docker/dockerfile:1.7
# The `h3` binary (CUDA + CPU backends) and the three CUDA libraries it loads
# at run time; nothing else. The driver (libcuda) comes from the host through
# the NVIDIA container runtime.
#
#   service:  docker run --gpus all -p 7860:7860 IMAGE serve --addr 0.0.0.0:7860 --model-dir … …
#   CLI:      docker run --gpus all IMAGE generate …   (or check --backend cuda)
#   RunPod serverless:  IMAGE worker   (see crates/h3-cli/src/worker.rs)

FROM rust:1.97-slim-bookworm AS build
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked -p h3-cli --no-default-features --features cuda \
    && install -D target/release/h3 /out/h3

# cuBLAS / cuBLASLt (GEMMs) and NVRTC (kernels are compiled at run time) at
# the CUDA version cudarc is built for (12.8: sm_120 needs it). The driver
# must support CUDA 12.8 (R570+).
FROM nvidia/cuda:12.8.1-base-ubuntu22.04 AS cudalibs
RUN apt-get update \
    && apt-get install -y --no-install-recommends libcublas-12-8 cuda-nvrtc-12-8 \
    && mkdir /out \
    && cp -a /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12* \
             /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublasLt.so.12* \
             /usr/local/cuda-12.8/targets/x86_64-linux/lib/libnvrtc.so.12* \
             /usr/local/cuda-12.8/targets/x86_64-linux/lib/libnvrtc-builtins.so.12* /out/

FROM debian:bookworm-slim
COPY --from=cudalibs /out/ /opt/h3/lib/
COPY --from=build /out/h3 /usr/local/bin/h3
ENV LD_LIBRARY_PATH=/opt/h3/lib \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
EXPOSE 7860
ENTRYPOINT ["/usr/local/bin/h3"]
CMD ["--help"]
