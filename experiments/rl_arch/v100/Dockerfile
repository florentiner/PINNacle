# Итоговый метод на a-dgx (V100): код этой ветки поверх образа абляции агента статьи, где уже есть
# torch 2.4.1 (cu124), numpy<2, scipy, scikit-learn, dill, huggingface_hub и т. д. Интернет при сборке не нужен.
#   docker --context a-dgx build -f experiments/rl_arch/v100/Dockerfile -t rlpinn-final-v100:<commit> .
ARG BASE=rlpinn-ablation-v100:6593142-dirty
FROM ${BASE}
RUN rm -rf /app
COPY . /app
WORKDIR /app
ENV PYTHONPATH=/app \
    DDEBACKEND=pytorch \
    MPLBACKEND=Agg \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    NVIDIA_VISIBLE_DEVICES=void
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}
CMD ["python", "-u", "experiments/rl_arch/v100/run_jobs.py", "--work", "/work"]
