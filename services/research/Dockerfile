FROM python:3.11-slim
WORKDIR /app
COPY requirements-runtime.txt /app/
RUN pip install --no-cache-dir -r requirements-runtime.txt
ARG MODEL_RUNTIME=onnx
RUN if [ "$MODEL_RUNTIME" = "torchscript" ]; then pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu; elif [ "$MODEL_RUNTIME" = "tbnet_tf" ]; then pip install --no-cache-dir tensorflow-cpu==2.15.1 opencv-python-headless==4.10.0.84; elif [ "$MODEL_RUNTIME" != "onnx" ]; then exit 1; fi
COPY config.py /app/
COPY src /app/src
COPY services /app/services
RUN useradd --uid 10001 --create-home inference
USER inference
ENV MEDISCAN_MODEL_BUNDLE=/models
EXPOSE 8000
CMD ["uvicorn", "services.research.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "4", "--no-access-log"]
