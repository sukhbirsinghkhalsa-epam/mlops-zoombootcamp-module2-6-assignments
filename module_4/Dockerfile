FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

RUN pip install pandas pyarrow scikit-learn==1.5.0

COPY predict_docker.py .

ENTRYPOINT ["python", "predict_docker.py"]