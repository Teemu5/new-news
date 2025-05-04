#FROM python:3.8-slim
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip

RUN pip install --default-timeout=600 --no-cache-dir -r requirements.txt
#RUN pip install nltk

EXPOSE 5000
ENV NAME World

CMD ["python", "backend.py"]
#CMD ["python", "fastapi.py"]
