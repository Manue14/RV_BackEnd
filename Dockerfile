FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# Instala dependencias de SO para compilar numpy/pandas
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential gcc gfortran \
      libatlas-base-dev liblapack-dev libffi-dev \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 5000
CMD ["gunicorn","--bind","0.0.0.0:5000","app:app","--workers","4","--threads","2"]
