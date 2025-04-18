FROM python:3.11-slim

RUN apt-get update && apt-get install -y         libglib2.0-0 libsm6 libxrender1 libxext6         libpulse-dev libsndfile1 ffmpeg         && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "-m", "gui.main_window"]