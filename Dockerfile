FROM python:3.10-slim

# System deps (FFmpeg is required for torchcodec/torchaudio saving)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python deps
COPY stemdecoder_optionA_decodeaudio/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app code
COPY stemdecoder_optionA_decodeaudio /app

EXPOSE 10000

CMD ["bash", "-lc", "streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.enableXsrfProtection false --server.enableCORS false"]

