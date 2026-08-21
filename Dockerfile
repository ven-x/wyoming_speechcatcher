FROM ghcr.io/home-assistant/base-debian:latest

LABEL io.hass.version="0.1.0"
LABEL io.hass.type="app"
LABEL io.hass.arch="aarch64|amd64"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-pip \
        build-essential \
        portaudio19-dev \
        git \
        git-lfs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchaudio

WORKDIR /app
COPY pyproject.toml requirements.txt VERSION LICENSE.md README.md ./
COPY wyoming_speechcatcher/ wyoming_speechcatcher/
RUN pip3 install --break-system-packages --no-cache-dir .

COPY run.sh /run.sh
RUN chmod a+x /run.sh

ENV PYTHONUNBUFFERED=1

CMD ["/run.sh"]
