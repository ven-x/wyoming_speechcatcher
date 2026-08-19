# syntax=docker/dockerfile:1
#
# Home-Assistant-Add-on-Image (App) fuer wyoming-speechcatcher.
#
# Build-Kontext ist der Repo-Root — der HA-Supervisor baut die App mit
# dem Ordner, der die config.yaml enthaelt (= hier der Repo-Root).
# Package und Add-on liegen im selben Ordner, daher baut das Dockerfile
# das Package direkt per COPY + pip install . ein (KEIN Git-Bezug,
# kein Commit-Pin noetig).
#
# Debian-Basis (glibc) statt Alpine, weil PyTorch keine musl-Wheels
# veroeffentlicht — auf glibc-Basis installieren die manylinux-CPU-Wheels
# direkt ohne Source-Build (AUDIT-029).

FROM ghcr.io/home-assistant/base-debian:latest

# HA-Add-on-Metadaten (Konvention).
LABEL io.hass.version="0.1.0"
LABEL io.hass.type="app"
LABEL io.hass.arch="aarch64|amd64"

# System-Abhaengigkeiten (Debian → apt, KEIN apk):
#   python3, python3-dev, python3-pip → Python + Header (Python.h fuer
#     C-Extensions) + pip — im base-debian-Image NICHT enthalten
#   build-essential  → gcc/g++/make (C-Extensions: pyaudio, ctc-segmentation, pyworld)
#   portaudio19-dev  → pyaudio (Build-Abhaengigkeit von speechcatcher)
#   git, git-lfs     → espnet_model_zoo (Modell-Download von HuggingFace)
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

# CPU-only torch VOR dem Package installieren.
# Sonst zieht speechcatcher das (groessere) CUDA-torch als Default
# (speechcatcher deklariert torch/torchaudio als harte Dependency).
# Auf Debian (glibc) greifen die manylinux-CPU-Wheels direkt.
# Debian (bookworm/trixie) ist PEP-668-verwaltet → --break-system-packages
# ist noetig, damit pip ins System-Python installieren darf.
RUN pip3 install --break-system-packages --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchaudio

# Package-Quellen kopieren und installieren.
# requirements.txt (mit speechcatcher Git-Pin) wird von pip mitverarbeitet.
WORKDIR /app
COPY pyproject.toml requirements.txt VERSION LICENSE.md README.md ./
COPY wyoming_speechcatcher/ wyoming_speechcatcher/
RUN pip3 install --break-system-packages --no-cache-dir .

# Start-Skript des Add-ons.
COPY run.sh /run.sh
RUN chmod a+x /run.sh

# Ungepufferte Logs im Supervisor/Docker-Log-Treiber.
ENV PYTHONUNBUFFERED=1

# s6-overlay Service-Skript (with-contenv/bashio in run.sh).
CMD ["/run.sh"]
