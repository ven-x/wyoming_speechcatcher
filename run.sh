#!/usr/bin/with-contenv bashio

set -e

CACHE_DIR="/share/wyoming-speechcatcher"
mkdir -p "$CACHE_DIR"

PORT="$(bashio::config 'port')"
MODEL="$(bashio::config 'model')"
LANGUAGE="$(bashio::config 'language')"
BEAM_SIZE="$(bashio::config 'beam_size')"
CTC_WEIGHT="$(bashio::config 'ctc_weight')"
DECODER="$(bashio::config 'decoder')"
NUM_CPU="$(bashio::config 'num_cpu')"
PENALTY="$(bashio::config 'penalty')"

ARGS=(
    --uri "tcp://0.0.0.0:${PORT}"
    --cache-dir "$CACHE_DIR"
    --model "$MODEL"
    --language "$LANGUAGE"
    --beam-size "$BEAM_SIZE"
    --ctc-weight "$CTC_WEIGHT"
    --decoder "$DECODER"
    --num-threads "$NUM_CPU"
    --penalty "$PENALTY"
)

for lang in $(bashio::config 'preload_languages'); do
    ARGS+=(--preload-language "$lang")
done

if bashio::config.true 'stream_transcript'; then
    ARGS+=(--stream-transcript)
fi
if bashio::config.true 'external_vad'; then
    ARGS+=(--external-vad)
fi
if bashio::config.true 'debug'; then
    ARGS+=(--debug)
fi
if bashio::config.true 'disable_bbd'; then
    ARGS+=(--disable-bbd)
fi

bashio::log.info "Starting wyoming-speechcatcher on tcp://0.0.0.0:${PORT} (model=$MODEL, language=$LANGUAGE, beam=$BEAM_SIZE)"

exec wyoming-speechcatcher "${ARGS[@]}"
