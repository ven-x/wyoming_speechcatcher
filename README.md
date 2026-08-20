# Note

This app is a small personal project that I’d like to make available to you. Due to time constraints, I’m unable to actively maintain the project.
It replaces the Vosk app, which I’d been using for a long time and which, after a few tweaks, had worked very reliably for me.

# Wyoming Speechcatcher — Home Assistant app

A Wyoming-compatible speech-to-text server as a Home Assistant app (add-on) with
[Speechcatcher](https://github.com/speechcatcher-asr/speechcatcher)
(EspNet2 Streaming Transformer) as the backend — a modern Vosk replacement for
Home Assistant Assist pipelines.

**Advantages over Vosk:** significantly better recognition quality
(WER ~8.5% instead of >15%), integrated punctuation, streaming recognition
with partial transcripts, actively maintained models (de/en/es).

- **Protocol:** [Wyoming](https://github.com/rhasspy/wyoming)
- **Backend:** Speechcatcher / EspNet2 Streaming Transformer (PyTorch)
- **Platforms:** `aarch64` and `amd64` (e.g. Raspberry Pi 4, x86_64 servers), CPU-only
- **Licence:** MIT (see `LICENSE.md`)

---

## Installation in Home Assistant

1. Open **Settings → Add-ons (Apps) → Add-on Store**.
2. In the top right-hand corner, select **⋮ (three-dot menu) → Repositories**.
3. Enter the repository URL and confirm:
   `https://github.com/ven-x/wyoming_speechcatcher`
4. **‘Wyoming Speechcatcher’** will now appear in the store list →
   select **Install**. The first build takes a while (Debian packages +
   PyTorch + speechcatcher are built into the image).
5. **Launch** the app. On first launch, the server downloads the default model from
   HuggingFace; it is saved to `/share/wyoming-speechcatcher`
   (persistent, survives app restarts and updates).

> **Note:** The app runs on the host network (`host_network: true`) and binds
> directly to the configured port. The Wyoming protocol has **no
> authentication** — therefore, only use the port on a trusted LAN
> and do not expose it unsecured to the internet.

---

## Configuration

All parameters are set in the app configuration (HA-UI) — no
need to edit files. Options with multiple values appear as drop-down menus,
and each option has a brief description directly in the UI.

| Option | Type | Default | Description |
|---|---|---|---|
| `port` | int (1024–65535) | `10300` | Port on which the Wyoming server listens (TCP) |
| `model` | Selection | `de_streaming_transformer_m` | ASR model (see [Models](#models)) |
| `language` | Selection (`de`/`en`/`es`) | `de` | Default recognition language |
| `preload_languages` | List | `[de]` | Languages loaded into memory when the server starts (more = more RAM, but immediately available) |
| `beam_size` | int (2–20) | `5` | Beam search width. Higher = marginally more accurate, slower |
| `ctc_weight` | float (0.0–1.0) | `0.3` | Weight of the acoustic CTC model. Higher = more accurate listening, but more choppy; lower = smoother, but greater risk of repetition |
| `decoder` | Selection (`native`/`espnet`) | `espnet` | Decoder implementation. `native` = fast, custom implementation; `espnet` = established reference decoder |
| `stream_transcript` | bool | `false` | Stream partial transcripts (HA displays text whilst the speaker is still speaking) |
| `external_vad` | bool | `false` | Require external VAD (for wake-word setups) |
| `num_cpu` | int (1–32) | `1` | PyTorch CPU threads. Higher values result in faster performance on multi-core CPUs |
| `penalty` | float (-1.0–1.0) | `0.0` | Length penalty (insertion penalty). Only for `decoder=espnet`; positive = shorter texts, negative = longer |
| `disable_bbd` | bool | `false` | Disable Block Boundary Detection (only for `decoder=native`). BBD prevents word repetition; disable only if problems arise |
| `debug` | bool | `false` | Enable detailed DEBUG logs |

---

## Models

All models are EspNet2 streaming transformers from
[speechcatcher-asr](https://huggingface.co/speechcatcher) (Attribution:
Benjamin Milde). The app registers all seven models with Home Assistant;
the one selected via `model` / `language` is loaded.

| Short name | Language | Size | Note |
|---|---|---|---|
| `de_streaming_transformer_m` | German | M (~500 MB RAM) | **Default** — good compromise for RPi 4 / x86 CPU |
| `de_streaming_transformer_l` | German | L | better quality, more RAM/CPU |
| `de_streaming_transformer_xl` | German | XL (26k h training) | best German quality; **not for RPi 3** |
| `en_streaming_transformer_m` | English | M | Default for `--language en` |
| `en_streaming_transformer_l` | English | L | |
| `es_streaming_transformer_m` | Spanish | M | Default for `--language es` |
| `es_streaming_transformer_l` | Spanish | L | |

---

## Using it in Home Assistant

1. After launching the app: **Settings → Devices & Services → Add integration
   → “Wyoming Protocol”**.
2. Enter the host `localhost` and port `10300` (or the configured port)
   and confirm.
3. Next: select **Settings → Voice Assistants → Assist Pipeline**
   (or create a new one) and under **‘Speech Recognition’ (Speech-to-text)**, select
   the entry **‘speechcatcher’**.

**Notes:**

- **Audio devices** (microphones, satellites) are configured exclusively in Home
  Assistant — the app is passive and device-agnostic.
  Suitable sources: Wyoming Satellite, ESPHome Voice, HA Companion app,
  browser. One app can control any number of devices simultaneously.
- **Partial transcripts:** Enable the `stream_transcript` option so that HA
  displays the recognised text whilst you are still speaking.
- **Wake word setups:** If an external VAD/wake word service is connected upstream,
  enable the `external_vad` option.

---

## Troubleshooting

| Symptom | Cause / Solution |
|---|---|
| App does not start | Check the log in the HA UI. Most common cause: invalid option (e.g. `beam_size` outside the range 2–20 or `ctc_weight` outside the range 0–1). |
| Initial recognition takes several seconds | The model is downloaded on first launch (several hundred MB). Afterwards, it is stored permanently in `/share/wyoming-speechcatcher`. |
| Model download fails | Network/HuggingFace unreachable. Check whether the HA host has internet access. |
| High RAM usage / OOM on RPi | The XL model is too large for 1 GB devices. Use the default `*_m`; reduce `beam_size` (e.g. 2–3). |
| HA cannot find the service | Check the port (`port` option); the host must be `localhost` (same host) or the HA host IP. |
| Empty transcript | Enable `debug` and check the log. If necessary, increase `beam_size` or change the model. |

---

## Licence & Attribution

This project (the Wyoming app/server code) is licensed under the
**MIT Licence** — see [`LICENSE.md`](LICENSE.md) in the repository root.

The ASR models used are sourced from
[**speechcatcher-asr**](https://github.com/speechcatcher-asr/speechcatcher)
([HuggingFace](https://huggingface.co/speechcatcher), copyright/attribution:
**Benjamin Milde**). Speechcatcher itself is an open-source project
(EspNet2-basiert); die App bindet es als Dependency ein. Für die genauen
Lizenzbedingungen von speechcatcher und der einzelnen Modelle gelten die
jeweiligen Angaben im [speechcatcher-Repository](https://github.com/speechcatcher-asr/speechcatcher)
bzw. auf den HuggingFace-Modellseiten.
