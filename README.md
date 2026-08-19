# Hinweis

Diese App ist ein kleines privates Projekt, das ich dir zur Verfügung stellen möchte. Aus Zeitgründen kann ich das Projekt nicht aktiv begleiten.
Es ersetzt die von mir lange Zeit genutzte Vosk App, die nach einigen Tweaks sehr zuverlässig für mich funktioniert hatte.

# Wyoming Speechcatcher — Home Assistant App

Wyoming-kompatibler Speech-to-Text-Server als Home-Assistant-App (Add-on) mit
[Speechcatcher](https://github.com/speechcatcher-asr/speechcatcher)
(EspNet2 Streaming Transformer) als Backend — ein moderner Vosk-Ersatz für
Home-Assistant-Assist-Pipelines.

**Vorteile gegenüber Vosk:** deutlich bessere Erkennungsqualität
(WER ~8,5 % statt >15 %), integrierte Interpunktion, Streaming-Erkennung
mit partiellen Transkripten, aktiv gepflegte Modelle (de/en/es).

- **Protokoll:** [Wyoming](https://github.com/rhasspy/wyoming)
- **Backend:** Speechcatcher / EspNet2 Streaming Transformer (PyTorch)
- **Plattformen:** `aarch64` und `amd64` (z. B. Raspberry Pi 4, x86_64-Server), CPU-only
- **Lizenz:** MIT (siehe `LICENSE.md`)

---

## Installation in Home Assistant

1. **Einstellungen → Add-ons (Apps) → Add-on Store** öffnen.
2. Rechts oben **⋮ (Drei-Punkte-Menü) → Repositories** wählen.
3. Die Repository-URL eintragen und bestätigen:
   `https://github.com/ven-x/wyoming_speechcatcher`
4. In der Store-Liste erscheint jetzt **„Wyoming Speechcatcher“** →
   **Installieren** wählen. Der erste Build dauert eine Weile (Debian-Pakete +
   PyTorch + speechcatcher werden im Image gebaut).
5. App **starten**. Beim ersten Start lädt der Server das Standard-Modell von
   HuggingFace herunter; es landet in `/share/wyoming-speechcatcher`
   (persistent, überlebt Neustarts und Updates der App).

> **Hinweis:** Die App läuft im Host-Netzwerk (`host_network: true`) und bindet
> direkt auf dem konfigurierten Port. Das Wyoming-Protokoll hat **keine
> Authentifizierung** — den Port daher nur im vertrauenswürdigen LAN verwenden
> und nicht ungesichert ins Internet exponieren.

---

## Konfiguration

Alle Parameter werden in der App-Konfiguration (HA-UI) gesetzt — kein
Bearbeiten von Dateien nötig. Mehrwertige Optionen erscheinen als Dropdowns,
jede Option hat eine Kurzbeschreibung direkt in der UI.

| Option | Typ | Default | Beschreibung |
|---|---|---|---|
| `port` | int (1024–65535) | `10300` | Port, auf dem der Wyoming-Server lauscht (TCP) |
| `model` | Auswahl | `de_streaming_transformer_m` | ASR-Modell (siehe [Modelle](#modelle)) |
| `language` | Auswahl (`de`/`en`/`es`) | `de` | Standard-Sprache der Erkennung |
| `preload_languages` | Liste | `[de]` | Sprachen, die beim Serverstart in den Speicher geladen werden (mehr = mehr RAM, aber sofort verfügbar) |
| `beam_size` | int (2–20) | `5` | Breite der Strahlsuche. Höher = minimal genauer, langsamer |
| `ctc_weight` | float (0.0–1.0) | `0.3` | Gewicht des akustischen CTC-Modells. Höher = genaueres Zuhören, holpriger; niedriger = flüssiger, aber mehr Wiederholungs-Risiko |
| `decoder` | Auswahl (`native`/`espnet`) | `native` | Decoder-Implementierung. `native` = schnelle eigene Implementierung; `espnet` = etablierter Referenz-Decoder |
| `stream_transcript` | bool | `false` | Partielle Transkripte streamen (HA zeigt Text schon während des Sprechens an) |
| `external_vad` | bool | `false` | Externes VAD voraussetzen (für Wake-Word-Setups) |
| `num_cpu` | int (1–32) | `1` | PyTorch-CPU-Threads. Höher = schneller auf Mehrkern-CPU |
| `penalty` | float (-1.0–1.0) | `0.0` | Längenstrafe (Insertion Penalty). Nur bei `decoder=espnet`; positiv = kürzere Texte, negativ = längere |
| `disable_bbd` | bool | `false` | Block Boundary Detection ausschalten (nur `decoder=native`). BBD verhindert Wortwiederholungen; deaktivieren nur bei Problemen |
| `debug` | bool | `false` | Ausführliche DEBUG-Logs aktivieren |

---

## Modelle

Alle Modelle sind EspNet2-Streaming-Transformer von
[speechcatcher-asr](https://huggingface.co/speechcatcher) (Attribution:
Benjamin Milde). Die App meldet alle sieben Modelle an Home Assistant;
geladen wird das per `model` / `language` gewählte.

| Kurzname | Sprache | Größe | Hinweis |
|---|---|---|---|
| `de_streaming_transformer_m` | Deutsch | M (~500 MB RAM) | **Default** — guter Kompromiss für RPi 4 / x86-CPU |
| `de_streaming_transformer_l` | Deutsch | L | bessere Qualität, mehr RAM/CPU |
| `de_streaming_transformer_xl` | Deutsch | XL (26k h Training) | beste deutsche Qualität; **nicht für RPi 3** |
| `en_streaming_transformer_m` | Englisch | M | Default für `--language en` |
| `en_streaming_transformer_l` | Englisch | L | |
| `es_streaming_transformer_m` | Spanisch | M | Default für `--language es` |
| `es_streaming_transformer_l` | Spanisch | L | |

---

## In Home Assistant nutzen

1. Nach dem Start der App: **Einstellungen → Geräte & Dienste → Integration
   hinzufügen → „Wyoming Protocol“**.
2. Host `localhost` und Port `10300` (bzw. den konfigurierten Port) eintragen
   und bestätigen.
3. Danach: **Einstellungen → Sprachassistenten → Assist-Pipeline** wählen
   (oder neu anlegen) und unter **„Spracherkennung“ (Speech-to-text)** den
   Eintrag **„speechcatcher“** auswählen.

**Hinweise:**

- **Audio-Geräte** (Mikrofone, Satelliten) werden ausschließlich in Home
  Assistant konfiguriert — die App ist passiv und geräte-agnostisch.
  Geeignete Quellen: Wyoming Satellite, ESPHome Voice, HA-Companion-App,
  Browser. Eine App bedient beliebig viele Geräte gleichzeitig.
- **Partielle Transkripte:** Option `stream_transcript` aktivieren, damit HA
  den erkannten Text schon während des Sprechens anzeigt.
- **Wake-Word-Setups:** Wenn ein externer VAD/Wake-Word-Dienst vorgeschaltet
  ist, Option `external_vad` aktivieren.

---

## Troubleshooting

| Symptom | Ursache / Lösung |
|---|---|
| App startet nicht | Log in der HA-UI prüfen. Häufigste Ursache: ungültige Option (z. B. `beam_size` außerhalb 2–20 oder `ctc_weight` außerhalb 0–1). |
| Erste Erkennung dauert Sekunden | Modell wird beim ersten Start heruntergeladen (einige hundert MB). Danach liegt es persistent in `/share/wyoming-speechcatcher`. |
| Modell-Download schlägt fehl | Netzwerk/HuggingFace nicht erreichbar. Prüfen, ob der HA-Host Internetzugang hat. |
| Hoher RAM-Verbrauch / OOM auf RPi | Das XL-Modell ist zu groß für 1-GB-Geräte. Default `*_m` verwenden; `beam_size` reduzieren (z. B. 2–3). |
| HA findet den Dienst nicht | Port prüfen (`port`-Option); Host muss `localhost` (gleicher Host) bzw. die HA-Host-IP sein. |
| Leeres Transkript | `debug` aktivieren und Log prüfen. Ggf. `beam_size` erhöhen oder Modell wechseln. |

---

## Lizenz & Attribution

Dieses Projekt (die Wyoming-App/der Server-Code) steht unter der
**MIT-Lizenz** — siehe [`LICENSE.md`](LICENSE.md) im Repository-Root.

Die verwendeten ASR-Modelle stammen von
[**speechcatcher-asr**](https://github.com/speechcatcher-asr/speechcatcher)
([HuggingFace](https://huggingface.co/speechcatcher), Copyright/Attribution:
**Benjamin Milde**). Speechcatcher selbst ist ein Open-Source-Projekt
(EspNet2-basiert); die App bindet es als Dependency ein. Für die genauen
Lizenzbedingungen von speechcatcher und der einzelnen Modelle gelten die
jeweiligen Angaben im [speechcatcher-Repository](https://github.com/speechcatcher-asr/speechcatcher)
bzw. auf den HuggingFace-Modellseiten.
