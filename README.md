---
title: NeuroSpeech Therapy Pro
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
---

# NeuroSpeech Therapy Pro

NeuroSpeech is an AI-assisted speech-practice application for articulation and speech-flow exercises. It combines audio capture or upload, Whisper transcription, IPA-based word feedback, speech-rate and pause analysis, and local session-history features.

**Live demo:** https://huggingface.co/spaces/parthbijpuriya/neurospeech

## Features

- Audio recording in the browser or audio-file upload.
- Speech-to-text using OpenAI Whisper.
- Transcript comparison and word-level feedback.
- IPA conversion and word-by-word pronunciation comparison.
- Speech-rate, pause, and waveform-based loudness analysis.
- Session history and progress views backed by SQLite.
- Docker-based deployment for Hugging Face Spaces.

## Technology

- Python 3.11+
- Streamlit
- OpenAI Whisper
- NumPy and SoundFile
- eng-to-ipa
- SQLite
- FFmpeg
- Docker

## Run locally

```bash
git clone https://github.com/parth656/neurospeech.git
cd neurospeech
```

Install system dependencies:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1
```

Create a virtual environment and install Python packages:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run src/streamlit_app.py
```

Open the local URL printed by Streamlit. The first run may download Whisper model files and require additional time and disk space.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `WHISPER_MODEL_SIZE` | `base.en` | Selects the Whisper model size |
| `PERSISTENT_STORAGE_PATH` | `/tmp/neurospeech_data` | Stores session database files |
| `WHISPER_CACHE_DIR` | `/tmp/whisper` | Whisper model cache directory |

Use persistent storage in deployments if progress should survive restarts. Confirm the exact environment configuration in the current application before deploying.

## Privacy and limitations

Audio and speech metrics can be sensitive. Explain where recordings and history are stored, avoid collecting more than needed, and provide a clear deletion path. This is a practice aid, not a clinical diagnostic system or a substitute for a speech-language pathologist. Transcript and phonetic comparisons can be wrong, especially with accents, background noise, and non-English speech.

## Development checklist

- [ ] Add screenshots or a short demo GIF.
- [ ] Test malformed audio, long recordings, and model-download failures.
- [ ] Add automated tests for audio preprocessing and scoring logic.
- [ ] Document supported languages and known limitations.
- [ ] Verify persistence and data isolation in the deployed environment.
