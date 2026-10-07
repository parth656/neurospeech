---
title: NeuroSpeech Therapy Pro
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
🩺 NeuroSpeech Therapy Pro
AI-powered speech practice app for cluttering and articulation, built with OpenAI Whisper, IPA phonetics and Streamlit.
![Python](https://img.shields.io/badge/python-3.11+-blue.svg) ![Streamlit](https://img.shields.io/badge/streamlit-1.53-red.svg) ![License](https://img.shields.io/badge/license-MIT-blue.svg)
🚀 Live demo: huggingface.co/spaces/parthbijpuriya/neurospeech
<!-- Add 2–3 screenshots or a GIF here, e.g. ![Demo](docs/demo.gif) -->
🌟 Features
12 therapy modules: Cluttering Control · Articulation (Bilabials, Alveolars, Velars, Fricatives) · Liquids (R, L) · Blends & Clusters · Pacing & Rhythm · Sentence Complexity · Prosody & Intonation · Repetition Drills · Conversational Practice
Speech analysis
In-browser microphone recording (or audio upload: wav, mp3, m4a, ogg, webm)
Whisper transcription (`base.en` by default, configurable via `WHISPER_MODEL_SIZE`)
Transcript match score (word- and character-level)
Speech rate measured over active speech only (leading/trailing silence excluded)
Pause detection and loudness from the waveform
IPA conversion and word-by-word comparison
Progress tracking
Private per-session history (SQLite), category stats and trend charts
Download your history as a backup and restore it later
🛠️ Tech Stack
Layer	Technology
UI	Streamlit (`st.audio_input`)
Speech-to-text	OpenAI Whisper (CPU)
Audio	FFmpeg + soundfile + NumPy
Phonetics	eng-to-ipa
Storage	SQLite
Deployment	Docker on Hugging Face Spaces
🔧 Run Locally
```bash
git clone https://github.com/parth656/neurospeech.git
cd neurospeech

# System dependencies
sudo apt-get install ffmpeg libsndfile1      # Ubuntu/Debian
brew install ffmpeg libsndfile               # macOS
# Windows: install ffmpeg from https://ffmpeg.org/download.html and add it to PATH

pip install -r requirements.txt
streamlit run src/streamlit_app.py
```
Open http://localhost:8501. The first run downloads the Whisper model. You need about 2 GB of RAM.
⚙️ Configuration
Variable	Default	Purpose
`WHISPER_MODEL_SIZE`	`base.en`	Whisper model (`tiny.en`, `base.en`, `small.en`, …)
`PERSISTENT_STORAGE_PATH`	`/tmp/neurospeech_data`	Where session databases are stored (`/data` for persistent HF storage)
`WHISPER_CACHE_DIR`	`/tmp/whisper`	Model cache directory
🌐 Deployment
Hugging Face Spaces (Docker): push this repo to a Docker Space; the app listens on port 8501. For persistent storage, enable Persistent Storage and set `PERSISTENT_STORAGE_PATH=/data`.
Streamlit Community Cloud: set the main file to `src/streamlit_app.py`. `packages.txt` installs ffmpeg and libsndfile.
📁 Project Structure
```
neurospeech/
├── src/
│   └── streamlit_app.py   # Streamlit application
├── Dockerfile             # Hugging Face Spaces runtime
├── requirements.txt       # Python dependencies
├── packages.txt           # System packages (Streamlit Cloud)
├── .gitignore
└── README.md
```
🔒 Privacy
Audio is decoded in memory and is not stored.
Transcripts and scores are saved in a database private to your browser session; other visitors cannot see or download them.
Without persistent storage, history is lost when the Space restarts, so use Download Progress to keep it.
⚠️ Disclaimer
This is a supplementary practice tool, not a medical device. Transcript matching cannot verify how individual sounds were articulated. Always consult a licensed speech-language pathologist for diagnosis and treatment.
🗺️ Roadmap
[ ] User authentication with persistent per-user history
[ ] Phoneme-level scoring (forced alignment)
[ ] PDF progress reports
[ ] Custom phrase lists
[ ] Therapist dashboard
👨‍💻 Author
Parth Bijpuriya · GitHub @parth656 · Hugging Face @parthbijpuriya
📝 License
MIT