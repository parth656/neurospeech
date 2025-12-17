# ================= HF + STREAMLIT SAFE SETUP =================
import os
os.environ["TRANSFORMERS_CACHE"] = "/tmp"
os.environ["HF_HOME"] = "/tmp"

import streamlit as st
import whisper
import eng_to_ipa as ipa
import sqlite3
import datetime
import difflib
import random
import numpy as np
import soundfile as sf

# ================= CONFIG =================
st.set_page_config(
    page_title="NeuroSpeech Pro",
    page_icon="🩺",
    layout="centered"
)

# ================= DATABASE =================
DB_PATH = "patient_history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            timestamp TEXT,
            target TEXT,
            spoken TEXT,
            accuracy REAL,
            feedback TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ================= LOAD WHISPER (LOW RAM, NO GPU) =================
@st.cache_resource
def load_whisper_model():
    return whisper.load_model(
        "tiny",        # 🔥 REQUIRED for HF free tier
        device="cpu"
    )

model = load_whisper_model()

# ================= ANALYSIS =================
def analyze_speech(target, spoken):
    target_ipa = ipa.convert(target)
    spoken_ipa = ipa.convert(spoken)

    accuracy = round(
        difflib.SequenceMatcher(
            None, target.lower(), spoken.lower()
        ).ratio() * 100,
        1
    )

    feedback = (
        f"Target IPA: /{target_ipa}/\n"
        f"Spoken IPA: /{spoken_ipa}/"
    )

    return accuracy, feedback

def save_session(target, spoken, accuracy, feedback):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        "INSERT INTO sessions VALUES (?, ?, ?, ?, ?)",
        (ts, target, spoken, accuracy, feedback)
    )
    conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT timestamp, accuracy FROM sessions ORDER BY rowid DESC LIMIT 10"
    )
    rows = c.fetchall()
    conn.close()
    return rows

# ================= UI =================
st.title("🩺 NeuroSpeech Pro")
st.markdown("### Clinical Speech Therapy Assistant")

phrases = {
    "Level 1: Bilabials (B, P, M)": [
        "Big black bears.",
        "Please pay promptly.",
        "My mom makes muffins."
    ],
    "Level 2: Alveolars (T, D, N)": [
        "Tiny turtles take tea.",
        "Daddy draws dogs.",
        "Nine nice nights."
    ],
    "Level 3: Liquids (R, L)": [
        "Red lorries, yellow lorries.",
        "Real rural roads.",
        "Little lucky lambs."
    ],
    "Level 4: Tongue Twisters": [
        "She sells sea shells.",
        "Specific pacific statistics.",
        "Irish wristwatch."
    ]
}

with st.sidebar:
    st.header("Settings")
    category = st.selectbox("Select Module", list(phrases.keys()))

if "current_target" not in st.session_state:
    st.session_state.current_target = random.choice(phrases[category])

if st.button("🎲 Get New Phrase"):
    st.session_state.current_target = random.choice(phrases[category])

st.markdown("---")
st.markdown(f"### 🎯 Target: **{st.session_state.current_target}**")
st.caption(f"IPA: /{ipa.convert(st.session_state.current_target)}/")
st.markdown("---")

# ================= AUDIO INPUT =================
audio = st.audio_input("🎙️ Record your voice (max 10 seconds)")

if audio is not None:
    # Save audio
    with open("temp.wav", "wb") as f:
        f.write(audio.read())

    # Load audio manually (NO ffmpeg)
    audio_data, sample_rate = sf.read("temp.wav")

    # Stereo → mono
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Duration check
    duration = len(audio_data) / sample_rate
    if duration > 10:
        st.error("❌ Audio too long. Please record less than 10 seconds.")
        st.stop()

    # Transcribe WITHOUT ffmpeg
    with st.spinner("🧠 Analyzing speech..."):
        result = model.transcribe(
            audio_data,
            fp16=False,
            language="en"
        )
        spoken_text = result["text"].strip()

    # Analyze
    accuracy, feedback = analyze_speech(
        st.session_state.current_target,
        spoken_text
    )

    # Display
    st.subheader(f'You said: "{spoken_text}"')
    st.metric("Accuracy", f"{accuracy}%")

    if accuracy > 85:
        st.success("Excellent articulation!")
    elif accuracy > 60:
        st.warning("Good effort — refine pronunciation.")
    else:
        st.error("Try again — focus on clarity.")

    with st.expander("Phonetic Feedback"):
        st.text(feedback)

    save_session(
        st.session_state.current_target,
        spoken_text,
        accuracy,
        feedback
    )

# ================= HISTORY =================
st.markdown("---")
if st.checkbox("Show Progress History"):
    history = get_history()
    if history:
        scores = [h[1] for h in history][::-1]
        st.line_chart(scores)
    else:
        st.info("No history yet.")
