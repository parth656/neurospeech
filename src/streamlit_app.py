# ================= HF + STREAMLIT SAFE SETUP =================
import os
import hashlib
import io
import subprocess
from pathlib import Path

WHISPER_CACHE_DIR = os.getenv("WHISPER_CACHE_DIR", "/tmp/whisper")
Path(WHISPER_CACHE_DIR).mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = os.getenv("HF_HOME", "/tmp/huggingface")
Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)

import streamlit as st
import whisper
import eng_to_ipa as ipa
import sqlite3
import datetime
import difflib
import random
import re
import numpy as np
import soundfile as sf

st.set_page_config(
    page_title="NeuroSpeech Therapy Pro",
    page_icon="🩺",
    layout="wide"
)

PERSISTENT_DIR = os.getenv("PERSISTENT_STORAGE_PATH", "/tmp/neurospeech_data")
Path(PERSISTENT_DIR).mkdir(parents=True, exist_ok=True)
DB_PATH = os.path.join(PERSISTENT_DIR, "patient_history.db")

def init_db():
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                category TEXT NOT NULL,
                target TEXT NOT NULL,
                spoken TEXT NOT NULL,
                accuracy REAL NOT NULL,
                speech_rate REAL NOT NULL,
                pause_count INTEGER,
                feedback TEXT,
                audio_duration REAL
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON sessions(timestamp DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_category ON sessions(category)")
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Database initialization failed: {e}")
        return False

init_db()

@st.cache_resource
def load_whisper_model():
    model_size = os.getenv("WHISPER_MODEL_SIZE", "base.en")
    return whisper.load_model(
        model_size,
        device="cpu",
        download_root=WHISPER_CACHE_DIR,
    )

model = None

therapy_modules = {
    "🎯 Cluttering Control": {
        "description": "Slow down, pause strategically, improve clarity",
        "exercises": [
            "I will speak slowly and clearly.",
            "Pause between each word now.",
            "Take a breath before speaking.",
            "One thought at a time please.",
            "Slow and steady wins the race."
        ],
        "tips": "Focus on: 1) Slowing speech rate 2) Adding pauses 3) Clear word boundaries"
    },
    "🔤 Articulation - Bilabials (P, B, M)": {
        "description": "Practice lip sounds",
        "exercises": [
            "Peter Piper picked a peck.",
            "Big brown bears bite.",
            "My mom makes muffins Monday.",
            "Purple paper people.",
            "Bob bought big balloons."
        ],
        "tips": "Keep lips closed, then release with controlled airflow"
    },
    "🔤 Articulation - Alveolars (T, D, N, L)": {
        "description": "Tongue tip behind teeth",
        "exercises": [
            "Tiny turtles take tea.",
            "Daddy drew a dog.",
            "Nine nice nights.",
            "Little lemon lollipops.",
            "Tell Dan to talk today."
        ],
        "tips": "Tongue touches ridge behind upper teeth"
    },
    "🔤 Articulation - Velars (K, G)": {
        "description": "Back of tongue exercises",
        "exercises": [
            "Kate can keep cookies.",
            "Go get good grades.",
            "King Kong kicked cans.",
            "Crisp crackers crunch.",
            "Gray geese go home."
        ],
        "tips": "Back of tongue rises to soft palate"
    },
    "🔤 Articulation - Fricatives (F, V, S, Z, TH)": {
        "description": "Continuous airflow sounds",
        "exercises": [
            "Five friendly foxes.",
            "Very vivid violets.",
            "Six silly snakes.",
            "Zebras zigzag zones.",
            "Think about thirty things."
        ],
        "tips": "Create friction with continuous airflow"
    },
    "🌊 Liquid Sounds (R, L)": {
        "description": "Complex tongue movements",
        "exercises": [
            "Red lorry yellow lorry.",
            "Really rural roads.",
            "Little lucky lambs leap.",
            "Round and round the rugged rock.",
            "Larry's really rare lizard."
        ],
        "tips": "R: Tongue tip up but not touching. L: Tongue tip touches ridge"
    },
    "🎭 Blends & Clusters": {
        "description": "Multiple consonants together",
        "exercises": [
            "Split spray string strong.",
            "Scrap screen script scratch.",
            "Plant plenty please.",
            "Crisp crown craft crack.",
            "Twist twelve twenty."
        ],
        "tips": "Don't insert vowels between consonants"
    },
    "⏱️ Pacing & Rhythm": {
        "description": "Control speech rate and timing",
        "exercises": [
            "I... will... speak... slowly.",
            "One. Two. Three. Four. Five.",
            "Stop and think before you speak.",
            "Breathe. Pause. Continue speaking.",
            "Take your time with every word."
        ],
        "tips": "Use metronome technique: pause after each phrase"
    },
    "🗣️ Sentence Complexity": {
        "description": "Maintain clarity in longer phrases",
        "exercises": [
            "The cat sat on the mat.",
            "She sells sea shells by the sea shore.",
            "How much wood would a woodchuck chuck?",
            "Peter Piper picked a peck of pickled peppers.",
            "Betty Botter bought some butter but the butter was bitter."
        ],
        "tips": "Break into chunks, pause at natural boundaries"
    },
    "🎵 Prosody & Intonation": {
        "description": "Stress, pitch, and melody of speech",
        "exercises": [
            "Are you HAPPY today?",
            "I LOVE chocolate ice cream.",
            "What TIME is it now?",
            "Please HELP me with this.",
            "That's AMAZING news!"
        ],
        "tips": "Emphasize capitalized words with pitch change"
    },
    "🔄 Repetition Drills": {
        "description": "Build motor memory",
        "exercises": [
            "Pa-Pa-Pa-Pa-Pa",
            "Ta-Ta-Ta-Ta-Ta",
            "Ka-Ka-Ka-Ka-Ka",
            "Pa-Ta-Ka-Pa-Ta-Ka",
            "La-La-La-La-La"
        ],
        "tips": "Repeat 10 times, gradually increase speed while maintaining clarity"
    },
    "💬 Conversational Practice": {
        "description": "Natural speech situations",
        "exercises": [
            "Hello, how are you today?",
            "Can you help me find something?",
            "What would you like for dinner?",
            "I need to make an appointment.",
            "Thank you very much for your help."
        ],
        "tips": "Practice at normal conversation speed with clear articulation"
    }
}

def normalize_text(text):
    return re.sub(r"[^a-z0-9']+", " ", str(text).lower()).strip()

def rms_db(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return -100.0
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
    return round(20 * np.log10(max(rms, 1e-7)), 1)

def estimate_silence(audio, sample_rate, threshold_db=-42.0):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0 or sample_rate <= 0:
        return 0.0, 0
    frame = max(int(sample_rate * 0.02), 1)
    usable = len(audio) - (len(audio) % frame)
    if usable <= 0:
        return 0.0, 0
    frames = audio[:usable].reshape(-1, frame)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-7))
    silent = db < threshold_db
    ratio = float(np.mean(silent))
    transitions = np.diff(np.r_[False, silent, False].astype(np.int8))
    runs = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]
    pauses = int(sum((e - s) * frame / sample_rate >= 0.20 for s, e in zip(runs, ends)))
    return round(ratio * 100, 1), pauses

def compare_words(target_words, spoken_words):
    matcher = difflib.SequenceMatcher(None, target_words, spoken_words)
    rows = []
    correct = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            correct += i2 - i1
            for word in target_words[i1:i2]:
                rows.append(("✓", word, word))
        elif tag == "replace":
            n = max(i2 - i1, j2 - j1)
            for k in range(n):
                expected = target_words[i1 + k] if i1 + k < i2 else "[missing]"
                said = spoken_words[j1 + k] if j1 + k < j2 else "[missing]"
                rows.append(("✗", expected, said))
        elif tag == "delete":
            for word in target_words[i1:i2]:
                rows.append(("✗", word, "[missing]"))
        elif tag == "insert":
            for word in spoken_words[j1:j2]:
                rows.append(("+", "[extra]", word))
    return rows, correct

def analyze_speech_advanced(target, spoken, audio_duration, audio_data=None, sample_rate=0):
    try:
        target_clean = normalize_text(target)
        spoken_clean = normalize_text(spoken)
        if not target_clean or not spoken_clean:
            return {"match": 0.0, "wpm": 0.0, "pause_count": 0, "silence_pct": 0.0, "loudness_db": -100.0, "word_rows": [], "feedback": "No usable speech transcript was detected."}
        if audio_duration <= 0:
            return {"match": 0.0, "wpm": 0.0, "pause_count": 0, "silence_pct": 0.0, "loudness_db": -100.0, "word_rows": [], "feedback": "Invalid audio duration."}
        target_words = target_clean.split()
        spoken_words = spoken_clean.split()
        word_rows, correct_words = compare_words(target_words, spoken_words)
        word_ratio = difflib.SequenceMatcher(None, target_words, spoken_words).ratio()
        char_ratio = difflib.SequenceMatcher(None, target_clean, spoken_clean).ratio()
        transcript_match = round((0.75 * word_ratio + 0.25 * char_ratio) * 100, 1)
        word_count = len(spoken_words)
        wpm = round((word_count / audio_duration) * 60, 1)
        silence_pct, waveform_pauses = estimate_silence(audio_data, sample_rate) if audio_data is not None and sample_rate else (0.0, 0)
        loudness = rms_db(audio_data) if audio_data is not None else -100.0
        try:
            target_ipa = ipa.convert(target_clean)
            spoken_ipa = ipa.convert(spoken_clean)
        except Exception:
            target_ipa = "Unavailable"
            spoken_ipa = "Unavailable"
        feedback = [
            f"Target IPA: /{target_ipa}/",
            f"Spoken IPA: /{spoken_ipa}/",
            f"Transcript match: {transcript_match}%",
            f"Speech rate: {wpm} words/minute",
            f"Estimated waveform silence: {silence_pct}%"
        ]
        if waveform_pauses:
            feedback.append(f"Estimated pauses (>=0.20s): {waveform_pauses}")
        if wpm > 200:
            feedback.append("Try a slower pace and add deliberate pauses.")
        elif wpm >= 120:
            feedback.append("Pace is in a common conversational range; focus on consistency.")
        elif wpm >= 80:
            feedback.append("Pace is relatively slow; keep speech comfortable and natural.")
        else:
            feedback.append("Pace is very slow; avoid forcing a rate and prioritize comfortable speech.")
        feedback.append("\n--- Word Analysis ---")
        for symbol, expected, said in word_rows[:30]:
            feedback.append(f"{symbol} Expected: {expected} | Heard: {said}")
        feedback.append("\nNote: Whisper transcript matching cannot determine whether an individual phoneme was articulated correctly. Use this as practice feedback, not a clinical diagnosis or articulation score.")
        return {"match": transcript_match, "wpm": wpm, "pause_count": waveform_pauses, "silence_pct": silence_pct, "loudness_db": loudness, "word_rows": word_rows, "feedback": "\n".join(feedback)}
    except Exception as exc:
        return {"match": 0.0, "wpm": 0.0, "pause_count": 0, "silence_pct": 0.0, "loudness_db": -100.0, "word_rows": [], "feedback": f"Analysis error: {exc}"}

def save_session(category, target, spoken, accuracy, speech_rate, pause_count, feedback, duration):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        c.execute("INSERT INTO sessions (timestamp, category, target, spoken, accuracy, speech_rate, pause_count, feedback, audio_duration) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (ts, category, target, spoken, accuracy, speech_rate, pause_count, feedback, duration))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Failed to save session: {e}")
        return False

def get_history(limit=20):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT timestamp, category, accuracy, speech_rate FROM sessions ORDER BY timestamp DESC LIMIT ?", (limit,))
        rows = c.fetchall()
        conn.close()
        return rows
    except Exception as e:
        st.error(f"Failed to fetch history: {e}")
        return []

def get_category_stats():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT category, AVG(accuracy) as avg_acc, COUNT(*) as count FROM sessions GROUP BY category ORDER BY count DESC")
        rows = c.fetchall()
        conn.close()
        return rows
    except Exception as e:
        st.error(f"Failed to fetch stats: {e}")
        return []

st.title("🩺 NeuroSpeech Therapy Pro")
st.markdown("### Comprehensive Speech Therapy for Cluttering & Articulation")

with st.sidebar:
    st.header("⚙️ Therapy Settings")
    category = st.selectbox("Select Module", list(therapy_modules.keys()), help="Choose your target area")
    st.markdown("---")
    st.markdown(f"**{therapy_modules[category]['description']}**")
    st.info(therapy_modules[category]["tips"])
    st.markdown("---")
    st.subheader("📊 Your Progress")
    if os.path.exists(DB_PATH):
        try:
            with open(DB_PATH, "rb") as f:
                st.download_button("💾 Download Progress", f, file_name="my_speech_progress.db", help="Save your therapy history")
        except Exception as e:
            st.warning(f"Cannot download database: {e}")
    uploaded_db = st.file_uploader("📂 Restore Progress", type=["db"], help="Upload previously saved database")
    if uploaded_db:
        try:
            restore_dir = Path(DB_PATH).parent
            candidate = restore_dir / ".restore_candidate.db"
            raw = uploaded_db.getvalue()
            if len(raw) > 25 * 1024 * 1024:
                raise ValueError("Database backup is larger than 25 MB.")
            candidate.write_bytes(raw)
            conn = sqlite3.connect(candidate, timeout=5)
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            conn.close()
            if "sessions" not in tables:
                raise ValueError("This file is not a NeuroSpeech session database.")
            backup = restore_dir / "patient_history.before_restore.db"
            if Path(DB_PATH).exists():
                Path(DB_PATH).replace(backup)
            candidate.replace(DB_PATH)
            st.success("✅ Progress restored safely. Previous data was backed up.")
            st.rerun()
        except Exception as e:
            try:
                candidate.unlink(missing_ok=True)
            except Exception:
                pass
            st.error(f"Failed to restore database: {e}")
    stats = get_category_stats()
    if stats:
        for cat, avg_acc, count in stats[:5]:
            short_name = cat.split(" ")[-1][:15] if " " in cat else cat[:15]
            st.metric(short_name, f"{avg_acc:.1f}%", f"{count} sessions")
    else:
        st.info("Start practicing to see stats")

col1, col2 = st.columns([2, 1])

with col1:
    if "current_target" not in st.session_state or "category" not in st.session_state:
        st.session_state.current_target = random.choice(therapy_modules[category]["exercises"])
        st.session_state.category = category
    if st.session_state.category != category:
        st.session_state.current_target = random.choice(therapy_modules[category]["exercises"])
        st.session_state.category = category
    if st.button("🎲 Get New Phrase", use_container_width=True):
        st.session_state.current_target = random.choice(therapy_modules[category]["exercises"])
        st.rerun()
    st.markdown("---")
    st.markdown("### 🎯 Target Phrase:")
    st.markdown(f"# {st.session_state.current_target}")
    try:
        ipa_text = ipa.convert(st.session_state.current_target)
        st.caption(f"IPA: /{ipa_text}/")
    except Exception:
        st.caption("IPA conversion unavailable")
    st.markdown("---")
    st.markdown("**Instructions:** Click the microphone button below and speak the target phrase clearly.")
    st.caption("🎙️ Allow microphone access when your browser asks. If recording fails, use the audio-file fallback below.")

    audio = st.audio_input(
        "🎙️ Record your voice",
        sample_rate=16000,
        key="neurospeech_voice_recorder",
        help="Record a short 1–30 second English speech sample."
    )

    uploaded_audio = st.file_uploader(
        "Or upload a recording",
        type=["wav", "mp3", "m4a", "ogg", "webm"],
        key="neurospeech_audio_upload",
        help="Use this if the microphone recorder shows an error."
    )

    if audio is not None:
        audio_bytes = audio.getvalue()
        audio_source_name = "microphone.wav"
    elif uploaded_audio is not None:
        audio_bytes = uploaded_audio.getvalue()
        audio_source_name = uploaded_audio.name or "uploaded_audio"
    else:
        audio_bytes = None
        audio_source_name = ""

    if audio_bytes:
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()
        if st.session_state.get("last_audio_hash") == audio_hash:
            st.info("This recording has already been analyzed. Record or upload a new sample.")
            audio_bytes = None

def decode_audio_bytes(audio_bytes):
    """Decode any supported upload to mono 16 kHz float32 audio for Whisper."""
    if not audio_bytes:
        raise ValueError("The recording is empty.")
    if len(audio_bytes) > 25 * 1024 * 1024:
        raise ValueError("Audio file is larger than 25 MB.")

    import subprocess

    # FFmpeg is the primary decoder because browser uploads may be WAV, WebM,
    # MP3, M4A, or OGG. Converting here also guarantees Whisper gets 16 kHz mono.
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0",
            "-f", "wav", "-ac", "1", "-ar", "16000",
            "pipe:1",
        ],
        input=audio_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", errors="replace").strip()
        # Keep a direct WAV fallback for unusual FFmpeg/container failures.
        try:
            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            if sample_rate != 16000:
                raise ValueError("Audio is not 16 kHz and FFmpeg conversion failed.")
            return np.asarray(audio_data, dtype=np.float32), int(sample_rate)
        except Exception as fallback_error:
            raise RuntimeError(
                "Could not decode the recording. "
                f"FFmpeg: {detail[-300:] or 'unknown error'}; "
                f"WAV fallback: {fallback_error}"
            ) from fallback_error

    try:
        audio_data, sample_rate = sf.read(io.BytesIO(proc.stdout), dtype="float32")
    except Exception as exc:
        raise RuntimeError(f"Decoded audio could not be read: {exc}") from exc

    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)
    audio_data = np.asarray(audio_data, dtype=np.float32)
    sample_rate = int(sample_rate)

    if sample_rate != 16000:
        raise RuntimeError(f"Audio normalization failed: expected 16000 Hz, got {sample_rate} Hz.")
    if audio_data.size == 0:
        raise ValueError("The recording contains no audio samples.")
    if not np.isfinite(audio_data).all():
        raise ValueError("The recording contains invalid audio samples.")
    return audio_data, sample_rate


def transcribe_audio(model, audio_data):
    """Transcribe a normalized 16 kHz waveform with conservative Whisper settings."""
    result = model.transcribe(
        audio_data,
        fp16=False,
        language="en",
        task="transcribe",
        temperature=0,
        condition_on_previous_text=False,
        verbose=False,
    )
    return (result.get("text") or "").strip()


if audio_bytes:
    try:
        if len(audio_bytes) > 25 * 1024 * 1024:
            raise ValueError("Audio file is larger than 25 MB.")

        with st.spinner("🔧 Preparing your recording..."):
            audio_data, sample_rate = decode_audio_bytes(audio_bytes)

        duration = len(audio_data) / sample_rate
        if duration > 30:
            st.error("❌ Audio too long. Please record less than 30 seconds.")
        elif duration < 0.5:
            st.error("❌ Audio too short. Please speak the full phrase.")
        else:
            with st.spinner("🧠 Loading speech model and analyzing your speech..."):
                model = load_whisper_model()
                if model is None:
                    raise RuntimeError("Whisper could not be loaded. Check the Space runtime logs.")
                spoken_text = transcribe_audio(model, audio_data)

            if not spoken_text:
                st.warning("⚠️ No speech was detected. Please move closer to the microphone and speak the complete phrase.")
            else:
                analysis = analyze_speech_advanced(
                    st.session_state.current_target,
                    spoken_text,
                    duration,
                    audio_data=audio_data,
                    sample_rate=sample_rate,
                )
                accuracy = analysis["match"]
                speech_rate = analysis["wpm"]
                pause_count = analysis["pause_count"]
                feedback = analysis["feedback"]
                st.markdown("### 📝 You said:")
                st.info(f'"{spoken_text}"')
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Transcript Match", f"{accuracy}%")
                with col_b:
                    st.metric("Speech Rate", f"{speech_rate} wpm")
                with col_c:
                    st.metric("Duration", f"{duration:.1f}s")
                with col_d:
                    st.metric("Estimated Pauses", pause_count)
                st.caption(f"Waveform silence: {analysis['silence_pct']}% · Average loudness: {analysis['loudness_db']} dBFS")
                if accuracy >= 85:
                    st.success("🎉 Excellent phrase match. Keep the same controlled pace.")
                elif accuracy >= 70:
                    st.warning("👍 Good effort. Review the word-by-word feedback.")
                elif accuracy >= 50:
                    st.warning("💪 Keep practicing. Slow down and focus on each target word.")
                else:
                    st.error("🔄 Try again and speak the complete target phrase clearly.")
                with st.expander("📋 Detailed Phonetic Feedback"):
                    st.code(feedback, language="text")
                if save_session(
                    category,
                    st.session_state.current_target,
                    spoken_text,
                    accuracy,
                    speech_rate,
                    pause_count,
                    feedback,
                    duration,
                ):
                    st.session_state.last_audio_hash = audio_hash
                    st.success("✅ Session saved to your progress history.")
    except subprocess.TimeoutExpired:
        st.error("❌ Audio decoding timed out. Please record a shorter clip and try again.")
    except Exception as e:
        st.error(f"❌ Error processing audio: {e}")
        st.caption("If microphone recording fails, try the upload box with a WAV/MP3/M4A/OGG/WebM file.")

with col2:
    st.markdown("### 💡 Quick Tips")
    if "Cluttering" in category:
        st.markdown("""
        **For Cluttering:**
        - 🐢 Slow down deliberately
        - 🫁 Breathe between phrases
        - ✋ Pause after each thought
        - 🎯 One word at a time
        - 📢 Exaggerate articulation
        """)
    else:
        st.markdown("""
        **For Articulation:**
        - 👄 Watch your mouth in mirror
        - 🔊 Exaggerate the target sound
        - 🐢 Practice slowly first
        - 🔁 Repeat 10 times daily
        - 📹 Record yourself
        """)

st.markdown("---")
st.subheader("📈 Progress Tracking")
if st.checkbox("Show Detailed History"):
    history = get_history()
    if history:
        st.markdown("**Recent Sessions:**")
        for ts, cat, acc, rate in history:
            col_x, col_y, col_z = st.columns([3, 2, 2])
            with col_x:
                short_cat = cat.split(' ')[-1][:20] if ' ' in cat else cat[:20]
                st.text(f"{ts} - {short_cat}")
            with col_y:
                st.text(f"Match: {acc:.1f}%")
            with col_z:
                st.text(f"Rate: {rate:.0f} wpm")
        st.markdown("---")
        st.markdown("**Accuracy Trend:**")
        scores = [h[2] for h in history][::-1]
        if scores:
            st.line_chart(scores)
        st.markdown("**Speech Rate Trend:**")
        rates = [h[3] for h in history if h[3] > 0][::-1]
        if rates:
            st.line_chart(rates)
    else:
        st.info("No history yet. Start practicing!")

st.markdown("---")
st.caption("💙 Practice daily for best results. Track progress over weeks.")
st.caption("⚠️ This app uses AI for analysis. Always consult a licensed speech therapist for professional advice.")
