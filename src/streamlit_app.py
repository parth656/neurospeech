# ================= NeuroSpeech Therapy Pro =================
import os
import io
import uuid
import random
import re
import sqlite3
import difflib
import hashlib
import datetime
import subprocess
from pathlib import Path

WHISPER_CACHE_DIR = os.getenv("WHISPER_CACHE_DIR", "/tmp/whisper")
Path(WHISPER_CACHE_DIR).mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = os.getenv("HF_HOME", "/tmp/huggingface")
Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import soundfile as sf
import streamlit as st
import whisper
import eng_to_ipa as ipa

st.set_page_config(page_title="NeuroSpeech Therapy Pro", page_icon="🩺", layout="wide")

MAX_AUDIO_BYTES = 25 * 1024 * 1024
MAX_DB_BYTES = 25 * 1024 * 1024
TARGET_SR = 16000
PAUSE_MIN_SEC = 0.20

# ---------------------------------------------------------------------------
# Per-session storage. Each browser session gets its own SQLite file, so
# visitors never see, download or overwrite each other's history.
# ---------------------------------------------------------------------------
PERSISTENT_DIR = Path(os.getenv("PERSISTENT_STORAGE_PATH", "/tmp/neurospeech_data"))
SESSIONS_DIR = PERSISTENT_DIR / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

if "user_id" not in st.session_state:
    st.session_state.user_id = uuid.uuid4().hex
DB_PATH = SESSIONS_DIR / f"{st.session_state.user_id}.db"

REQUIRED_COLUMNS = {
    "id", "timestamp", "category", "target", "spoken", "accuracy",
    "speech_rate", "pause_count", "feedback", "audio_duration",
}


def db_connect(path=DB_PATH):
    return sqlite3.connect(str(path), timeout=10)


def init_db(path=DB_PATH):
    try:
        with db_connect(path) as conn:
            conn.execute("""
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON sessions(timestamp DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON sessions(category)")
        return True
    except Exception as e:
        st.error(f"Database initialization failed: {e}")
        return False


init_db()


@st.cache_resource(show_spinner=False)
def load_whisper_model():
    model_size = os.getenv("WHISPER_MODEL_SIZE", "base.en")
    return whisper.load_model(model_size, device="cpu", download_root=WHISPER_CACHE_DIR)


therapy_modules = {
    "🎯 Cluttering Control": {
        "short": "Cluttering",
        "description": "Slow down, pause strategically, improve clarity",
        "exercises": [
            "I will speak slowly and clearly.",
            "Pause between each word now.",
            "Take a breath before speaking.",
            "One thought at a time please.",
            "Slow and steady wins the race.",
        ],
        "tips": "Focus on: 1) Slowing speech rate 2) Adding pauses 3) Clear word boundaries",
    },
    "🔤 Articulation - Bilabials (P, B, M)": {
        "short": "Bilabials",
        "description": "Practice lip sounds",
        "exercises": [
            "Peter Piper picked a peck.",
            "Big brown bears bite.",
            "My mom makes muffins Monday.",
            "Purple paper people.",
            "Bob bought big balloons.",
        ],
        "tips": "Keep lips closed, then release with controlled airflow",
    },
    "🔤 Articulation - Alveolars (T, D, N, L)": {
        "short": "Alveolars",
        "description": "Tongue tip behind teeth",
        "exercises": [
            "Tiny turtles take tea.",
            "Daddy drew a dog.",
            "Nine nice nights.",
            "Little lemon lollipops.",
            "Tell Dan to talk today.",
        ],
        "tips": "Tongue touches ridge behind upper teeth",
    },
    "🔤 Articulation - Velars (K, G)": {
        "short": "Velars",
        "description": "Back of tongue exercises",
        "exercises": [
            "Kate can keep cookies.",
            "Go get good grades.",
            "King Kong kicked cans.",
            "Crisp crackers crunch.",
            "Gray geese go home.",
        ],
        "tips": "Back of tongue rises to soft palate",
    },
    "🔤 Articulation - Fricatives (F, V, S, Z, TH)": {
        "short": "Fricatives",
        "description": "Continuous airflow sounds",
        "exercises": [
            "Five friendly foxes.",
            "Very vivid violets.",
            "Six silly snakes.",
            "Zebras zigzag zones.",
            "Think about thirty things.",
        ],
        "tips": "Create friction with continuous airflow",
    },
    "🌊 Liquid Sounds (R, L)": {
        "short": "Liquids",
        "description": "Complex tongue movements",
        "exercises": [
            "Red lorry yellow lorry.",
            "Really rural roads.",
            "Little lucky lambs leap.",
            "Round and round the rugged rock.",
            "Larry's really rare lizard.",
        ],
        "tips": "R: Tongue tip up but not touching. L: Tongue tip touches ridge",
    },
    "🎭 Blends & Clusters": {
        "short": "Blends",
        "description": "Multiple consonants together",
        "exercises": [
            "Split spray string strong.",
            "Scrap screen script scratch.",
            "Plant plenty please.",
            "Crisp crown craft crack.",
            "Twist twelve twenty.",
        ],
        "tips": "Don't insert vowels between consonants",
    },
    "⏱️ Pacing & Rhythm": {
        "short": "Pacing",
        "description": "Control speech rate and timing",
        "exercises": [
            "I... will... speak... slowly.",
            "One. Two. Three. Four. Five.",
            "Stop and think before you speak.",
            "Breathe. Pause. Continue speaking.",
            "Take your time with every word.",
        ],
        "tips": "Use metronome technique: pause after each phrase",
    },
    "🗣️ Sentence Complexity": {
        "short": "Sentences",
        "description": "Maintain clarity in longer phrases",
        "exercises": [
            "The cat sat on the mat.",
            "She sells sea shells by the sea shore.",
            "How much wood would a woodchuck chuck?",
            "Peter Piper picked a peck of pickled peppers.",
            "Betty Botter bought some butter but the butter was bitter.",
        ],
        "tips": "Break into chunks, pause at natural boundaries",
    },
    "🎵 Prosody & Intonation": {
        "short": "Prosody",
        "description": "Stress, pitch, and melody of speech",
        "exercises": [
            "Are you HAPPY today?",
            "I LOVE chocolate ice cream.",
            "What TIME is it now?",
            "Please HELP me with this.",
            "That's AMAZING news!",
        ],
        "tips": "Emphasize capitalized words with pitch change",
    },
    "🔄 Repetition Drills": {
        "short": "Repetition",
        "description": "Build motor memory",
        "exercises": [
            "Pa-Pa-Pa-Pa-Pa",
            "Ta-Ta-Ta-Ta-Ta",
            "Ka-Ka-Ka-Ka-Ka",
            "Pa-Ta-Ka-Pa-Ta-Ka",
            "La-La-La-La-La",
        ],
        "tips": "Repeat 10 times, gradually increase speed while maintaining clarity",
    },
    "💬 Conversational Practice": {
        "short": "Conversation",
        "description": "Natural speech situations",
        "exercises": [
            "Hello, how are you today?",
            "Can you help me find something?",
            "What would you like for dinner?",
            "I need to make an appointment.",
            "Thank you very much for your help.",
        ],
        "tips": "Practice at normal conversation speed with clear articulation",
    },
}


def short_name(category):
    """Readable label for a stored category (old rows may use unknown names)."""
    module = therapy_modules.get(category)
    if module:
        return module["short"]
    return re.sub(r"[^\w\s-]", "", str(category)).strip()[:20] or "Other"


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def normalize_text(text):
    return re.sub(r"[^a-z0-9']+", " ", str(text).lower()).strip()


def rms_db(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return -100.0
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
    return round(20 * np.log10(max(rms, 1e-7)), 1)


def speech_bounds(silent):
    """Index of first and last non-silent frame, or None if all silent."""
    voiced = np.flatnonzero(~silent)
    if voiced.size == 0:
        return None
    return int(voiced[0]), int(voiced[-1]) + 1


def estimate_silence(audio, sample_rate, threshold_db=-42.0):
    """Return (silence %, internal pause count, active speech seconds).

    Leading/trailing silence is excluded, so the time before you start
    speaking and after you finish is not counted as a pause or in the rate.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0 or sample_rate <= 0:
        return 0.0, 0, 0.0
    frame = max(int(sample_rate * 0.02), 1)
    usable = len(audio) - (len(audio) % frame)
    if usable <= 0:
        return 0.0, 0, 0.0
    frames = audio[:usable].reshape(-1, frame)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-7))
    silent = db < threshold_db

    bounds = speech_bounds(silent)
    if bounds is None:
        return 100.0, 0, 0.0
    start, end = bounds
    inner = silent[start:end]

    ratio = float(np.mean(inner)) if inner.size else 0.0
    transitions = np.diff(np.r_[False, inner, False].astype(np.int8))
    runs = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]
    pauses = int(sum((e - s) * frame / sample_rate >= PAUSE_MIN_SEC for s, e in zip(runs, ends)))
    active_sec = (end - start) * frame / sample_rate
    return round(ratio * 100, 1), pauses, round(active_sec, 2)


def compare_words(target_words, spoken_words):
    matcher = difflib.SequenceMatcher(None, target_words, spoken_words, autojunk=False)
    rows, correct = [], 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            correct += i2 - i1
            rows.extend(("✓", w, w) for w in target_words[i1:i2])
        elif tag == "replace":
            for k in range(max(i2 - i1, j2 - j1)):
                expected = target_words[i1 + k] if i1 + k < i2 else "[extra]"
                said = spoken_words[j1 + k] if j1 + k < j2 else "[missing]"
                rows.append(("✗" if i1 + k < i2 else "+", expected, said))
        elif tag == "delete":
            rows.extend(("✗", w, "[missing]") for w in target_words[i1:i2])
        elif tag == "insert":
            rows.extend(("+", "[extra]", w) for w in spoken_words[j1:j2])
    return rows, correct


EMPTY_RESULT = {"match": 0.0, "wpm": 0.0, "pause_count": 0, "silence_pct": 0.0,
                "loudness_db": -100.0, "word_rows": [], "correct_words": 0}


def analyze_speech_advanced(target, spoken, audio_duration, audio_data=None, sample_rate=0):
    try:
        target_clean = normalize_text(target)
        spoken_clean = normalize_text(spoken)
        if not target_clean or not spoken_clean:
            return {**EMPTY_RESULT, "feedback": "No usable speech transcript was detected."}
        if audio_duration <= 0:
            return {**EMPTY_RESULT, "feedback": "Invalid audio duration."}

        target_words = target_clean.split()
        spoken_words = spoken_clean.split()
        word_rows, correct_words = compare_words(target_words, spoken_words)
        word_ratio = difflib.SequenceMatcher(None, target_words, spoken_words, autojunk=False).ratio()
        char_ratio = difflib.SequenceMatcher(None, target_clean, spoken_clean, autojunk=False).ratio()
        transcript_match = round((0.75 * word_ratio + 0.25 * char_ratio) * 100, 1)

        if audio_data is not None and sample_rate:
            silence_pct, pauses, active_sec = estimate_silence(audio_data, sample_rate)
            loudness = rms_db(audio_data)
        else:
            silence_pct, pauses, active_sec, loudness = 0.0, 0, 0.0, -100.0

        # Rate over the time actually spent speaking, not the whole clip.
        rate_window = active_sec if active_sec >= 0.3 else audio_duration
        wpm = round(len(spoken_words) / rate_window * 60, 1)

        try:
            target_ipa = ipa.convert(target_clean)
            spoken_ipa = ipa.convert(spoken_clean)
        except Exception:
            target_ipa = spoken_ipa = "Unavailable"

        feedback = [
            f"Target IPA: /{target_ipa}/",
            f"Spoken IPA: /{spoken_ipa}/",
            f"Transcript match: {transcript_match}%",
            f"Words matched: {correct_words}/{len(target_words)}",
            f"Speech rate: {wpm} words/minute (over {rate_window:.1f}s of speech)",
            f"Estimated silence within speech: {silence_pct}%",
            f"Estimated pauses (>= {PAUSE_MIN_SEC:.2f}s): {pauses}",
        ]
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
        feedback.append(
            "\nNote: Whisper transcript matching cannot determine whether an individual "
            "phoneme was articulated correctly. Use this as practice feedback, not a "
            "clinical diagnosis or articulation score."
        )
        return {"match": transcript_match, "wpm": wpm, "pause_count": pauses,
                "silence_pct": silence_pct, "loudness_db": loudness, "word_rows": word_rows,
                "correct_words": correct_words, "feedback": "\n".join(feedback)}
    except Exception as exc:
        return {**EMPTY_RESULT, "feedback": f"Analysis error: {exc}"}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_session(category, target, spoken, accuracy, speech_rate, pause_count, feedback, duration):
    try:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with db_connect() as conn:
            conn.execute(
                "INSERT INTO sessions (timestamp, category, target, spoken, accuracy, speech_rate, "
                "pause_count, feedback, audio_duration) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ts, category, target, spoken, accuracy, speech_rate, pause_count, feedback, duration),
            )
        return True
    except Exception as e:
        st.error(f"Failed to save session: {e}")
        return False


def get_history(limit=20):
    try:
        with db_connect() as conn:
            return conn.execute(
                "SELECT timestamp, category, accuracy, speech_rate FROM sessions "
                "ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
    except Exception as e:
        st.error(f"Failed to fetch history: {e}")
        return []


def get_category_stats():
    try:
        with db_connect() as conn:
            return conn.execute(
                "SELECT category, AVG(accuracy), COUNT(*) FROM sessions "
                "GROUP BY category ORDER BY COUNT(*) DESC"
            ).fetchall()
    except Exception as e:
        st.error(f"Failed to fetch stats: {e}")
        return []


def export_db_bytes():
    """Consistent snapshot of the user's DB (safe even mid-write)."""
    buf_path = SESSIONS_DIR / f".export_{st.session_state.user_id}.db"
    try:
        with db_connect() as src, sqlite3.connect(str(buf_path)) as dst:
            src.backup(dst)
        return buf_path.read_bytes()
    finally:
        buf_path.unlink(missing_ok=True)


def restore_db(raw):
    if len(raw) > MAX_DB_BYTES:
        raise ValueError("Database backup is larger than 25 MB.")
    if not raw.startswith(b"SQLite format 3\x00"):
        raise ValueError("This file is not a SQLite database.")
    candidate = SESSIONS_DIR / f".restore_{st.session_state.user_id}.db"
    try:
        candidate.write_bytes(raw)
        with sqlite3.connect(str(candidate), timeout=5) as conn:
            if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("The backup file is corrupted.")
            cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
        if not REQUIRED_COLUMNS.issubset(cols):
            raise ValueError("This file is not a NeuroSpeech session database.")
        backup = SESSIONS_DIR / f"{st.session_state.user_id}.before_restore.db"
        if DB_PATH.exists():
            DB_PATH.replace(backup)
        candidate.replace(DB_PATH)
    finally:
        candidate.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
def decode_audio_bytes(audio_bytes):
    """Decode any supported audio to mono 16 kHz float32 for Whisper."""
    if not audio_bytes:
        raise ValueError("The recording is empty.")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise ValueError("Audio file is larger than 25 MB.")

    audio_data = None
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
             "-f", "wav", "-ac", "1", "-ar", str(TARGET_SR), "pipe:1"],
            input=audio_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=30, check=False,
        )
        ffmpeg_error = proc.stderr.decode("utf-8", errors="replace").strip()
        if proc.returncode == 0 and proc.stdout:
            audio_data, sample_rate = sf.read(io.BytesIO(proc.stdout), dtype="float32")
    except FileNotFoundError:
        ffmpeg_error = "ffmpeg is not installed"

    if audio_data is None:
        # Fallback: plain 16 kHz WAV can be read without FFmpeg.
        try:
            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        except Exception as fallback_error:
            raise RuntimeError(
                f"Could not decode the recording. FFmpeg: {ffmpeg_error[-300:] or 'unknown error'}; "
                f"WAV fallback: {fallback_error}"
            ) from fallback_error
        if int(sample_rate) != TARGET_SR:
            raise RuntimeError(f"FFmpeg failed ({ffmpeg_error[-200:] or 'unknown'}) "
                               f"and the audio is {sample_rate} Hz, not 16 kHz.")

    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    audio_data = np.ascontiguousarray(audio_data, dtype=np.float32)
    if audio_data.size == 0:
        raise ValueError("The recording contains no audio samples.")
    if not np.isfinite(audio_data).all():
        raise ValueError("The recording contains invalid audio samples.")
    return audio_data, TARGET_SR


def transcribe_audio(model, audio_data):
    result = model.transcribe(
        audio_data, fp16=False, language="en", task="transcribe",
        temperature=0, condition_on_previous_text=False, verbose=None,
    )
    return (result.get("text") or "").strip()


def new_phrase(category, avoid=None):
    options = therapy_modules[category]["exercises"]
    choices = [p for p in options if p != avoid] or options
    return random.choice(choices)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
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
    st.caption("History is private to this browser session. Download it to keep it.")

    try:
        st.download_button(
            "💾 Download Progress", data=export_db_bytes(),
            file_name="my_speech_progress.db", mime="application/x-sqlite3",
            help="Save your therapy history",
        )
    except Exception as e:
        st.warning(f"Cannot prepare download: {e}")

    uploaded_db = st.file_uploader("📂 Restore Progress", type=["db"], help="Upload a previously saved backup")
    if uploaded_db is not None:
        raw = uploaded_db.getvalue()
        digest = hashlib.sha256(raw).hexdigest()
        # The uploader keeps its file across reruns; only restore each file once.
        if st.session_state.get("restored_db_hash") != digest:
            try:
                restore_db(raw)
                st.session_state.restored_db_hash = digest
                st.session_state.pop("last_result", None)
                st.toast("✅ Progress restored. Previous data was backed up.")
                st.rerun()
            except Exception as e:
                st.session_state.restored_db_hash = digest
                st.error(f"Failed to restore database: {e}")

    stats = get_category_stats()
    if stats:
        for cat, avg_acc, count in stats[:5]:
            st.metric(short_name(cat), f"{avg_acc:.1f}%")
            st.caption(f"{count} session{'s' if count != 1 else ''}")
    else:
        st.info("Start practicing to see stats")

col1, col2 = st.columns([2, 1])

with col1:
    if st.session_state.get("category") != category or "current_target" not in st.session_state:
        st.session_state.current_target = new_phrase(category)
        st.session_state.category = category
        st.session_state.pop("last_result", None)

    if st.button("🎲 Get New Phrase", width="stretch"):
        st.session_state.current_target = new_phrase(category, st.session_state.current_target)
        st.session_state.pop("last_result", None)
        st.rerun()

    st.markdown("---")
    st.markdown("### 🎯 Target Phrase:")
    st.markdown(f"# {st.session_state.current_target}")
    try:
        st.caption(f"IPA: /{ipa.convert(st.session_state.current_target)}/")
    except Exception:
        st.caption("IPA conversion unavailable")

    st.markdown("---")
    st.markdown("**Instructions:** Click the microphone, speak the target phrase, then click stop.")

    # Key includes the phrase so the recorder resets for every new phrase.
    rec_key = f"mic_{hashlib.md5(st.session_state.current_target.encode()).hexdigest()[:8]}"
    recorded = st.audio_input("🎙️ Record your voice", key=rec_key)

    st.markdown("#### Or upload an existing recording")
    uploaded_audio = st.file_uploader(
        "🎧 Upload voice recording", type=["wav", "mp3", "m4a", "ogg", "webm"],
        key="neurospeech_audio_upload",
        help="Optional fallback if browser microphone access is unavailable.",
    )

    source = recorded if recorded is not None else uploaded_audio
    audio_bytes = source.getvalue() if source is not None else None

    if audio_bytes:
        # Hash audio + phrase: the same upload can be re-scored against a new phrase,
        # but unrelated reruns (checkbox clicks etc.) don't re-transcribe or re-save.
        audio_hash = hashlib.sha256(audio_bytes + st.session_state.current_target.encode()).hexdigest()
        if st.session_state.get("last_audio_hash") != audio_hash:
            try:
                with st.spinner("🔧 Preparing your recording..."):
                    audio_data, sample_rate = decode_audio_bytes(audio_bytes)
                duration = len(audio_data) / sample_rate
                if duration > 30:
                    st.error("❌ Audio too long. Please record less than 30 seconds.")
                elif duration < 0.5:
                    st.error("❌ Audio too short. Please speak the full phrase.")
                else:
                    with st.spinner("🧠 Loading speech model and analyzing your speech..."):
                        spoken_text = transcribe_audio(load_whisper_model(), audio_data)
                    if not spoken_text:
                        st.warning("⚠️ No speech was detected. Move closer to the microphone and try again.")
                    else:
                        analysis = analyze_speech_advanced(
                            st.session_state.current_target, spoken_text, duration,
                            audio_data=audio_data, sample_rate=sample_rate,
                        )
                        saved = save_session(
                            category, st.session_state.current_target, spoken_text,
                            analysis["match"], analysis["wpm"], analysis["pause_count"],
                            analysis["feedback"], duration,
                        )
                        st.session_state.last_result = {
                            "spoken": spoken_text, "duration": duration, "saved": saved, **analysis,
                        }
                st.session_state.last_audio_hash = audio_hash
            except subprocess.TimeoutExpired:
                st.error("❌ Audio decoding timed out. Please record a shorter clip and try again.")
            except Exception as e:
                st.error(f"❌ Error processing audio: {e}")
                st.caption("If microphone access fails, check browser permission or use the upload option.")

    # Results persist across reruns instead of disappearing.
    res = st.session_state.get("last_result")
    if res:
        st.markdown("### 📝 You said:")
        st.info(f'"{res["spoken"]}"')
        a, b, c, d = st.columns(4)
        a.metric("Transcript Match", f"{res['match']}%")
        b.metric("Speech Rate", f"{res['wpm']} wpm")
        c.metric("Duration", f"{res['duration']:.1f}s")
        d.metric("Estimated Pauses", res["pause_count"])
        st.caption(f"Silence within speech: {res['silence_pct']}% · "
                   f"Average loudness: {res['loudness_db']} dBFS")
        acc = res["match"]
        if acc >= 85:
            st.success("🎉 Excellent phrase match. Keep the same controlled pace.")
        elif acc >= 70:
            st.warning("👍 Good effort. Review the word-by-word feedback.")
        elif acc >= 50:
            st.warning("💪 Keep practicing. Slow down and focus on each target word.")
        else:
            st.error("🔄 Try again and speak the complete target phrase clearly.")
        with st.expander("📋 Detailed Phonetic Feedback"):
            st.code(res["feedback"], language="text")
        if res["saved"]:
            st.success("✅ Session saved to your progress history.")

with col2:
    st.markdown("### 💡 Quick Tips")
    if "Cluttering" in category or "Pacing" in category:
        st.markdown("""
**For Cluttering & Pacing:**
- 🐢 Slow down deliberately
- 🫁 Breathe between phrases
- ✋ Pause after each thought
- 🎯 One word at a time
- 📢 Exaggerate articulation
""")
    else:
        st.markdown("""
**For Articulation:**
- 👄 Watch your mouth in a mirror
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
        st.dataframe(
            [{"Time": ts, "Module": short_name(cat), "Match %": round(acc, 1), "Rate (wpm)": round(rate)}
             for ts, cat, acc, rate in history],
            hide_index=True, width="stretch",
        )
        st.markdown("**Accuracy Trend:**")
        st.line_chart([h[2] for h in history][::-1])
        rates = [h[3] for h in history if h[3] > 0][::-1]
        if rates:
            st.markdown("**Speech Rate Trend:**")
            st.line_chart(rates)
    else:
        st.info("No history yet. Start practicing!")

st.markdown("---")
st.caption("💙 Practice daily for best results. Track progress over weeks.")
st.caption("⚠️ This app uses AI for analysis. Always consult a licensed speech therapist for professional advice.")
