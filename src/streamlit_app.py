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
import time
from pathlib import Path
import tempfile

# ================= CONFIG =================
st.set_page_config(
    page_title="NeuroSpeech Therapy Pro",
    page_icon="🩺",
    layout="wide"
)

# ================= DATABASE =================
# Use proper persistent storage for Hugging Face Spaces
PERSISTENT_DIR = os.getenv("PERSISTENT_STORAGE_PATH", "./data")
Path(PERSISTENT_DIR).mkdir(parents=True, exist_ok=True)
DB_PATH = os.path.join(PERSISTENT_DIR, "patient_history.db")

def init_db():
    """Initialize database with proper error handling"""
    try:
        conn = sqlite3.connect(DB_PATH)
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
        # Add index for faster queries
        c.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON sessions(timestamp DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_category ON sessions(category)")
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Database initialization failed: {e}")
        return False

init_db()

# ================= LOAD WHISPER =================
@st.cache_resource
def load_whisper_model():
    """Load Whisper model with error handling"""
    try:
        # Use 'base' for better accuracy while maintaining reasonable speed
        return whisper.load_model("base", device="cpu")
    except Exception as e:
        st.error(f"Failed to load Whisper model: {e}")
        return None

model = load_whisper_model()

# ================= COMPREHENSIVE THERAPY PHRASES =================
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

# ================= SPEECH ANALYSIS =================
def analyze_speech_advanced(target, spoken, audio_duration):
    """Advanced analysis with robust error handling"""
    try:
        target_clean = target.lower().replace("...", "").strip()
        spoken_clean = spoken.lower().strip()
        
        # Validate inputs
        if not target_clean or not spoken_clean:
            return 0.0, 0.0, 0, "Invalid input - empty text"
        
        if audio_duration <= 0:
            return 0.0, 0.0, 0, "Invalid audio duration"
        
        # Accuracy
        accuracy = round(
            difflib.SequenceMatcher(None, target_clean, spoken_clean).ratio() * 100,
            1
        )
        
        # Speech rate (words per minute)
        word_count = len(spoken.split())
        speech_rate = round((word_count / audio_duration) * 60, 1)
        
        # Estimate pause count
        pause_count = spoken.count('.') + spoken.count(',') + spoken.count('...')
        
        # IPA comparison with error handling
        try:
            target_ipa = ipa.convert(target_clean)
            spoken_ipa = ipa.convert(spoken_clean)
        except Exception:
            target_ipa = "N/A"
            spoken_ipa = "N/A"
        
        # Detailed feedback
        feedback_parts = []
        feedback_parts.append(f"Target IPA: /{target_ipa}/")
        feedback_parts.append(f"Spoken IPA: /{spoken_ipa}/")
        feedback_parts.append(f"\nSpeech Rate: {speech_rate} words/minute")
        
        # Rate guidance
        if speech_rate > 200:
            feedback_parts.append("⚠️ Speaking too fast - try slowing down")
        elif speech_rate < 100:
            feedback_parts.append("✓ Good controlled pace")
        elif speech_rate < 150:
            feedback_parts.append("✓ Excellent moderate pace")
        
        # Word-by-word comparison
        target_words = target_clean.split()
        spoken_words = spoken_clean.split()
        
        if len(target_words) > 0 and len(spoken_words) > 0:
            feedback_parts.append("\n--- Word Analysis ---")
            max_len = max(len(target_words), len(spoken_words))
            for i in range(min(max_len, 20)):  # Limit to 20 words to prevent overflow
                t_word = target_words[i] if i < len(target_words) else "[missing]"
                s_word = spoken_words[i] if i < len(spoken_words) else "[extra]"
                
                if t_word.lower() == s_word.lower():
                    feedback_parts.append(f"✓ {t_word}")
                else:
                    feedback_parts.append(f"✗ Expected: {t_word} | Said: {s_word}")
        
        return accuracy, speech_rate, pause_count, "\n".join(feedback_parts)
    
    except Exception as e:
        return 0.0, 0.0, 0, f"Analysis error: {str(e)}"

def save_session(category, target, spoken, accuracy, speech_rate, pause_count, feedback, duration):
    """Save session with error handling"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute(
            "INSERT INTO sessions (timestamp, category, target, spoken, accuracy, speech_rate, pause_count, feedback, audio_duration) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, category, target, spoken, accuracy, speech_rate, pause_count, feedback, duration)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Failed to save session: {e}")
        return False

def get_history(limit=20):
    """Get session history with error handling"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "SELECT timestamp, category, accuracy, speech_rate FROM sessions ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
        rows = c.fetchall()
        conn.close()
        return rows
    except Exception as e:
        st.error(f"Failed to fetch history: {e}")
        return []

def get_category_stats():
    """Get category statistics with error handling"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "SELECT category, AVG(accuracy) as avg_acc, COUNT(*) as count FROM sessions GROUP BY category ORDER BY count DESC"
        )
        rows = c.fetchall()
        conn.close()
        return rows
    except Exception as e:
        st.error(f"Failed to fetch stats: {e}")
        return []

# ================= UI =================
st.title("🩺 NeuroSpeech Therapy Pro")
st.markdown("### Comprehensive Speech Therapy for Cluttering & Articulation")

# Check if model loaded successfully
if model is None:
    st.error("❌ Failed to load speech recognition model. Please refresh the page.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Therapy Settings")
    
    category = st.selectbox(
        "Select Module",
        list(therapy_modules.keys()),
        help="Choose your target area"
    )
    
    st.markdown("---")
    st.markdown(f"**{therapy_modules[category]['description']}**")
    st.info(therapy_modules[category]['tips'])
    
    st.markdown("---")
    st.subheader("📊 Your Progress")
    
    # Database backup/restore
    if os.path.exists(DB_PATH):
        try:
            with open(DB_PATH, "rb") as f:
                st.download_button(
                    "💾 Download Progress",
                    f,
                    file_name="my_speech_progress.db",
                    help="Save your therapy history"
                )
        except Exception as e:
            st.warning(f"Cannot download database: {e}")
    
    uploaded_db = st.file_uploader(
        "📂 Restore Progress",
        type=["db"],
        help="Upload previously saved database"
    )
    if uploaded_db:
        try:
            with open(DB_PATH, "wb") as f:
                f.write(uploaded_db.read())
            st.success("✅ Progress restored!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to restore database: {e}")
    
    stats = get_category_stats()
    if stats:
        for cat, avg_acc, count in stats[:5]:  # Show top 5
            short_name = cat.split(" ")[-1][:15] if " " in cat else cat[:15]
            st.metric(short_name, f"{avg_acc:.1f}%", f"{count} sessions")
    else:
        st.info("Start practicing to see stats")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    # Initialize session state properly
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
    st.markdown(f"### 🎯 Target Phrase:")
    st.markdown(f"# {st.session_state.current_target}")
    
    try:
        ipa_text = ipa.convert(st.session_state.current_target)
        st.caption(f"IPA: /{ipa_text}/")
    except Exception:
        st.caption("IPA conversion unavailable")
    
    st.markdown("---")

    # Audio input with instructions
    st.markdown("**Instructions:** Click the microphone button below and speak the target phrase clearly.")
    audio = st.audio_input("🎙️ Record your voice")

    if audio is not None:
        start_time = time.time()
        
        # Use tempfile for better cleanup
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(audio.read())
        
        try:
            # Read audio file
            audio_data, sample_rate = sf.read(tmp_path, dtype="float32")

            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1).astype(np.float32)

            duration = len(audio_data) / sample_rate
            
            # Validation
            if duration > 30:
                st.error("❌ Audio too long. Please record less than 30 seconds.")
            elif duration < 0.5:
                st.error("❌ Audio too short. Please speak the full phrase.")
            else:
                with st.spinner("🧠 Analyzing your speech..."):
                    result = model.transcribe(audio_data, fp16=False, language="en")
                    spoken_text = result["text"].strip()

                if not spoken_text:
                    st.error("❌ No speech detected. Please try again and speak clearly.")
                else:
                    accuracy, speech_rate, pause_count, feedback = analyze_speech_advanced(
                        st.session_state.current_target,
                        spoken_text,
                        duration
                    )

                    st.markdown("### 📝 You said:")
                    st.info(f'"{spoken_text}"')

                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Accuracy", f"{accuracy}%")
                    with col_b:
                        st.metric("Speech Rate", f"{speech_rate} wpm")
                    with col_c:
                        st.metric("Duration", f"{duration:.1f}s")

                    if accuracy >= 85:
                        st.success("🎉 Excellent! Your articulation is clear and accurate.")
                    elif accuracy >= 70:
                        st.warning("👍 Good effort! Review the feedback below for improvement.")
                    elif accuracy >= 50:
                        st.warning("💪 Keep practicing! Focus on the tips provided.")
                    else:
                        st.error("🔄 Try again - slow down and focus on each sound.")

                    with st.expander("📋 Detailed Phonetic Feedback"):
                        st.text(feedback)

                    save_session(
                        category,
                        st.session_state.current_target,
                        spoken_text,
                        accuracy,
                        speech_rate,
                        pause_count,
                        feedback,
                        duration
                    )
        
        except Exception as e:
            st.error(f"❌ Error processing audio: {str(e)}")
        
        finally:
            # Cleanup temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

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

# History section
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
                st.text(f"Accuracy: {acc:.1f}%")
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