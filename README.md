# 🩺 NeuroSpeech Therapy Pro

AI-powered speech therapy application for cluttering and articulation disorders.

![Status](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌟 Features

- **12 Comprehensive Therapy Modules**
  - Cluttering Control
  - Articulation (Bilabials, Alveolars, Velars, Fricatives)
  - Liquid Sounds (R, L)
  - Blends & Clusters
  - Pacing & Rhythm
  - Sentence Complexity
  - Prosody & Intonation
  - Repetition Drills
  - Conversational Practice

- **Advanced Speech Analysis**
  - Real-time transcription using OpenAI Whisper
  - Accuracy scoring (0-100%)
  - Speech rate calculation (words per minute)
  - IPA (International Phonetic Alphabet) conversion
  - Word-by-word comparison
  - Phonetic feedback

- **Progress Tracking**
  - Session history with timestamps
  - Category-wise statistics
  - Visual progress charts
  - Downloadable database backup
  - Restore previous sessions

## 🚀 Demo

**Live Demo:** [Try NeuroSpeech](https://huggingface.co/spaces/parthbijpuriya/neurospeech)

## 📸 Screenshots

*[Add screenshots here after deployment]*

## 🛠️ Technologies

- **Frontend:** Streamlit
- **AI Model:** OpenAI Whisper (Base model)
- **Speech Processing:** librosa, soundfile
- **Phonetics:** eng-to-ipa
- **Database:** SQLite
- **Language:** Python 3.11+

## 📋 Prerequisites

- Python 3.11 or higher
- ffmpeg (for audio processing)
- At least 2GB RAM (for Whisper model)

## 🔧 Installation

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/neurospeech.git
   cd neurospeech
   ```

2. **Install system dependencies** (Ubuntu/Debian)
   ```bash
   sudo apt-get update
   sudo apt-get install ffmpeg libsndfile1
   ```

   **macOS:**
   ```bash
   brew install ffmpeg libsndfile
   ```

   **Windows:**
   - Download ffmpeg from https://ffmpeg.org/download.html
   - Add to PATH

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   - Navigate to `http://localhost:8501`

## 🌐 Deployment

### Deploy to Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `app.py` as the main file
5. Click Deploy!

### Deploy to Render

1. Create account at [render.com](https://render.com)
2. Connect GitHub repository
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `streamlit run app.py --server.port $PORT`
5. Deploy!

## 📁 Project Structure

```
neurospeech/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── packages.txt             # System dependencies (for Streamlit Cloud)
├── README.md                # This file
├── .gitignore              # Git ignore rules
└── data/                   # Database storage (auto-created)
    └── patient_history.db  # SQLite database
```

## 💡 Usage

1. **Select a therapy module** from the sidebar
2. **Read the target phrase** displayed on screen
3. **Click the microphone** and record yourself saying the phrase
4. **View instant feedback** including:
   - Accuracy percentage
   - Speech rate
   - Phonetic comparison
   - Word-by-word analysis
5. **Track your progress** over time with charts and statistics

## 🎯 Target Audience

- Individuals with cluttering disorders
- People with articulation difficulties
- Speech therapy patients practicing at home
- Speech-language pathology students
- Anyone wanting to improve speech clarity

## ⚠️ Disclaimer

This application is designed as a **supplementary practice tool** and is not a replacement for professional speech therapy. Always consult with a licensed speech-language pathologist for proper diagnosis and treatment.

## 🔒 Privacy

- All speech analysis happens locally or on the server
- Audio recordings are temporary and deleted after processing
- Session history is stored locally in SQLite database
- No audio or personal data is sent to external services (except Whisper processing)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Parth Bijpuriya**
- GitHub: [@parthbijpuriya](https://github.com/parthbijpuriya)
- Hugging Face: [@parthbijpuriya](https://huggingface.co/parthbijpuriya)

## 🙏 Acknowledgments

- OpenAI Whisper team for the excellent speech recognition model
- Streamlit team for the amazing framework
- Speech-language pathology community for inspiration

## 📞 Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Contact via [your email or social media]

## 🗺️ Roadmap

- [ ] Add user authentication
- [ ] Export progress reports as PDF
- [ ] Multi-language support
- [ ] Mobile app version
- [ ] Integration with therapy platforms
- [ ] Advanced voice quality metrics
- [ ] Custom phrase uploads
- [ ] Therapist dashboard

## ⭐ Star History

If you find this project helpful, please consider giving it a star!

---

Made with ❤️ for the speech therapy community
