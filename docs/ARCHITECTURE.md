# NeuroSpeech Architecture

This document describes the architecture implied by the checked application entry point, requirements, and README. Confirm any implementation changes in source before relying on this guide.

## Runtime flow

```text
Browser microphone or uploaded audio
        |
        v
Streamlit UI (src/streamlit_app.py)
        |
        +--> Validate file type and size
        |
        +--> Audio decoding / preprocessing
        |
        +--> Whisper transcription
        |
        +--> Transcript comparison and IPA conversion
        |
        +--> Speech-rate, pause, and waveform analysis
        |
        +--> Session-history persistence (SQLite)
        |
        v
Feedback, history, and progress UI
```

## Main components

- **UI/orchestration:** Streamlit receives user input and coordinates analysis.
- **Transcription:** Whisper produces a transcript from audio.
- **Analysis:** transcript matching, IPA comparison, speech-rate and pause calculations provide practice-oriented feedback.
- **Persistence:** SQLite stores session data according to the configured storage path.
- **Deployment:** Docker runs the Streamlit app on port 8501 for the documented Hugging Face Spaces setup.

## Configuration

The README documents `WHISPER_MODEL_SIZE`, `PERSISTENT_STORAGE_PATH`, and `WHISPER_CACHE_DIR`. Check the source for defaults and validation rules. For persistent deployment, ensure the configured data path is backed by persistent storage.

## Data and privacy boundaries

Audio and speech-related history may be sensitive. Minimize retention, explain storage location, and provide a deletion workflow. Ensure one browser session cannot read or overwrite another session's data. Do not expose local SQLite files or audio uploads through static hosting.

## Failure modes to test

- Unsupported or corrupted audio
- Files exceeding the configured size limit
- FFmpeg or audio-decoder failures
- Whisper model download/load failures
- Empty or inaccurate transcripts
- Database write failures and deployment restarts
- Session isolation and persistence behavior

## Quality signals

Test audio preprocessing and scoring helpers independently. Evaluate transcription separately from pronunciation feedback; transcript similarity is not a clinical measure of articulation. Document the evaluation corpus, languages, recording conditions, and known error patterns.
