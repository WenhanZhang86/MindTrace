# Context Assistant

This app captures:
- audio from your input device and transcribes it to text
- text visible on your screen with OCR
- then lets you summarize or ask questions based on captured context

Each run is stored as a session JSON file, similar to a conversation history.

## Important

- Each user needs their own OpenAI API key for AI summarize/Q&A features.
- Without an API key, capture and session saving still work, but AI features will not run.

## What this app includes

- `Start to Work` button
- `End to Work` button
- automatic session save to `sessions/`
- max duration guard (auto-stop with warning and save)
- session summary button (LLM)
- ask-a-question box over captured content (LLM)

## Files

- `app.py` - main desktop app (Tkinter UI)
- `requirements.txt` - Python dependencies
- `config.example.json` - runtime settings template
- `sessions/` - generated during runtime

## Prerequisites

1. Python 3.10+
2. Tesseract OCR installed:
   - `brew install tesseract`
3. FFmpeg installed:
   - `brew install ffmpeg`
4. Optional for internal system audio capture:
   - install a virtual audio device for your OS and set it as input device
   - otherwise mic input works

## Setup

1. Open Terminal:
   - `cd Desktop/Mindtrace
2. Create and activate venv:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
3. Install dependencies:
   - `pip install -r requirements.txt`
4. Create runtime config:
   - `cp config.example.json config.json`
5. Set API key for summarize/Q&A:
   - `export OPENAI_API_KEY="your_api_key_here"`
6. Run:
   - `python app.py`

## GitHub publishing safety

- Never commit real API keys.
- Keep your real runtime config in `config.json` (ignored by git).
- Only commit `config.example.json`.
- `.gitignore` is set to exclude `.venv/`, `sessions/`, `.DS_Store`, and `config.json`.
- If a key was ever exposed, revoke it in OpenAI and create a new one.

## Permissions you must grant

- Screen capture permission for Python/Terminal (for OCR)
- Microphone permission for Python/Terminal (for audio)

Without these permissions, capture will fail.

## Notes on long runtime

- Controlled by `max_duration_seconds` in `config.json`
- default is `7200` seconds (2 hours)
- when limit is reached, app:
  1) shows warning
  2) stops capture
  3) saves the session

## Session format

Each session is saved as `sessions/<session_id>.json` with:
- timestamp
- source (`audio` or `screen`)
- captured text

## Current limitations

- "System audio" capture on macOS often needs virtual audio device setup (BlackHole/Loopback).
- OCR quality depends on text size and contrast.
- LLM summary/Q&A needs internet and valid OpenAI API key.
- This is a strong MVP/starter, not a signed production app bundle yet.

## Next upgrades (recommended)

- Use native macOS app packaging (py2app or Swift frontend)
- Add local vector search for better Q&A retrieval
- Add per-app capture filters and privacy controls
- Add pause/resume button
- Export summaries as Markdown/PDF
