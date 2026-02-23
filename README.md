# Sign Language Detection App

Two-section app for sign language recognition:
- Upload a video and get a caption of detected signs.
- Real-time webcam detection for deaf or hard of hearing users.

## Project Layout
- `app/app.py` – Streamlit UI with two sections
- `app/utils.py` – Mediapipe keypoint extraction and dataset loader
- `app/train_model.py` – LSTM training pipeline
- `app/inference.py` – Video inference and caption generation
- `MP_Data/` – Dataset directory (create one folder per sign, e.g., `hello/`, `thanks/`); inside each sign folder, create numbered sequence folders (`1/`, `2/`, ...) with frame `.npy` files (30 per sequence)
- `models/` – Saved model and `actions.npy` will be created after training

## Setup
1. Use the virtual environment in `.venv` or your preferred Python environment.
2. Install dependencies:
   - Windows PowerShell: `.venv\Scripts\python -m pip install -r requirements.txt`

## Training
1. Prepare your dataset under `MP_Data` (see layout above).
2. Run training:
   - `.venv\Scripts\python app/train_model.py`
3. Output: `models/slr_lstm.keras` and `models/actions.npy`.

## Run the App
- `.venv\Scripts\python -m streamlit run app/app.py`
- Tab 1: Upload a video and get caption text.
- Tab 2: Start webcam for real-time detection.

## Notes
- If `models/slr_lstm.keras` is missing, the app displays a warning and disables detection.
- Actions are derived from `models/actions.npy`. If missing, the app falls back to listing sign folders under `MP_Data`.
- Default sequence length is 30 frames; ensure each sequence folder contains at least 30 `.npy` frames.

## Future Improvements
- Add data collection UI to record sequences directly.
- Expand from isolated sign classification to continuous sign language translation.
- Add text-to-speech for captions.