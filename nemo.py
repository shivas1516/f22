# Install dependencies (run once)
# !pip install nemo_toolkit[all] sounddevice numpy

import sounddevice as sd
import numpy as np
import torch
import nemo.collections.asr as nemo_asr

# -----------------------------
# 1. Load the pretrained ASR model
# -----------------------------
print("Loading Parakeet TDT 0.6B V2 model...")
model = nemo_asr.models.EncDecCTCModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Model loaded on {device}")

# -----------------------------
# 2. Audio capture settings
# -----------------------------
SAMPLE_RATE = 16000      # Model expects 16kHz
CHUNK_SIZE = 3           # Seconds per audio chunk

# -----------------------------
# 3. Callback function for streaming
# -----------------------------
def audio_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}", flush=True)
    # Convert audio chunk to float32
    audio_chunk = indata[:, 0].astype(np.float32)
    # Normalize if necessary
    audio_chunk = audio_chunk / np.max(np.abs(audio_chunk))
    # Transcribe the chunk
    transcription = model.transcribe([audio_chunk])
    if transcription:
        print("Transcript:", transcription[0], flush=True)

# -----------------------------
# 4. Start real-time streaming
# -----------------------------
print("Starting real-time transcription. Speak into your microphone...")
with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=int(SAMPLE_RATE*CHUNK_SIZE)):
    print("Press Ctrl+C to stop.")
    try:
        while True:
            sd.sleep(1000)  # Keep stream alive
    except KeyboardInterrupt:
        print("Stopped transcription.")
