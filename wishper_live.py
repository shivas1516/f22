import gradio as gr
from faster_whisper import WhisperModel
import numpy as np
import torch
from datetime import datetime, timedelta
import io
from pydub import AudioSegment


# Load Faster-Whisper model
MODEL_NAME = "small.en"  # Options: tiny.en, base.en, small.en, medium.en, large-v2
model = WhisperModel(MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="int8")

# Streaming state
class TranscriptionState:
    def __init__(self):
        self.phrase_time = None
        self.phrase_timeout = 2.0
        self.running_transcript = []
        self.audio_buffer = np.array([], dtype=np.float32)
        self.sample_rate = 16000
        
    def reset(self):
        self.phrase_time = None
        self.running_transcript = []
        self.audio_buffer = np.array([], dtype=np.float32)

# Global state (for demo purposes - use gr.State for production)
state = TranscriptionState()


def transcribe_audio(audio_np, sample_rate):
    """Transcribe audio using faster-whisper"""
    if len(audio_np) == 0:
        return ""
    
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        # Simple resampling (for production, use librosa or torchaudio)
        duration = len(audio_np) / sample_rate
        target_length = int(duration * 16000)
        audio_np = np.interp(
            np.linspace(0, len(audio_np), target_length),
            np.arange(len(audio_np)),
            audio_np
        )
    
    # Transcribe with faster-whisper
    segments, info = model.transcribe(
        audio_np,
        beam_size=1,  # Faster inference
        best_of=1,
        temperature=0,
        vad_filter=True,  # Voice Activity Detection
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    text = " ".join([segment.text for segment in segments]).strip()
    return text


def transcribe_streaming(audio_chunk, transcript_state):
    """Handle streaming microphone input"""
    if audio_chunk is None:
        return transcript_state
    
    sr, data = audio_chunk
    now = datetime.utcnow()
    
    # Convert to float32
    if data.dtype == np.int16:
        audio_np = data.astype(np.float32) / 32768.0
    else:
        audio_np = data.astype(np.float32)
    
    # Handle stereo to mono
    if len(audio_np.shape) > 1:
        audio_np = audio_np.mean(axis=1)
    
    # Check for phrase completion (silence detection)
    phrase_complete = False
    if state.phrase_time and now - state.phrase_time > timedelta(seconds=state.phrase_timeout):
        phrase_complete = True
    state.phrase_time = now
    
    # Append to buffer
    state.audio_buffer = np.append(state.audio_buffer, audio_np)
    
    # Limit buffer size (max 30 seconds)
    max_samples = 30 * sr
    if len(state.audio_buffer) > max_samples:
        state.audio_buffer = state.audio_buffer[-max_samples:]
    
    # Transcribe current buffer
    if len(state.audio_buffer) > sr * 0.5:  # At least 0.5 seconds of audio
        text = transcribe_audio(state.audio_buffer, sr)
        
        if phrase_complete and text:
            state.running_transcript.append(text)
            state.audio_buffer = np.array([], dtype=np.float32)
            return "\n\n".join(state.running_transcript)
        else:
            # Show current partial transcription
            current_transcript = "\n\n".join(state.running_transcript)
            if text:
                current_transcript += "\n\n[Live] " + text
            return current_transcript
    
    return "\n\n".join(state.running_transcript)


def transcribe_file(audio_file):
    """Transcribe uploaded audio file"""
    if audio_file is None:
        return "No audio file provided."
    
    try:
        # Load audio file
        sr, data = audio_file
        
        # Convert to float32
        if data.dtype == np.int16:
            audio_np = data.astype(np.float32) / 32768.0
        else:
            audio_np = data.astype(np.float32)
        
        # Handle stereo to mono
        if len(audio_np.shape) > 1:
            audio_np = audio_np.mean(axis=1)
        
        # Transcribe with progress
        segments, info = model.transcribe(
            audio_np,
            beam_size=5,  # More accurate for file transcription
            best_of=5,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Collect all segments with timestamps
        transcript_parts = []
        for segment in segments:
            start_time = format_timestamp(segment.start)
            end_time = format_timestamp(segment.end)
            transcript_parts.append(f"[{start_time} → {end_time}] {segment.text}")
        
        full_transcript = "\n\n".join(transcript_parts)
        
        if not full_transcript:
            return "No speech detected in the audio file."
        
        return full_transcript
    
    except Exception as e:
        return f"Error transcribing file: {str(e)}"


def format_timestamp(seconds):
    """Format seconds to MM:SS"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def reset_transcript():
    """Reset the streaming transcript"""
    state.reset()
    return ""


# Gradio UI
with gr.Blocks(title="Real-time Whisper Transcription", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎙️ Real-time Speech Transcription
        ### Powered by Faster-Whisper
        Choose between live microphone streaming or file upload transcription.
        """
    )
    
    with gr.Tabs():
        # Tab 1: Streaming Microphone
        with gr.Tab("🎤 Live Microphone"):
            gr.Markdown("**Speak into your microphone for real-time transcription**")
            
            with gr.Row():
                with gr.Column(scale=1):
                    audio_stream = gr.Audio(
                        sources=["microphone"],
                        streaming=True,
                        type="numpy",
                        label="Microphone Input",
                        show_download_button=False
                    )
                    
                    with gr.Row():
                        reset_btn = gr.Button("🔄 Reset Transcript", variant="secondary")
                
                with gr.Column(scale=2):
                    stream_output = gr.Textbox(
                        label="Live Transcript",
                        lines=15,
                        interactive=False,
                        placeholder="Start speaking to see transcription...",
                        show_copy_button=True
                    )
            
            gr.Markdown(
                """
                **Tips:**
                - Speak clearly and at a moderate pace
                - Pauses longer than 2 seconds will create new paragraphs
                - Click Reset to clear the transcript
                """
            )
        
        # Tab 2: File Upload
        with gr.Tab("📁 Upload Audio File"):
            gr.Markdown("**Upload an audio file for transcription with timestamps**")
            
            with gr.Row():
                with gr.Column(scale=1):
                    audio_file = gr.Audio(
                        sources=["upload"],
                        type="numpy",
                        label="Upload Audio File",
                        show_download_button=False
                    )
                    transcribe_btn = gr.Button("▶️ Transcribe File", variant="primary")
                
                with gr.Column(scale=2):
                    file_output = gr.Textbox(
                        label="Transcription with Timestamps",
                        lines=15,
                        interactive=False,
                        placeholder="Upload an audio file and click Transcribe...",
                        show_copy_button=True
                    )
            
            gr.Markdown(
                """
                **Supported formats:** WAV, MP3, MP4, FLAC, OGG, and more
                
                **Note:** File transcription uses higher quality settings and includes timestamps.
                """
            )
    
    # Event handlers
    audio_stream.stream(
        fn=transcribe_streaming,
        inputs=[audio_stream, gr.State()],
        outputs=stream_output
    )
    
    reset_btn.click(
        fn=reset_transcript,
        outputs=stream_output
    )
    
    transcribe_btn.click(
        fn=transcribe_file,
        inputs=audio_file,
        outputs=file_output
    )
    
    # Footer
    gr.Markdown(
        """
        ---
        **Model:** Faster-Whisper (small.en) | **Device:** """ + 
        ("GPU (CUDA)" if torch.cuda.is_available() else "CPU")
    )


if __name__ == "__main__":
    demo.launch(share=False)