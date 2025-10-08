import gradio as gr
import torch
from TTS.api import TTS
import numpy as np
import re
from pydub import AudioSegment
import io
import tempfile
import os

class XTTSDemo:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize TTS model
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        
    def chunk_text(self, text, max_chars=250):
        # Split on sentence boundaries (., !, ?)
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence exceeds max_chars and current_chunk is not empty
            if len(current_chunk) + len(sentence) > max_chars and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def combine_audio_segments(self, audio_files):
        combined = AudioSegment.empty()
        
        for audio_file in audio_files:
            audio = AudioSegment.from_wav(audio_file)
            combined += audio
            # Add small pause between chunks (100ms)
            combined += AudioSegment.silent(duration=100)
        
        # Export combined audio
        output_path = tempfile.mktemp(suffix=".wav")
        combined.export(output_path, format="wav")
        
        return output_path
    
    def generate_speech(self, text, speaker_wav, language, progress=gr.Progress()):
        try:
            if not text or not text.strip():
                return None, "⚠️ Please enter some text to convert to speech."
            
            if not speaker_wav:
                return None, "⚠️ Please upload a speaker reference audio file."
            
            progress(0, desc="Chunking text...")
            
            # Chunk the text
            chunks = self.chunk_text(text, max_chars=250)
            total_chunks = len(chunks)
            
            info_text = f"Processing {total_chunks} chunk(s)...\n"
            
            # Generate audio for each chunk
            audio_files = []
            
            for i, chunk in enumerate(chunks):
                progress((i + 1) / total_chunks, desc=f"Generating chunk {i+1}/{total_chunks}")
                
                # Generate audio for this chunk
                temp_file = tempfile.mktemp(suffix=".wav")
                
                self.tts.tts_to_file(
                    text=chunk,
                    speaker_wav=speaker_wav,
                    language=language,
                    file_path=temp_file
                )
                
                audio_files.append(temp_file)
                info_text += f"✓ Chunk {i+1}: {len(chunk)} chars\n"
            
            progress(0.9, desc="Combining audio chunks...")
            
            # Combine all audio files
            if len(audio_files) == 1:
                final_audio = audio_files[0]
            else:
                final_audio = self.combine_audio_segments(audio_files)
                
                # Clean up temporary files
                for temp_file in audio_files:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            
            progress(1.0, desc="Complete!")
            
            info_text += f"\n✅ Successfully generated speech!"
            
            return final_audio, info_text
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"

# Initialize the demo
demo_instance = XTTSDemo()

# Create Gradio interface
with gr.Blocks(title="XTTS-v2 Text-to-Speech Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎙️ XTTS-v2 Text-to-Speech Demo
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input section
            text_input = gr.Textbox(
                label="Text to Convert",
                placeholder="Enter your text here... (Any length supported, text will be automatically chunked)",
                lines=10,
                value="Hello! This is a demonstration of the XTTS-v2 text-to-speech model. This model can clone voices and generate natural-sounding speech in multiple languages. Try entering a long paragraph to see how it handles text chunking automatically!"
            )
            
            speaker_audio = gr.Audio(
                label="Speaker Reference Audio (6-30 seconds)",
                type="filepath",
                sources=["upload", "microphone"]
            )
            
            language_dropdown = gr.Dropdown(
                label="Language",
                choices=[
                    "en", "es", "fr", "de", "it", "pt", "pl", "tr", 
                    "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"
                ],
                value="en",
                info="Select the target language for speech generation"
            )
            
            generate_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            # Output section
            audio_output = gr.Audio(
                label="Generated Speech",
                type="filepath"
            )
            
            info_output = gr.Textbox(
                label="Generation Info",
                lines=10,
                interactive=False
            )
    
    # Example section
    gr.Markdown("### 📚 Example Texts")
    
    examples = [
        ["The quick brown fox jumps over the lazy dog. This is a short sentence to test the model."],
        ["Artificial intelligence is transforming the way we live and work. From voice assistants to autonomous vehicles, AI applications are becoming increasingly sophisticated. Machine learning algorithms can now perform tasks that once seemed impossible, opening up new possibilities for innovation across all industries."],
        ["In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to eat: it was a hobbit-hole, and that means comfort. It had a perfectly round door like a porthole, painted green, with a shiny yellow brass knob in the exact middle."]
    ]
    
    gr.Examples(
        examples=examples,
        inputs=text_input,
        label="Click to load example text"
    )
    
    # Event handlers
    generate_btn.click(
        fn=demo_instance.generate_speech,
        inputs=[text_input, speaker_audio, language_dropdown],
        outputs=[audio_output, info_output]
    )
    
    gr.Markdown("""
    ---
    ### ℹ️ Notes:
    - **Reference Audio:** Upload a clean audio sample (6-30 seconds) of the voice you want to clone
    - **Text Length:** Any length is supported - long texts are automatically chunked
    - **Languages:** Supports 17+ languages including English, Spanish, French, German, Chinese, Japanese, and more
    - **Processing Time:** Longer texts will take more time to process (approximately 1-3 seconds per chunk)
    
    ### 🔧 Technical Details:
    - Model: `coqui/XTTS-v2` from Hugging Face
    - Chunking: Sentences are grouped into ~250 character chunks
    - Audio Combining: Chunks are merged with 100ms pauses for natural flow
    """)

# Launch the demo
if __name__ == "__main__":
    demo.launch(share=True)