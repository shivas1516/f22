import gradio as gr
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


client = OpenAI(api_key=openai_api_key)

def from_mic(audio_file):
    if audio_file is None:
        return "No audio received."
    
    with open(audio_file, "rb") as f:

        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    return transcription.text

mic_demo = gr.Interface(
    fn=from_mic,
    inputs=gr.Audio(sources=["microphone"], type="filepath"),
    outputs="text",
    title="Whisper Demo"
)

mic_demo.launch()
