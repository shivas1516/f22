import gradio as gr
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


client = OpenAI(api_key=openai_api_key)

def from_file(audio_file):
    with open(audio_file, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=f
        )
    return transcription.text

file_demo = gr.Interface(
    fn=from_file,
    inputs=gr.Audio(sources=["upload"], type="filepath"),
    outputs="text",
    title="Whisper Demo - File Input"
)

file_demo.launch()
