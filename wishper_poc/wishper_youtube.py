from openai import OpenAI
import yt_dlp
import gradio as gr
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


client = OpenAI(api_key=openai_api_key)

def download_youtube_audio(url, output_path='./downloads'):

    try:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(output_path, '%(title)s.mp3'),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "postprocessors": [],
            "keepvideo": False,
        }
        
        print(f"Downloading video from: {url}\n")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title", "YouTube Video")

            downloaded_files = os.listdir(output_path)
            if not downloaded_files:
                print("No files were downloaded")
                return None

            downloaded_file = os.path.join(output_path, downloaded_files[0])

            return downloaded_file

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def from_youtube(url):
    audio_file = download_youtube_audio(url)
    print(f"Request forwarded to Whisper API with file")
    with open(audio_file, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    return transcription.text

yt_demo = gr.Interface(
    fn=from_youtube,
    inputs=gr.Textbox(label="YouTube URL"),
    outputs="text",
    title="Whisper Large V3 - YouTube URL"
)

yt_demo.launch()
