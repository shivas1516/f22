import gradio as gr
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import time
import GPUtil

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

print("Loading model...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    dtype=dtype,
    low_cpu_mem_usage=True,  # Reduce RAM usage
    device_map="auto" if torch.cuda.is_available() else None,
    use_safetensors=True,
)
processor = AutoProcessor.from_pretrained(model_id)

# pipeline
asr_pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor)


def transcribe_audio(audio_file):
    start_time = time.time()
    result = asr_pipe(audio_file)
    latency = time.time() - start_time
    if torch.cuda.is_available():
        gpus = GPUtil.getGPUs()
        gpu_info = "\n".join([f"GPU {gpu.id}: {gpu.memoryUsed}/{gpu.memoryTotal} MB used, {gpu.load*100:.1f}% load" for gpu in gpus])
    else:
        gpu_info = "CPU mode, no GPU available."

    print(f"Transcription latency: {latency:.2f} seconds")
    print(gpu_info)

    return result["text"]


file_demo = gr.Interface(
    fn=transcribe_audio, inputs=gr.Audio(sources=["upload"], type="filepath"), outputs="text", title="Optimized Whisper Transformers Demo - File Input"
)

file_demo.launch()
