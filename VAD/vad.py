from faster_whisper import WhisperModel

model_size = "medium"
model = WhisperModel(model_size, device="cuda", compute_type="float16")

def get_vad(audio_file):
    segments, info = model.transcribe(audio_file, beam_size=5, word_timestamps=True)
    vad_segments = []
    for segment in segments:
        vad_segments.append((segment.start, segment.end))
    return vad_segments


from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

def diarize(audio_file):
    diarization = pipeline(audio_file)
    # Output: list of segments with speaker labels
    results = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        results.append((turn.start, turn.end, speaker))
    return results

import gradio as gr

def process_audio(audio_file):
    vad_segments = get_vad(audio_file.name)
    diarization_segments = diarize(audio_file.name)
    
    output_text = "VAD Segments:\n" + str(vad_segments) + "\n\n"
    output_text += "Speaker Diarization:\n" + str(diarization_segments)
    
    return output_text

ui = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(source="upload", type="file"),
    outputs="text",
    title="VAD + Speaker Diarization Demo"
)

if __name__ == "__main__":
    ui.launch()
