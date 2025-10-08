from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import gradio as gr
import time

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def chat(user_input, history):

    start_time = time.time()

    response = client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=[user_input],
        config=types.GenerateContentConfig(
            
            max_output_tokens=1024,

            temperature=0.5,
            top_p=1.0,
            # top_k=40,

            system_instruction="You are a helpful assistant. Your name is GeminiBot.",
            thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking

        )
    )

    full_response = ""
    for chunk in response:
        if chunk.text is not None:
            full_response += chunk.text
            yield full_response  # Yield the accumulated response so far

    endtime = time.time()
    print(f"Response time: {endtime - start_time} seconds")

demo = gr.ChatInterface(
    title="Gemini LLM Chat Interface",
    fn=chat,
    type="messages",
    flagging_mode="manual",
    flagging_options=["Like", "Dislike"], # Feedback options for the responses
    save_history=True,
    autoscroll=True,
    fill_height=True,
    fill_width=True,
    theme="soft"

)

# Launch the Gradio interface
if __name__ == "__main__":
    demo.launch()