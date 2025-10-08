import os
from groq import Groq
import gradio as gr
from dotenv import load_dotenv

load_dotenv()


client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def chat(user_input, history):
    stream = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Your name is GroqBot.",
            },
            {
                "role": "user",
                "content": user_input,
            }
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.5, # For randomness and creativity
        max_completion_tokens=1024, # Limit the response length
        top_p=1,
        stop=None, 
        stream=True, # For streaming responses
    )

    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            full_response += chunk.choices[0].delta.content
            yield full_response  # Yield the accumulated response so far

# Set up the Gradio interface
demo = gr.ChatInterface(
    title="Groq LLM Chat Interface",
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