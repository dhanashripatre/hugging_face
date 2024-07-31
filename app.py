import gradio as gr
from transformers import pipeline

# Load the summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    # Generate summary
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Create the Gradio interface
iface = gr.Interface(
    fn=summarize_text,
    inputs="text",
    outputs="text",
    title="Text Summarization",
    description="Enter a text to get its summary."
)

# Launch the interface
if __name__ == "__main__":
  iface.launch()