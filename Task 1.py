from transformers import pipeline

# Load the summarization pipeline
summarizer = pipeline("summarization")

# Input text (you can replace this with any lengthy article)
input_text = """
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. 
These machines are capable of performing tasks that typically require human intelligence, such as understanding natural language, recognizing patterns, solving problems, and making decisions. 
AI is being applied across various industries including healthcare, finance, transportation, and more. As technology continues to evolve, the capabilities of AI are expected to expand, offering new possibilities and transforming the way we live and work.
"""

# Generate summary
summary = summarizer(input_text, max_length=50, min_length=25, do_sample=False)

# Print results
print("Original Text:\n", input_text)
print("\nSummarized Text:\n", summary[0]['summary_text'])
