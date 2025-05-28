from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained model and tokenizer
model_name = "gpt2"  # You can also use 'gpt2-medium', 'gpt2-large' etc.
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# Function to generate text
def generate_text(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, 
                                no_repeat_ngram_size=2, 
                                top_k=50,
                                top_p=0.95,
                                temperature=0.7,
                                do_sample=True)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Example Usage
user_prompt = "Artificial Intelligence will transform the future because"
generated = generate_text(user_prompt)
print("Generated Text:\n", generated)
