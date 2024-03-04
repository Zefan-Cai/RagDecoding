from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the tokenizer and model
model_checkpoint_path = "/home/models/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint_path)

# Check if CUDA is available and move the model to CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function for manual inference on CUDA
def manual_infer_with_llama(prompt, max_length=50):
    # Tokenize the prompt and move tensors to the device
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    # Generate tokens one by one
    for _ in range(max_length):
        # Get logits of the next token
        output = model(input_ids).logits
        next_token_logits = output[:, -1, :]

        # Choose the most likely next token (you can also sample)
        next_token = torch.argmax(next_token_logits, dim=-1)

        # Append the new token to the existing sequence
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)

        # Check if the last token is an end-of-sequence token
        if next_token in tokenizer.all_special_ids:
            break

    # Decode the generated tokens and return the text
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

# Example usage
prompt = "Write a short story about a space adventure."
print(manual_infer_with_llama(prompt))
