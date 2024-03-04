from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import torch
import json

# Load the tokenizer and model



model_checkpoint_path = "/home/caizf/models/Llama-2-7b-chat-hf/"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint_path, torch_dtype=torch.float16)

# Check if CUDA is available and move the model to CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


model.config.output_attention = True

# Function for manual inference on CUDA
def manual_infer_with_llama_with_attention(prompt, max_length=50):
    # Tokenize the prompt and move tensors to the device
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
     
    # Generate tokens one by one
    for _ in range(max_length):
        # Get logits of the next token
        raw_outputs = model(input_ids, output_attentions=True)
        output = raw_outputs.logits
        next_token_logits = output[:, -1, :]
        # print(next_token_logits.shape)
        
        # print(f"debug {raw_outputs.keys()}")
        
        attentions = raw_outputs.attentions
        last_layer_attentions = attentions[-1]
        
        # print(f"debug attentions {len(attentions)}")
        # print(f"debug {last_layer_attentions.shape}")
        # print(f"debug last_layer_attentions {last_layer_attentions[0, -1, :, :]}")
        
        # break

        # Choose the most likely next token (you can also sample)
        next_token = torch.argmax(next_token_logits, dim=-1)

        # Append the new token to the existing sequence
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
        
        # print(f"debug {input_ids.shape}")

        # print(input_ids.shape)
        # print(next_token.shape)

        # Check if the last token is an end-of-sequence token
        if next_token in tokenizer.all_special_ids:
            break

    # Decode the generated tokens and return the text
    return tokenizer.decode(input_ids[0], skip_special_tokens=True), input_ids[0], last_layer_attentions


input = """
You are given some documents, and you need to answer a question based on these documents. 
document 18: title: List of Back to the Future characters text: The character was played by Claudia Wells in Back to the Future. However, Wells was not available to film the sequels for personal reasons, and the role was recast to Elisabeth Shue although Wells reprised her role as Jennifer in Back to the Future: The Game as a punk rock version of her character. Consequently, the opening scene of Back to the Future Part II was re-shot with Shue taking Wells' place, rather than using the ending of Back to the Future. In the spin - off Back to the Future: the Animated Series, Jennifer was voiced by Cathy Cavadini. 
document 19: title: Paul Koulibaly text: Keba Paul Koulibaly is a Burkinabé football defender who plays for the Burkina Faso national football team. He plays as a centre back or a left back. He currently plays for ENPPI Club in Egypt. 
Question: Who played the girlfriend of Alex P. Keaton's actor on Family Ties in Back to the Future? 
Answer: 
"""

results, input_ids, attention = manual_infer_with_llama_with_attention(input)

attention = attention * 10000

attention = attention[0, 0]

id2token = []
for id in input_ids:
    id2token.append(tokenizer.decode(id.item()))

id2token = id2token[0:]

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 假设attentions是你从模型中获取的注意力权重矩阵
# attentions.shape == (num_heads, sequence_length, sequence_length)
# 为了示例，我们这里创建一个随机矩阵
num_heads = 1
sequence_length = 10
# attentions = np.random.rand(num_heads, sequence_length, sequence_length)

# 选择要可视化的头（这里我们只有一个头，所以选择第0个）
attention = attention.cpu().detach().numpy()

# 使用Seaborn的heatmap函数来绘制热力图
plt.figure(figsize=(100, 80))
sns.heatmap(attention, annot=True, fmt=".2f", cmap='viridis', xticklabels=id2token, yticklabels=id2token, vmax=100)

# fig, ax = plt.subplots()
# ax.imshow(attention, vmax=100)
# ax.set_xticks(np.arange(len(id2token)), labels=id2token)
# ax.set_yticks(np.arange(len(id2token)), labels=id2token)

# 添加标题和轴标签
plt.title('Attention Weights Heatmap')
plt.xlabel('Key Positions')
plt.ylabel('Query Positions')

# 显示热力图
plt.show()
plt.savefig('attention_test_1.png', dpi=300, format='png')
# np.save('attention.npy', attention)

with open('./token.json', 'w') as fp:
    json.dump(id2token, fp)