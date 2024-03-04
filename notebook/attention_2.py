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
document 11: title: Saudi Arabia text: The area of modern - day Saudi Arabia formerly consisted of four distinct regions: Hejaz, Najd and parts of Eastern Arabia (Al - Ahsa) and Southern Arabia ('Asir). The Kingdom of Saudi Arabia was founded in 1932 by Ibn Saud. He united the four regions into a single state through a series of conquests beginning in 1902 with the capture of Riyadh, the ancestral home of his family, the House of Saud. Saudi Arabia has since been an absolute monarchy, effectively a hereditary dictatorship governed along Islamic lines. The ultraconservative Wahhabi religious movement within Sunni Islam has been called the predominant feature of Saudi culture '', with its global spread largely financed by the oil and gas trade. Saudi Arabia is sometimes called the Land of the Two Holy Mosques'' in reference to Al - Masjid al - Haram (in Mecca) and Al - Masjid an - Nabawi (in Medina), the two holiest places in Islam. As of 2013, the state had a total population of 28.7 million, of which 20 million were Saudi nationals and 8 million were foreigners. As of 2017, the population is 33 million. The state's official language is Arabic. 
Question: When was the region immediately north of the region where the country that secured southern Lebanon is located and the Persian Gulf established? 
Answer: 
"""

results, input_ids, attention = manual_infer_with_llama_with_attention(input)

attention = attention * 10000

# attention = attention[0, 1]

attention_average = torch.mean(attention, dim=1)

attention_average = attention_average[0]

attention = attention_average

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
# sns.heatmap(attention, annot=True, fmt=".2f", cmap='viridis', xticklabels=id2token, yticklabels=id2token, vmax=100)

fig, ax = plt.subplots()
ax.imshow(attention, vmax=100)
ax.set_xticks(np.arange(len(id2token)), labels=id2token)
ax.set_yticks(np.arange(len(id2token)), labels=id2token)

# 添加标题和轴标签
plt.title('Attention Weights Heatmap')
plt.xlabel('Key Positions')
plt.ylabel('Query Positions')

# 显示热力图
plt.show()
plt.savefig('attention_test_1080p_2_head_average.png', dpi=300, format='png')
# np.save('attention.npy', attention)

with open('./token.json', 'w') as fp:
    json.dump(id2token, fp)