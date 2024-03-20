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
[INST] <<SYS>>
             You are given some documents, and you need to answer a question based on these documents.
            Your answer should be less than five words.
              
<</SYS>>

Document: Roman Republic After having declined in size following the subjugation of the Mediterranean, the Roman navy underwent short-term upgrading and revitalisation in the late Republic to meet several 
new demands. Under Caesar, an invasion fleet was assembled in the English Channel to allow the invasion of Britannia; under Pompey, a large fleet was raised in the Mediterranean Sea to clear the sea of Cili
cian pirates. During the civil war that followed, as many as a thousand ships were either constructed or pressed into service from Greek cities. 
Document: North Sea The North Sea is bounded by the Orkney Islands and east coast of Great Britain to the west and the northern and central European mainland to the east and south, including Norway, Denmark
, Germany, the Netherlands, Belgium, and France. In the southwest, beyond the Straits of Dover, the North Sea becomes the English Channel connecting to the Atlantic Ocean. In the east, it connects to the Ba
ltic Sea via the Skagerrak and Kattegat, narrow straits that separate Denmark from Norway and Sweden respectively. In the north it is bordered by the Shetland Islands, and connects with the Norwegian Sea, w
hich lies in the very north - eastern part of the Atlantic. 
Document: Rhine The Rhine (Romansh: Rein, German: Rhein, French: le Rhin, Dutch: Rijn) is a European river that begins in the Swiss canton of Graubünden in the southeastern Swiss Alps, forms part of the Swi
ss-Austrian, Swiss-Liechtenstein border, Swiss-German and then the Franco-German border, then flows through the Rhineland and eventually empties into the North Sea in the Netherlands. The biggest city on th
e river Rhine is Cologne, Germany with a population of more than 1,050,000 people. It is the second-longest river in Central and Western Europe (after the Danube), at about 1,230 km (760 mi),[note 2][note 1
] with an average discharge of about 2,900 m3/s (100,000 cu ft/s). 
Question: Who sent naval ships to the body of water that joins the Atlantic and the sea where the Rhine ends? 
Answer:  [/INST]
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