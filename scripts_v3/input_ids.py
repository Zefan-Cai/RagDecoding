from transformers import BertTokenizer
import json

import torch
import torch.nn.functional as F

data_supportdoc_llama2chat = []
with open('/mnt/users/v-caizefan/RAGDecoding/results/results_v3_full/musique_llama2chat_supportdoc.json', 'r') as fp:
    for line in fp.readlines():
        data_supportdoc_llama2chat.append(json.loads(line))

data_supportdoc_llama2chat = data_supportdoc_llama2chat[0]

data_supportdoc_llama2chat_rag = []
with open('/mnt/users/v-caizefan/RAGDecoding/results/results_v3_full/RAG_musique_llama2chat_supportdoc.json', 'r') as fp:
    for line in fp.readlines():
        data_supportdoc_llama2chat_rag.append(json.loads(line))

data_supportdoc_llama2chat_rag = data_supportdoc_llama2chat_rag[0]


all_completion = []
all_prediction = []
all_prediction_rag = []

for index in range(len(data_supportdoc_llama2chat_rag)):
    
    prompt = data_supportdoc_llama2chat_rag[index]["prompt"]
    
    completion = data_supportdoc_llama2chat_rag[index]["completion"]
    prediction = data_supportdoc_llama2chat[index]["prediction"]
    prediction_rag = data_supportdoc_llama2chat_rag[index]["prediction"]
    
    all_completion.append(completion)
    all_prediction.append(prediction)
    all_prediction_rag.append(prediction_rag)



# 初始化分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# 将所有文本一起编码到一个批次中，并设置最大序列长度
encoded_inputs_completion = tokenizer(all_completion, padding=True, truncation=True, max_length=2, return_tensors="pt")
encoded_inputs_prediction = tokenizer(all_prediction, padding=True, truncation=True, max_length=2, return_tensors="pt")
encoded_inputs_prediction_rag = tokenizer(all_prediction_rag, padding=True, truncation=True, max_length=2, return_tensors="pt")


# encoded_inputs_completion = encoded_inputs_completion["input_ids"]
# encoded_inputs_prediction = encoded_inputs_prediction["input_ids"]
# encoded_inputs_prediction_rag = encoded_inputs_prediction_rag["input_ids"]

# target_size = encoded_inputs_prediction_rag.shape # 比如我们想要结果是4x5的tensor

# # 计算每个维度需要填充的量
# padding_left = padding_top = 0 # 填充到左边和上边的量，根据需要调整
# padding_right = target_size[1] - encoded_inputs_completion.size(1) - padding_left
# padding_bottom = target_size[0] - encoded_inputs_completion.size(0) - padding_top

# # 应用填充
# padded_encoded_inputs_prediction = F.pad(encoded_inputs_completion, (padding_left, padding_right, padding_top, padding_bottom), "constant", 0)




# print(f"debug encoded_inputs_completion {encoded_inputs_completion.shape}")
# print(f"debug encoded_inputs_prediction {encoded_inputs_prediction.shape}")
# print(f"debug encoded_inputs_prediction_rag {encoded_inputs_prediction_rag.shape}")

encoded_inputs_completion = encoded_inputs_completion["input_ids"].tolist()
encoded_inputs_prediction = encoded_inputs_prediction["input_ids"].tolist()
encoded_inputs_prediction_rag = encoded_inputs_prediction_rag["input_ids"].tolist()

# 保存为JSON
with open("./musique_full_encoded_inputs_completion.json", "w") as file:
    json.dump(encoded_inputs_completion, file)
    
# 保存为JSON
with open("./musique_full_encoded_inputs_prediction.json", "w") as file:
    json.dump(encoded_inputs_prediction, file)
    
# 保存为JSON
with open("./musique_full_encoded_inputs_prediction_rag.json", "w") as file:
    json.dump(encoded_inputs_prediction_rag, file)