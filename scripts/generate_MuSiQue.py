import json

input_data = []
with open('./MuSiQue/musique_full_v1.0_dev.jsonl', 'r') as fp:
    for line in fp:
        input_data.append(json.loads(line))
document_number = 20

output_data = []

for index in range(len(input_data)):
    
    prompt = "You are given some documents, and you need to answer a question based on these documents. \n"
    
    this_document_number = min(document_number, len(input_data[index]['paragraphs']))
    
    for index_document in range(this_document_number):
        title = input_data[index]['paragraphs'][index_document]["title"]
        paragraph_text = input_data[index]['paragraphs'][index_document]["paragraph_text"]
        is_supporting = input_data[index]['paragraphs'][index_document]["is_supporting"]
        if is_supporting == True:
            prompt += f"Document: {title} {paragraph_text} \n"
    prompt += f"Question: {input_data[index]['question']} \n"
    prompt += "Answer: "
    
    answer = input_data[index]["answer"]
    
    output_data.append({
        "prompt": prompt,
        "completion": answer
    })

with open(f'./data/musique_support_doc.jsonl', 'w') as fp:
    for line in output_data:
        json.dump(line, fp)
        fp.write('\n')