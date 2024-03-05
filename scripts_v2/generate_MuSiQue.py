import os
import json
import argparse
from tqdm import tqdm




class Data():
    def __init__(self, args):
        self.args = args
        
        self.service2dir = {
            "pku": "/home/caizf/projects/RagDecoding"
        }
        self.base_dir = self.service2dir[self.args.service]
        if self.args.base_dir: self.base_dir = self.args.base_dir

        input_data = []
        with open(os.path.join(self.args.base_dir, self.args.input_dir), 'r') as fp:
            for line in fp:
                input_data.append(json.loads(line))
        document_number = 20

        output_data = []

        for index in range(len(input_data)):
            
            if self.args.model == "llama2chat":
                prompt = ""
            elif self.args.model == "llama2":
                if self.args.setting == "0doc": 
                    prompt = "You are given a question, and you need to answer it. \n"
                else:
                    prompt = "You are given some documents, and you need to answer a question based on these documents. \n"
            
            this_document_number = min(document_number, len(input_data[index]['paragraphs']))
            
            
            for index_document in range(this_document_number):
                title = input_data[index]['paragraphs'][index_document]["title"]
                paragraph_text = input_data[index]['paragraphs'][index_document]["paragraph_text"]
                is_supporting = input_data[index]['paragraphs'][index_document]["is_supporting"]

                if self.args.setting == "0doc": 
                    pass
                if self.args.setting == "20doc": 
                    prompt += f"Document: {title} {paragraph_text} \n"
                if self.args.setting == "supportdoc":
                    if is_supporting == True:
                        prompt += f"Document: {title} {paragraph_text} \n"
            prompt += f"Question: {input_data[index]['question']} \n"
            prompt += "Your answer should be less than five words. \n"
            prompt += "Answer: "
            
            answer = input_data[index]["answer"]
            
            if self.args.model == "llama2chat":
                prompt = self.get_prompt_llama2chat(prompt, self.args.setting)
            elif self.args.model == "llama2":
                prompt = prompt
            
            output_data.append({
                "prompt": prompt,
                "completion": answer
            })

        with open(os.path.join(self.args.base_dir, self.args.output_dir, self.args.output_name), 'w') as fp:
            for line in output_data:
                json.dump(line, fp)
                fp.write('\n')

    def get_prompt_llama2chat(self, prompt, setting):
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        # DEFAULT_SYSTEM_PROMPT = """\
        # You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
        
        # If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

        if setting == "0doc":
            DEFAULT_SYSTEM_PROMPT = """\
            You are a helpful assistant. Always answer as helpfully as possible.
            """
        else:
            DEFAULT_SYSTEM_PROMPT = """\
            You are a helpful assistant. Always answer as helpfully as possible.
            You are given some documents, and you need to answer a question based on these documents.
            """

        # prompt_tokens=f"<s>{B_INST} {B_SYS} { DEFAULT_SYSTEM_PROMPT} {E_SYS} {prompt} {E_INST}"
        prompt_tokens=f"{B_INST} {B_SYS} { DEFAULT_SYSTEM_PROMPT} {E_SYS} {prompt} {E_INST}"
        return prompt_tokens

def main():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--service', default='pku', type=str, help='pku')
    parser.add_argument('--model', default='llama2chat', type=str, help='llama2 or llama2chat')
    parser.add_argument('--base_dir', default='', type=str, help='pku: /home/caizf/projects/RagDecoding/ ')
    parser.add_argument('--input_dir', default='', type=str, help='dir to ')
    parser.add_argument('--output_dir', default='', type=str, help='dir to ')
    parser.add_argument('--output_name', default='', type=str, help='')
    parser.add_argument('--setting', default="0doc", type=str, help='')

    args = parser.parse_args()

    data = Data(args)




if __name__ == "__main__":
    main()