import argparse
import os
import re
import json
import tqdm
import glob
import torch
import random
import evaluate
from open_instruct.eval.utils import load_hf_lm_and_tokenizer, generate_completions, generate_completions_RAG



exact_match = evaluate.load("exact_match")

@torch.no_grad()
def eval_hf_model(args, model, tokenizer, examples, save_path=None):
    
    examples_withDoc = examples["data_withDoc"]
    examples_withoutDoc = examples["data_withoutDoc"]

    if save_path:
        fout = open(save_path, "w")

    prompts_withDoc = []
    for example in examples_withDoc:
        prompt = example["prompt"].strip()
        prompts_withDoc.append(prompt)

    prompts_withoutDoc = []
    for example in examples_withoutDoc:
        prompt = example["prompt"].strip()
        prompts_withoutDoc.append(prompt)

    outputs = generate_completions_RAG(
        model=model,
        tokenizer=tokenizer,
        prompts_withDoc=prompts_withDoc,
        prompts_withoutDoc=prompts_withoutDoc,
        max_length=args.max_new_tokens,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.eval_batch_size if args.eval_batch_size else 1
    )


    if save_path:
        fout = open(save_path, "w")

    for example in examples_withDoc:
        prediction = outputs.pop(0)
        example["prediction"] = prediction
    if save_path:
        fout.write(json.dumps(examples_withDoc) + "\n")



def main(args):
    random.seed(42)
    
    if args.use_flash_attn:
        from open_instruct.eval.MuSiQue.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()

    all_tasks = {}

    task_files = [
        os.path.join(args.data_dir, args.file_withoutDoc),
        os.path.join(args.data_dir, args.file_withDoc)
    ]

    file_withoutDoc = task_files[0]
    data_withoutDoc = []

    with open(file_withoutDoc, "r") as f:
        
        for line in f.readlines():
            data_withoutDoc.append(json.loads(line))
        
        all_tasks["data_withoutDoc"] = data_withoutDoc
        if args.max_num_examples_per_task:
            if args.sample_method == "first":
                all_tasks["data_withoutDoc"] = all_tasks["data_withoutDoc"][:args.max_num_examples_per_task]
            elif args.sample_method == "random":
                all_tasks["data_withoutDoc"] = random.sample(all_tasks["data_withoutDoc"], args.max_num_examples_per_task)

    file_withDoc = task_files[1]
    data_withDoc = []

    with open(file_withDoc, "r") as f:
        
        for line in f.readlines():
            data_withDoc.append(json.loads(line))
        
        all_tasks["data_withDoc"] = data_withDoc
        if args.max_num_examples_per_task:
            if args.sample_method == "first":
                all_tasks["data_withDoc"] = all_tasks["data_withDoc"][:args.max_num_examples_per_task]
            elif args.sample_method == "random":
                all_tasks["data_withDoc"] = random.sample(all_tasks["data_withDoc"], args.max_num_examples_per_task)


    os.makedirs(args.save_dir, exist_ok=True)
    # os.makedirs(os.path.join(args.save_dir, "predictions"), exist_ok=True)

    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path, 
            tokenizer_name_or_path=args.tokenizer_name_or_path, 
            load_in_8bit=args.load_in_8bit,
            convert_to_half=args.load_in_half,
            gptq_model=args.gptq
        )

    if args.model_name_or_path:
        task_perf = eval_hf_model(
            args, 
            model, 
            tokenizer, 
            all_tasks,
            save_path=os.path.join(args.save_dir, f"RAG_{args.file_withDoc}")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/ace")
    parser.add_argument("--file_withoutDoc", type=str, default="")
    parser.add_argument("--file_withDoc", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None, help="if specified, we will load the tokenizer from here.")
    parser.add_argument("--max_num_examples_per_task", type=int, default=None, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="")
    parser.add_argument("--sample_method", type=str, default="random, first")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    parser.add_argument("--load_in_8bit", action="store_true", help="load model in 8bit mode, which will reduce memory and speed up inference.")
    parser.add_argument("--load_in_half", action="store_true", help="load model in 8bit mode, which will reduce memory and speed up inference.")
    parser.add_argument("--gptq", action="store_true", help="If given, we're evaluating a 4-bit quantized GPTQ model.")
    parser.add_argument("--use_flash_attn", action="store_true", help="If passed, will use flash attention to train the model.")
    args = parser.parse_args()

    main(args)
