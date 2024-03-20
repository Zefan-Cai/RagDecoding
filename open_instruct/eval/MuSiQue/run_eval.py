import argparse
import os
import re
import json
import tqdm
import glob
import torch
import random
import evaluate
from open_instruct.eval.utils import load_hf_lm_and_tokenizer, generate_completions



exact_match = evaluate.load("exact_match")

@torch.no_grad()
def eval_hf_model(args, model, tokenizer, examples, save_path=None):

    if save_path:
        fout = open(save_path, "w")

    prompts = []
    for example in examples:
        prompt = example["prompt"].strip()
        prompts.append(prompt)

    outputs = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        strategy=args.strategy,
        max_new_tokens=args.max_new_tokens,
        top_k=1,
        batch_size=args.eval_batch_size if args.eval_batch_size else 1
    )


    if save_path:
        fout = open(save_path, "w")
    predictions = []
    for example in examples:
        prediction = outputs.pop(0)
        example["prediction"] = prediction
    if save_path:
        fout.write(json.dumps(examples) + "\n")



def main(args):
    random.seed(42)
    
    if args.use_flash_attn:
        from open_instruct.eval.MuSiQue.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()

    all_tasks = {}

    task_files = [
        os.path.join(args.data_dir, args.evaluation_file),
    ]
    for task_file in tqdm.tqdm(task_files, desc="Loading tasks"):
        data = []
        with open(task_file, "r") as f:
            task_name = os.path.basename(task_file).split(".")[0]
            
            for line in f.readlines():
                data.append(json.loads(line))
            
            all_tasks[task_name] = data
            if args.max_num_examples_per_task:
                if args.sample_method == "first":
                    all_tasks[task_name] = all_tasks[task_name][:args.max_num_examples_per_task]
                elif args.sample_method == "random":
                    all_tasks[task_name] = random.sample(all_tasks[task_name], args.max_num_examples_per_task)

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

    for task_name in tqdm.tqdm(all_tasks.keys(), desc="Evaluating"):
        task_examples = all_tasks[task_name]

        if args.model_name_or_path:
            task_perf = eval_hf_model(
                args, 
                model, 
                tokenizer, 
                task_examples,
                save_path=os.path.join(args.save_dir)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/ace")
    parser.add_argument("--evaluation_file", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--strategy", type=str, default="")
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
