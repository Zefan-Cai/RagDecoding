import torch
import tqdm
import json
import time
import asyncio
import os
import numpy as np
import torch.nn.functional as F
from importlib import import_module
from transformers import StoppingCriteria
from open_instruct.eval.dispatch_openai_requests import dispatch_openai_chat_requesets, dispatch_openai_prompt_requesets


class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)
    
    
@torch.no_grad()
def generate_completions(model, tokenizer, prompts, batch_size=1, stop_id_sequences=None, add_special_tokens=True, disable_tqdm=False, **generation_kwargs):
    generations = []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=add_special_tokens)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()


        try:
            batch_outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                stopping_criteria=[KeyWordsCriteria(stop_id_sequences)] if stop_id_sequences else None,
                **generation_kwargs
            )
        
            # the stopping criteria is applied at batch level, so if other examples are not stopped, the entire batch will continue to generate.
            # so some outputs still have the stop sequence, which we need to remove.
            if stop_id_sequences:
                for output_idx in range(batch_outputs.shape[0]):
                    for token_idx in range(batch_input_ids.shape[1], batch_outputs.shape[1]):
                        if any(batch_outputs[output_idx, token_idx: token_idx+len(stop_sequence)].tolist() == stop_sequence for stop_sequence in stop_id_sequences):
                            batch_outputs[output_idx, token_idx:] = tokenizer.pad_token_id
                            break

            # remove the prompt from the output
            # we need to re-encode the prompt because we need to make sure the special tokens are treated the same way as in the outputs.
            # we changed our previous way of truncating the output token ids dicrectly because some tokenizer (e.g., llama) won't add space token before the first token.
            # space is important for some tasks (e.g., code completion).
            batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
            batch_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
            
            # print(f"debug batch_prompts {batch_prompts[0]}")
            
            # duplicate the prompts to match the number of return sequences
            batch_prompts = [prompt for prompt in batch_prompts for _ in range(num_return_sequences)]
            batch_generations = [
                output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
            ]
        except Exception as e:
            print("Error when generating completions for batch:")
            print(batch_prompts)
            print("Error message:")
            print(e)
            print("Use empty string as the completion.")
            batch_generations = [""] * len(batch_prompts) * num_return_sequences

        generations += batch_generations

        # for prompt, generation in zip(batch_prompts, batch_generations):
        #     print("========")
        #     print(prompt)
        #     print("--------")
        #     print(generation)

        if not disable_tqdm:
            progress.update(len(batch_prompts)//num_return_sequences)

    assert len(generations) == len(prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"
    return generations




def get_relative_top_filter(scores: torch.FloatTensor, relative_top: float = 0.1, min_tokens_to_keep: int = 1):
    scores_normalized = scores.log_softmax(dim=-1) 
    sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
    min_thresh = sorted_logits[..., min_tokens_to_keep-1] 
    probs_max = torch.max(scores_normalized, dim=-1).values
    probs_thresh = probs_max + np.log(relative_top)
    probs_thresh = torch.min(min_thresh, probs_thresh)
    probs_thresh = probs_thresh.unsqueeze(-1)
    return scores_normalized < probs_thresh




@torch.no_grad()
def generate_completions_RAG(model, tokenizer, prompts_withDoc, prompts_withoutDoc, batch_size=1, stop_id_sequences=None, add_special_tokens=True, disable_tqdm=False, max_length=30, **generation_kwargs):
    
    generations = []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts_withDoc), desc="Generating Completions")

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    for i in range(0, len(prompts_withDoc), batch_size):
        batch_prompts_withDoc = prompts_withDoc[i:i+batch_size]
        tokenized_prompts_withDoc = tokenizer(batch_prompts_withDoc, padding="longest", return_tensors="pt", add_special_tokens=add_special_tokens)
        batch_input_ids_withDoc = tokenized_prompts_withDoc.input_ids
        attention_mask_withDoc = tokenized_prompts_withDoc.attention_mask

        batch_prompts_withoutDoc = prompts_withoutDoc[i:i+batch_size]
        tokenized_prompts_withoutDoc = tokenizer(batch_prompts_withoutDoc, padding="longest", return_tensors="pt", add_special_tokens=add_special_tokens)
        batch_input_ids_withoutDoc = tokenized_prompts_withoutDoc.input_ids
        attention_mask_withoutDoc = tokenized_prompts_withoutDoc.attention_mask

        # print(f"debug tokenized_prompts_withDoc {tokenized_prompts_withDoc.input_ids[0]}")
        # print(f"debug tokenized_prompts_withoutDoc {tokenized_prompts_withoutDoc.input_ids[0]}")

        if model.device.type == "cuda":
            batch_input_ids_withDoc = batch_input_ids_withDoc.cuda()
            attention_mask_withDoc = attention_mask_withDoc.cuda()
            batch_input_ids_withoutDoc = batch_input_ids_withoutDoc.cuda()
            attention_mask_withoutDoc = attention_mask_withoutDoc.cuda()
            batch_input_ids_withoutDoc_copy = batch_input_ids_withoutDoc.clone()

        for _ in range(max_length):
            # Get logits of the next token
            output_withDoc = model(batch_input_ids_withDoc).logits
            output_withoutDoc = model(batch_input_ids_withoutDoc).logits
            next_token_logits_withDoc = output_withDoc[:, -1, :]
            next_token_logits_withoutDoc = output_withoutDoc[:, -1, :]

            # print(f"debug tokenized_prompts_withDoc {batch_input_ids_withDoc[0]}")
            # print(f"debug tokenized_prompts_withoutDoc {batch_input_ids_withoutDoc[0]}")

            contrastive_next_token_logits = next_token_logits_withDoc - next_token_logits_withoutDoc
            # print(next_token_logits_doc[:10])
            # print(next_token_logits[:10])
            # print(contrastive_next_token_logits[:10])
            relative_top_value=-1000.0

            relative_top_mask = get_relative_top_filter(next_token_logits_withDoc, 0.1)
            diff_logits = torch.where(relative_top_mask, relative_top_value, contrastive_next_token_logits)
            
            # log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()
            
            # log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()


            # Choose the most likely next token (you can also sample)
            next_token = F.log_softmax(diff_logits, dim=-1)
            next_token = torch.argmax(next_token, dim=-1)


            # Append the new token to the existing sequence
            # print(input_ids.shape)
            # print(next_token.shape)
            batch_input_ids_withoutDoc = torch.cat([batch_input_ids_withoutDoc, next_token.unsqueeze(-1)], dim=-1)
            batch_input_ids_withDoc = torch.cat([batch_input_ids_withDoc, next_token.unsqueeze(-1)], dim=-1)

            # Check if the last token is an end-of-sequence token
            # if next_token in tokenizer.all_special_ids:
            #     break

            """
                    # Check if the last token is an end-of-sequence token
                    if next_token in tokenizer.all_special_ids:
                        break

                    try:
                        batch_outputs = model.generate(
                            input_ids=batch_input_ids,
                            attention_mask=attention_mask,
                            stopping_criteria=[KeyWordsCriteria(stop_id_sequences)] if stop_id_sequences else None,
                            **generation_kwargs
                        )
                    
                        # the stopping criteria is applied at batch level, so if other examples are not stopped, the entire batch will continue to generate.
                        # so some outputs still have the stop sequence, which we need to remove.
                        if stop_id_sequences:
                            for output_idx in range(batch_outputs.shape[0]):
                                for token_idx in range(batch_input_ids.shape[1], batch_outputs.shape[1]):
                                    if any(batch_outputs[output_idx, token_idx: token_idx+len(stop_sequence)].tolist() == stop_sequence for stop_sequence in stop_id_sequences):
                                        batch_outputs[output_idx, token_idx:] = tokenizer.pad_token_id
                                        break

                        # remove the prompt from the output
                        # we need to re-encode the prompt because we need to make sure the special tokens are treated the same way as in the outputs.
                        # we changed our previous way of truncating the output token ids dicrectly because some tokenizer (e.g., llama) won't add space token before the first token.
                        # space is important for some tasks (e.g., code completion).

            """

            batch_outputs = tokenizer.batch_decode(batch_input_ids_withoutDoc, skip_special_tokens=True)
            batch_prompts = tokenizer.batch_decode(batch_input_ids_withoutDoc_copy, skip_special_tokens=True)
            # duplicate the prompts to match the number of return sequences
            batch_prompts = [prompt for prompt in batch_prompts for _ in range(num_return_sequences)]
            batch_generations = [
                output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
            ]
        # except Exception as e:
        #     print("Error when generating completions for batch:")
        #     print(batch_prompts)
        #     print("Error message:")
        #     print(e)
        #     print("Use empty string as the completion.")
        #     batch_generations = [""] * len(batch_prompts) * num_return_sequences



        generations += batch_generations

        # for prompt, generation in zip(batch_prompts, batch_generations):
        #     print("========")
        #     print(prompt)
        #     print("--------")
        #     print(generation)

        if not disable_tqdm:
            progress.update(len(batch_prompts)//num_return_sequences)



    assert len(generations) == len(prompts_withDoc) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"
    return generations









@torch.no_grad()
def get_next_word_predictions(model, tokenizer, prompts, candidate_token_ids=None, batch_size=1, return_token_predictions=False, add_special_tokens=True, disable_tqdm=False):
    predictions, probs = [], []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Getting Predictions")

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i: i+batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=add_special_tokens)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()

        batch_logits = model(input_ids=batch_input_ids, attention_mask=attention_mask).logits[:, -1, :]
        batch_probs = torch.softmax(batch_logits, dim=-1)
        if candidate_token_ids is not None:
            batch_probs = batch_probs[:, candidate_token_ids]
        batch_prediction_indices = torch.argmax(batch_probs, dim=-1)
        if return_token_predictions:
            if candidate_token_ids is not None:
                candidate_tokens = tokenizer.convert_ids_to_tokens(candidate_token_ids)
                batch_predictions = [candidate_tokens[idx] for idx in batch_prediction_indices]
            else:
                batch_predictions = tokenizer.convert_ids_to_tokens(batch_prediction_indices)
            predictions += batch_predictions
        else:
            predictions += batch_prediction_indices.tolist()
        probs += batch_probs.tolist()

        if not disable_tqdm:
            progress.update(len(batch_prompts))

    assert len(predictions) == len(prompts), "number of predictions should be equal to number of prompts"
    return predictions, probs


@torch.no_grad()
def score_completions(model, tokenizer, scoring_examples, batch_size=1, aggregation="sum", disable_tqdm=False):
    '''
    Each scoring example is a dict, which contains the following keys:
    - prompt: the prompt to score
    - completions: a list of completions to score
    '''
    
    # unroll the scoring examples
    unrolled_examples = []
    for scoring_example in scoring_examples:
        prompt = scoring_example["prompt"]
        for completion in scoring_example["completions"]:
            unrolled_examples.append({
                "prompt": prompt,
                "completion": completion
            })
    
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(unrolled_examples), desc="Scoring Completions")

    scores = []
    for i in range(0, len(unrolled_examples), batch_size):
        batch_prompts = [example["prompt"] for example in unrolled_examples[i:i+batch_size]]
        batch_examples = [
            (example["prompt"] if example["prompt"][-1] in ["\n", " "] else example["prompt"] + " ")
            + example["completion"] for example in unrolled_examples[i:i+batch_size]
        ]
        tokenized_batch = tokenizer(batch_examples, padding="longest", return_tensors="pt")
        if model.device.type == "cuda":
            tokenized_batch = {
                key: value.cuda() for key, value in tokenized_batch.items()
            }
        outputs = model(**tokenized_batch)

        for example_idx, (prompt, example) in enumerate(zip(batch_prompts, batch_examples)):
            tokenized_prompt = tokenizer(prompt, padding=False, return_tensors="pt").input_ids.squeeze(0)
            tokenized_example = tokenizer(example, padding=False, return_tensors="pt").input_ids.squeeze(0)
            completion_ids = tokenized_example[len(tokenized_prompt):]
            
            # get the logits for the entire example, removing the padding logits
            if tokenizer.padding_side == "right":
                example_logits = outputs.logits[example_idx, :len(tokenized_example), :]
            else:            
                example_logits = outputs.logits[example_idx, -len(tokenized_example):, :]

            # get the logits for the completion portion - note we need to shift the index left by 1 because logits are computed for the next token
            completion_logits = example_logits[len(tokenized_prompt)-1:len(tokenized_example)-1, :]
            completion_log_probs = torch.log_softmax(completion_logits, dim=-1)[range(len(completion_ids)), completion_ids]

            if aggregation == "sum":
                score = completion_log_probs.sum().item()
            elif aggregation == "mean":
                score = completion_log_probs.mean().item()
            elif aggregation == "max":
                score = completion_log_probs.max().item()
            else:
                raise ValueError("Invalid aggregation method: {}".format(aggregation))
            scores.append(score)

        if not disable_tqdm:
            progress.update(len(batch_examples))

    # roll up the scores
    rolled_up_scores = {}
    for unrolled_example, score in zip(unrolled_examples, scores):
        prompt = unrolled_example["prompt"]
        completion = unrolled_example["completion"]
        if prompt not in rolled_up_scores:
            rolled_up_scores[prompt] = {}
        rolled_up_scores[prompt][completion] = score

    return rolled_up_scores



def load_hf_lm_and_tokenizer(
        model_name_or_path, 
        tokenizer_name_or_path=None, 
        device_map="auto", 
        torch_dtype="auto",
        load_in_8bit=False, 
        convert_to_half=False,
        gptq_model=False,
        use_fast_tokenizer=True,
        padding_side="left",
    ):
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM, GPTNeoXForCausalLM

    if gptq_model:
        from auto_gptq import AutoGPTQForCausalLM
        model_wrapper = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path, device="cuda:0", use_triton=True
        )
        model = model_wrapper.model  
    elif load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            device_map=device_map, 
            load_in_8bit=True
        )
    else:
        if device_map:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=device_map, torch_dtype=torch_dtype)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch_dtype)
            if torch.cuda.is_available():
                model = model.cuda()
        if convert_to_half:
            model = model.half()
    model.eval()

    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_name_or_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=use_fast_tokenizer)
    except:
        # some tokenizers (e.g., GPTNeoXTokenizer) don't have the slow or fast version, so we just roll back to the default one
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    # set padding side to left for batch generation
    tokenizer.padding_side = padding_side
    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # for OPT and Pythia models, we need to set tokenizer.model_max_length to model.config.max_position_embeddings 
    # to avoid wrong embedding index.    
    if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, OPTForCausalLM):
        tokenizer.model_max_length = model.config.max_position_embeddings
        print("Set tokenizer.model_max_length to model.config.max_position_embeddings: {}".format(model.config.max_position_embeddings))
        
    return model, tokenizer



def query_openai_chat_model(engine, instances, output_path=None, batch_size=10, retry_limit=5, reuse_existing_outputs=True, **completion_kwargs):
    '''
    Query OpenAI chat model and save the results to output_path.
    `instances` is a list of dictionaries, each dictionary contains a key "prompt" and a key "id".
    '''
    existing_data = {}
    if reuse_existing_outputs and output_path is not None and os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                instance = json.loads(line)
                existing_data[instance["id"]] = instance

    # by default, we use temperature 0.0 to get the most likely completion.
    if "temperature" not in completion_kwargs:
        completion_kwargs["temperature"] = 0.0

    results = []
    if output_path is not None:
        fout = open(output_path, "w")

    retry_count = 0
    progress_bar = tqdm.tqdm(total=len(instances))
    for i in range(0, len(instances), batch_size):
        batch = instances[i:i+batch_size]
        if all([x["id"] in existing_data for x in batch]):
            results.extend([existing_data[x["id"]] for x in batch])
            if output_path is not None:
                for instance in batch:
                    fout.write(json.dumps(existing_data[instance["id"]]) + "\n")
                    fout.flush()
            progress_bar.update(batch_size)
            continue
        messages_list = []
        for instance in batch:
            messages = [{"role": "user", "content": instance["prompt"]}]
            messages_list.append(messages)
        while retry_count < retry_limit:
            try:
                outputs = asyncio.run(
                    dispatch_openai_chat_requesets(
                    messages_list=messages_list,
                    model=engine,
                    **completion_kwargs,
                ))
                retry_count = 0
                break
            except Exception as e:
                retry_count += 1
                print(f"Error while requesting OpenAI API.")
                print(e)
                print(f"Sleep for {30*retry_count} seconds.")
                time.sleep(30*retry_count)
                print(f"Retry for the {retry_count} time.")
        if retry_count == retry_limit:
            raise RuntimeError(f"Failed to get response from OpenAI API after {retry_limit} retries.")
        assert len(outputs) == len(batch)
        for instance, output in zip(batch, outputs):
            instance[f"output"] = output["choices"][0]["message"]["content"]
            instance["response_metadata"] = output
            results.append(instance)
            if output_path is not None:
                fout.write(json.dumps(instance) + "\n")
                fout.flush()
        progress_bar.update(batch_size)
    return results
 

def query_openai_model(engine, instances, output_path=None, batch_size=10, retry_limit=5, reuse_existing_outputs=True, **completion_kwargs):
    '''
    Query OpenAI chat model and save the results to output_path.
    `instances` is a list of dictionaries, each dictionary contains a key "prompt" and a key "id".
    '''
    existing_data = {}
    if reuse_existing_outputs and output_path is not None and os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                instance = json.loads(line)
                existing_data[instance["id"]] = instance

    # by default, we use temperature 0.0 to get the most likely completion.
    if "temperature" not in completion_kwargs:
        completion_kwargs["temperature"] = 0.0

    results = []
    if output_path is not None:
        fout = open(output_path, "w")

    retry_count = 0
    progress_bar = tqdm.tqdm(total=len(instances))
    for i in range(0, len(instances), batch_size):
        batch = instances[i:i+batch_size]
        if all([x["id"] in existing_data for x in batch]):
            results.extend([existing_data[x["id"]] for x in batch])
            if output_path is not None:
                for instance in batch:
                    fout.write(json.dumps(existing_data[instance["id"]]) + "\n")
                    fout.flush()
            progress_bar.update(batch_size)
            continue
        messages_list = []
        for instance in batch:
            messages = instance["prompt"]
            messages_list.append(messages)
        while retry_count < retry_limit:
            try:
                outputs = asyncio.run(
                    dispatch_openai_prompt_requesets(
                    prompt_list=messages_list,
                    model=engine,
                    **completion_kwargs,
                ))
                retry_count = 0
                break
            except Exception as e:
                retry_count += 1
                print(f"Error while requesting OpenAI API.")
                print(e)
                print(f"Sleep for {30*retry_count} seconds.")
                time.sleep(30*retry_count)
                print(f"Retry for the {retry_count} time.")
        if retry_count == retry_limit:
            raise RuntimeError(f"Failed to get response from OpenAI API after {retry_limit} retries.")
        assert len(outputs) == len(batch)
        for instance, output in zip(batch, outputs):
            instance[f"output"] = output["choices"][0]["text"]
            instance["response_metadata"] = output
            results.append(instance)
            if output_path is not None:
                fout.write(json.dumps(instance) + "\n")
                fout.flush()
        progress_bar.update(batch_size)
    return results


def dynamic_import_function(function_path):
    '''
    Dynamically import a function from a path string (e.g., "module.submodule.my_function")
    '''
    module_path, function_name = function_path.rsplit(".", 1)
    module = import_module(module_path)
    function = getattr(module, function_name)
    return function
 