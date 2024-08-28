import os
import sys
import pdb
import json
import fire
import torch
import random
import transformers
import pandas as pd
from tqdm import tqdm
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    GenerationConfig,
)

def api_call_criteria(self, input_ids, scores, **kwargs):
    if "Action Input: {" in tokenizer.decode(input_ids[0]):
        return True
    return False
transformers.generation.stopping_criteria.MaxTimeCriteria.__call__ = api_call_criteria

def main(
    model_name_or_path,
    lora_path,
    data_path,
    load_in_8bit=True,
    max_new_tokens=512,
    max_length=2048,
    device="cuda",
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    num_beams=1,
):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        load_in_8bit=load_in_8bit,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    if lora_path:
        model = PeftModel.from_pretrained(
            model,
            lora_path,
            torch_dtype=torch.float16,
        )
    if tokenizer.pad_token_id == None:
        tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))
    use_gpu = (True if device == "cuda" else False)
    model.eval()

    num_search = 0
    num_no_search = 0
    num_search_call = 0
    num_search_no_call = 0
    total_search = 0
    total_no_search = 0
    total_search_call = 0
    total_search_no_call = 0
    devset = json.load(open(data_path, "r"))
    random.seed(42)
    random.shuffle(devset)

    for cur_cnt, item in enumerate(tqdm(devset)):
        query = item["query"]
        gt_label = item["api_label"]

        if item["is_call"]:
            total_search_call += 1
        else:
            total_search_no_call += 1
        
        tool_usage = ""
        for d in item["api_list"]:
            tool_usage += d["name"] + ": " + d["description"] + "\nParameters: " + str(d["parameters"]) + "\nOutput: Successful response.\n" 
        
        template = """Answer the following questions as best you can. You have access to the following tools:

%s
NoCallAPI: If you believe that there are no suitable tool to address the query, you can use this.
Parameters: {"is_call": "bool"}
Output: Successful response.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of %s
Action Input: the input to the action, in JSON format.
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: %s
Thought:"""

        prompt = template % (tool_usage[:-1], item["api_unique"], query)
        inputs = tokenizer(
            prompt,
            padding=True,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        inputs_len = inputs["input_ids"].shape[1]
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
        )

        with torch.no_grad():
            generated_outputs = model.generate(
                input_ids=(inputs["input_ids"].cuda() if use_gpu else inputs["input_ids"]),
                attention_mask=(inputs["attention_mask"].cuda() if use_gpu else inputs["attention_mask"]),
                generation_config=generation_config,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                max_time=60,
            )
        decoded_output = tokenizer.batch_decode(generated_outputs[..., inputs_len:], skip_special_tokens=True)[0]
        
        if item["is_call"]:
            if "Action: " in decoded_output:
                try:
                    call_api = decoded_output.split("Action: ")[-1].split("\n")[0]
                    if call_api in gt_label:
                        num_search_call += 1
                except:
                    pass
        else:
            if "NoCallAPI" in decoded_output:
                num_search_no_call += 1

        if (cur_cnt + 1) % 10 == 0 or (cur_cnt + 1) == len(devset):
            print("\nnum_no_search:", num_no_search)
            print("num_search:", num_search)
            print("num_search_no_call:", num_search_no_call)
            print("num_search_call:", num_search_call)
            print("total_no_search:", total_no_search)
            print("total_search:", total_search)
            print("total_search_no_call:", total_search_no_call)
            print("total_search_call:", total_search_call)
            if total_no_search != 0:
                print("no_search_acc:", num_no_search / total_no_search)
            if total_search != 0:
                print("search_acc:", num_search / total_search)
            if total_no_search != 0 or total_search != 0:
                print("total_search_acc:", (num_no_search + num_search) / (total_no_search + total_search))
            if total_search_no_call != 0:
                print("no_call_acc:", num_search_no_call / total_search_no_call)
            if total_search_call != 0:
                print("call_acc", num_search_call / total_search_call)
            if total_search_no_call != 0 or total_search_call != 0:
                print("total_call_acc", (num_search_no_call + num_search_call) / (total_search_no_call + total_search_call))

if __name__ == "__main__":
    fire.Fire(main)
