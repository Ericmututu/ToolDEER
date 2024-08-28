import os
import sys
import pdb
import json
import fire
import torch
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
    def convert(token_str):
        ids_list = tokenizer.encode(token_str, add_special_tokens=False)
        return "-".join([str(i) for i in ids_list])
    search_api_ids = convert("#SearchAPI#")
    no_call_api_ids = convert("#NoCallAPI#")
    api_response_ids = convert("#APIResponse#")
    end_ids = convert("#End#")
    inp_ids = "-".join([str(int(i)) for i in input_ids[0][-8:]]) # TODO
    if search_api_ids in inp_ids or api_response_ids in inp_ids \
        or end_ids in inp_ids or no_call_api_ids in inp_ids:
        return True
    return False
transformers.generation.stopping_criteria.MaxTimeCriteria.__call__ = api_call_criteria

def main(
    model_name_or_path,
    lora_path,
    data_path,
    load_in_8bit=True,
    max_new_tokens=128,
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

    for cur_cnt, item in enumerate(tqdm(devset)):
        query = item["conversations"][0]["value"]
        gt_label = item["conversations"][1]["value"]

        if "#Begin# Don't need" in gt_label:
            total_no_search += 1
        elif "#CallAPI#" in gt_label:
            total_search_call += 1
            total_search += 1
        elif "#NoCallAPI#" in gt_label:
            total_search_no_call += 1
            total_search += 1
        else:
            continue

        template = ("A chat between a curious user and an artificial intelligence assistant who can use external tools "
        "and APIs to solve the user's question. The assistant gives tools and APIs calling processes or final answer "
        "to the human's question. Human: {} Assistant:")
        prompt = template.format(query)
        inputs = tokenizer(
            prompt,
            padding=True,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
        )

        cur_generate_cnt = 0
        while True:
            cur_generate_cnt += 1
            if cur_generate_cnt >= 3:
                break
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
            decoded_output = tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)[0]

            if "#Begin#" not in decoded_output:
                print("Error Output!")
                break

            if "#End#" in decoded_output:
                if "#Begin# Don't need" in gt_label and "#Begin# Don't need" in decoded_output:
                    num_no_search += 1
                else:
                    print("Error Output!")
                break

            if "#NoCallAPI#" in decoded_output:
                if "#Begin# Need to" in gt_label:
                    num_search += 1
                else:
                    print("Error Output!")
                    break
                
                if "#NoCallAPI#" not in gt_label:
                    print("Error Output!")
                else:
                    num_search_no_call += 1
                break
            
            if "#CallAPI#" in decoded_output:
                if "#Begin# Need to" in gt_label:
                    num_search += 1
                else:
                    print("Error Output!")
                    break

                if "#NoCallAPI#" in gt_label:
                    print("Error Output!")
                elif "#APIResponse#" not in decoded_output:
                    print("Error Output!")
                else:
                    try:
                        call_api = decoded_output.split("#CallAPI# ")[-1].split(" #APIResponse#")[0]
                        idx = call_api.find("(")
                        gt_call = gt_label.split("#CallAPI#")[-1].split("#APIResponse#")[0]
                        if call_api[:idx] in gt_call:
                            num_search_call += 1
                        else:
                            print("Error Output!")
                    except:
                        print("Error Output!")
                break

            if "#SearchAPI#" in decoded_output:
                if "#SearchAPI#" not in gt_label:
                    print("Error Output!")
                    break
                if "#CallAPI#" in gt_label:
                    inputs = prompt + " " + gt_label.split("#CallAPI#")[0] + "#"
                elif "#NoCallAPI#" in gt_label:
                    inputs = prompt + " " + gt_label.split("#NoCallAPI#")[0] + "#"
                inputs = tokenizer(
                    inputs,
                    padding=True,
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

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
