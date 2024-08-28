import os
import sys
import time
import json
import fire
import openai
import random
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
openai.api_key = "your_api_key"
openai.api_base = "https://xxx"

def llm(prompt, system_prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )
    return response['choices'][0]['message']['content']

def check(query, call, api_function):
    system_prompt = "You are a highly skilled assistant for checking the quality of generation."
    template = """Now, giving the following information:
The API function's format: {api_function}
The user's query: {query}
The user's call: {call}
Please check whether the user's call meets the following conditions:
1. Conform to the call format of the API;
2. The parameters filled in the call appear in query or can be inferred from query;
Please give your answer [YES] or [NO], without additional output.
"""
    prompt = template.format(
        api_function=api_function,
        query=query,
        call=call,
    )
    try_cnt = 0
    max_try_cnt = 5
    while try_cnt < max_try_cnt:
        try:
            output = llm(prompt, system_prompt)
            break
        except:
            time.sleep(5)
        try_cnt += 1
    if try_cnt == max_try_cnt:
        return False
    if "yes" in output.lower():
        return True
    if "no" in output.lower():
        return False


system_prompt = "You are a highly skilled assistant for tool's query and call generation."
template = """Here is a tool for ChatGPT, which can help it solve users' requests better. The tool's name, the description of users, and the description of ChatGPT are as follows:
The name of tool: "%s"
The description of users: "%s"
The description of ChatGPT: "%s"
Please devise an API function (including potential parameters) to call this tool, and give 10 examples where you would use this tool to answer a user's query and you should tell me what users will say and how to call the API function.
Please ensure that the provided examples are distinct from one another. Feel free to employ various sentence styles, such as instructions or requests, and vary the level of detail as needed.
The format of your answer should be like:
API function: %s(args1: str, args2: int, ...)
1. {"query": "xxx", "call": "%s(args1=xxx, args2=yyy, ...)"}
2. {"query": "xxx", "call": "%s(args1=xxx, args2=yyy, ...)"}
Note that
(1) the args filled in the call should appear in query or can be inferred from query;
(2) the generated sample with {} can be successfully loaded by json.loads().
"""

def main(data_path, write_to_path):
    data = pd.read_json(data_path)
    tool_unique = []
    data_filtered = []
    for item in data['items']:
        name = item['manifest']['name_for_model']
        if name not in tool_unique:
            tool_unique.append(name)
            data_filtered.append(item)

    gen_data = []
    max_try_cnt = 5

    for item in tqdm(data_filtered):
        prompt = template % (
            item["manifest"]["name_for_model"],
            item["manifest"]["description_for_human"],
            item["manifest"]["description_for_model"],
            item["manifest"]["name_for_model"],
            item["manifest"]["name_for_model"],
            item["manifest"]["name_for_model"],
        )
        try_cnt = 0
        while try_cnt < max_try_cnt:
            try:
                output = llm(prompt, system_prompt)
                break
            except:
                time.sleep(5)
            try_cnt += 1
        if try_cnt == max_try_cnt:
            print("*** exceed maximum request ***")
            continue
        output = output.split("\n")
        if "API function: " in output[0]:
            api_function = output[0].replace("API function: ", "")
        else:
            print("*** extract fail ***")
            continue
        
        example = []
        for out in output[1:]:
            if "{\"query\": " in out and "\"call\": " in out:
                out = out[out.index("{\"query\": "):]
                try:
                    out = json.loads(out)
                    if check(out["query"], out["call"], api_function):
                        example.append(out)
                    else:
                        print("*** check fail ***")
                        continue
                except:
                    print("*** extract fail ***")
            elif len(out) > 3:
                print("*** generate fail ***")
        
        gen_data.append({
            "api_name": item["manifest"]["name_for_model"],
            "description_for_human": item["manifest"]["description_for_human"],
            "description_for_model": item["manifest"]["description_for_model"],
            "api_function": api_function,
            "example": example,
        })

    json.dump(gen_data, open(write_to_path, "w"), ensure_ascii=False, indent=4)


if __name__ == "__main__":
    fire.Fire(main)
