import os
import json
import fire
import random
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
random.seed(42)



def sampling(sampling_mode, api_name, num_candidate, tool_unique=None, tool2neig=None, tool_cluster=None):
    if sampling_mode == "random":
        api_cand = list(np.random.choice(tool_unique, num_candidate, replace=False))
    elif sampling_mode == "intra_class":
        api_cand = list(np.random.choice(tool2neig[api_name], num_candidate, replace=False))
    elif sampling_mode == "inter_class":
        tool_clusters = []
        for tools in tool_cluster:
            if api_name not in tools:
                tool_clusters.append(tools)
        cand_idx = np.random.choice(len(tool_clusters), num_candidate, replace=False)
        cluster_cand = [tool_clusters[idx] for idx in cand_idx]
        api_cand = []
        for cluster in cluster_cand:
            api_cand.append(np.random.choice(cluster, 1)[0])
    else:
        raise ValueError(f"Not support {sampling_mode} sampling!")
    return api_cand

def generate_instance(processed, is_call, seed_cnt, sample, sampling_mode, num_candidate, tool_unique, tool2neig, tool_cluster, tool2info, prompt_call_api, prompt_nocall_api):
    np.random.seed(seed_cnt)
    api_name = sample["api_name"]
    api_cand = sampling(sampling_mode, api_name, num_candidate, tool_unique, tool2neig, tool_cluster)
    if is_call:
        if api_name not in api_cand:
            api_cand[np.random.randint(0, num_candidate)] = api_name
        api_intro_usage = "\n" + "\n".join(str(tool2info[x]) for x in api_cand)
        reason_chain = prompt_call_api.format(
            api_cand=api_cand, 
            api_intro_usage=api_intro_usage,
            call_api=sample["call"],
            api_response="",
        )
    else:
        while api_name in api_cand:
            api_cand = sampling(sampling_mode, api_name, num_candidate, tool_unique, tool2neig, tool_cluster)
        api_intro_usage = "\n" + "\n".join(str(tool2info[x]) for x in api_cand)
        reason_chain = prompt_nocall_api.format(
            api_cand=api_cand, 
            api_intro_usage=api_intro_usage,
        )
    return processed.append({
        "conversations": [
            {
                "from": "human",
                "value": sample["query"],
            },
            {
                "from": "gpt",
                "value": reason_chain,
            }
        ]
    })


def generate(toolset, tool_call_ratio, num_candidate, n_clusters, mix_ratio, sampling_mode, prompt_call_api, prompt_nocall_api):
    samples = []
    tool2info = {}
    for item in toolset:
        tool2info[item["api_name"]] = {
            "api_name": item["api_name"],
            "description": item["description_for_human"],
            "api_function": item["api_function"],
        }
        for example in item["example"]:
            samples.append({
                "api_name": item["api_name"],
                "query": example["query"],
                "call": example["call"],
            })

    tool_unique = list(tool2info.keys())
    random.shuffle(samples)
    call_size = int(len(samples) * tool_call_ratio)
    call_samples, nocall_samples = samples[:call_size], samples[call_size:]

    tool2neig = {}
    tool_cluster = []
    if sampling_mode != "random":
        tool_info = [tool2info[tool]["description"] for tool in tool_unique]
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(tool_info) # num_tool * 384
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(embeddings)
        labels = kmeans.labels_
        for i in range(n_clusters):
            tools = list(np.array(tool_unique)[labels == i])
            tool_cluster.append(tools)
            for tool in tools:
                tool2neig[tool] = tools

    processed = []
    seed_cnt = 0

    if sampling_mode == "random":
        for sample in call_samples:
            processed = generate_instance(processed, True, seed_cnt, sample, sampling_mode, num_candidate, tool_unique, tool2neig, tool_cluster, tool2info, prompt_call_api, prompt_nocall_api)
            seed_cnt += 1

        for sample in nocall_samples:
            processed = generate_instance(processed, False, seed_cnt, sample, sampling_mode, num_candidate, tool_unique, tool2neig, tool_cluster, tool2info, prompt_call_api, prompt_nocall_api)
            seed_cnt += 1

    elif sampling_mode == "mix":
        call_samples_size = len(call_samples)
        call_samples_random = call_samples[:int(call_samples_size * mix_ratio[0])]
        call_samples_in = call_samples[int(call_samples_size * mix_ratio[0]) : int(call_samples_size * (mix_ratio[0] + mix_ratio[1]))]
        call_samples_out = call_samples[int(call_samples_size * (mix_ratio[0] + mix_ratio[1])):]
    
        for mode, cur_call_samples in zip(["random", "intra_class", "inter_class"], [call_samples_random, call_samples_in, call_samples_out]):
            for sample in cur_call_samples:
                processed = generate_instance(processed, True, seed_cnt, sample, mode, num_candidate, tool_unique, tool2neig, tool_cluster, tool2info, prompt_call_api, prompt_nocall_api)
                seed_cnt += 1

        nocall_samples_size = len(nocall_samples)
        nocall_samples_random = nocall_samples[:int(nocall_samples_size * mix_ratio[0])]
        nocall_samples_in = nocall_samples[int(nocall_samples_size * mix_ratio[0]) : int(nocall_samples_size * (mix_ratio[0] + mix_ratio[1]))]
        nocall_samples_out = nocall_samples[int(nocall_samples_size * (mix_ratio[0] + mix_ratio[1])):]

        for mode, cur_nocall_samples in zip(["random", "intra_class", "inter_class"], [nocall_samples_random, nocall_samples_in, nocall_samples_out]):
            for sample in cur_nocall_samples:
                processed = generate_instance(processed, False, seed_cnt, sample, mode, num_candidate, tool_unique, tool2neig, tool_cluster, tool2info, prompt_call_api, prompt_nocall_api)
                seed_cnt += 1

    return processed


def main(
    tool_path,
    general_path,
    save_dir,
    tool_call_ratio=0.6,
    num_candidate=5,
    tool_size=900,
    nosearch_size=2000,
    sampling_mode="mix", # {"random", "intra_class", "inter_class", "mix"}
    mix_ratio=[0.4, 0.2, 0.4], # i.e., [random, intra_class, inter_class]
    n_clusters=30,
):
    prompt_no_api = "#Begin# Don't need to look up the API Pool, I can answer this question with my own knowledge. #End#"
    prompt_call_api = "#Begin# Need to look up the API Pool. #SearchAPI# The API Pool including: {api_cand}, and the introduction and usage of each API is as follows: {api_intro_usage} #CallAPI# {call_api} #APIResponse# {api_response} #End#"
    prompt_nocall_api = "#Begin# Need to look up the API Pool. #SearchAPI# The API Pool including: {api_cand}, and the introduction and usage of each API is as follows: {api_intro_usage} #NoCallAPI# There is no proper API and I need to answer this question with my own knowledge. #End#"

    toolset = json.load(open(tool_path, "r"))
    random.shuffle(toolset)
    gen_data = generate(toolset[:tool_size], tool_call_ratio, num_candidate, n_clusters, mix_ratio, sampling_mode, prompt_call_api, prompt_nocall_api)
    gen_test = generate(toolset[tool_size:], tool_call_ratio, num_candidate, n_clusters, mix_ratio, "random", prompt_call_api, prompt_nocall_api)

    general_data = json.load(open(general_path, "r"))
    random.shuffle(general_data)
    for item in general_data[:nosearch_size]:
        gen_data.append({
            "conversations": [
                {
                    "from": "human",
                    "value": item["conversations"][0]["value"]
                },
                {
                    "from": "gpt",
                    "value": prompt_no_api
                }
            ]
        })
    random.shuffle(gen_data)
    train_size = int(len(gen_data) * 0.9)
    gen_train = gen_data[:train_size]
    gen_valid = gen_data[train_size:]

    os.makedirs(save_dir, exist_ok=True)
    json.dump(gen_train, open(os.path.join(save_dir, "train.json"), "w"), indent=4, ensure_ascii=False)
    json.dump(gen_valid, open(os.path.join(save_dir, "valid.json"), "w"), indent=4, ensure_ascii=False)
    json.dump(gen_test, open(os.path.join(save_dir, "test.json"), "w"), indent=4, ensure_ascii=False)

if __name__ == "__main__":
    fire.Fire(main)
