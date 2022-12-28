# Generates positive movie reviews by tuning a pretrained model on IMDB dataset
# with a sentiment reward function

from datasets import load_dataset
from transformers import pipeline
import os
import json

import trlx
import torch
from typing import List

import requests

from trlx.data.configs import TRLConfig

url = 'http://65.108.33.75:5000/rewards'

def reward_fn(samples: List[str]) -> List[float]:
     return requests.post(url, json = {"texts": samples}).json()["rewards"]

PROMPT = """Question: {query}

Relevant paper:
Title: {title}
Abstract: {abstract}

Write a helpful 1-line summary of the paper based on the question.

Helpful summary:"""

with open("/root/fine-tuning-takeaway-models/human_ft_data.json", "r") as f:
    data = json.load(f)

with open("/root/fine-tuning-takeaway-models/human_ft_data_test.json", "r") as f:
    data_test = json.load(f)

prompts = [
    PROMPT.format(query=d["query"], title=d["title"], abstract=d["abstract"][-2200:])
    for d in data
]

eval_prompts = [
    PROMPT.format(query=d["query"], title=d["title"], abstract=d["abstract"][-2200:])
    for d in data_test
]

def main():

    config = TRLConfig.load_yaml("configs/ppo_config_t5.yml")
    
    model = trlx.train(
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=eval_prompts,
        config=config,
    )

if __name__ == "__main__":
    main()