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

url = 'http://65.108.33.71:5000/api_batched'


# def reward_fn(samples: List[str]) -> List[float]:
#     return requests.post(url, json = {"texts": samples}).json()

def reward_fn(samples: List[str]) -> List[float]:
    return [s.count("the") for s in samples]

def main():

    # with open('/root/trlx/examples/prompts.json', 'r') as f:
    #     prompts = json.load(f)

    prompts = ["The cat went to", "The dog went to", "I took", "I went to", "The women went to"]

    config = TRLConfig.load_yaml("configs/ppo_config_t5.yml")
    
    model = trlx.train(
        "google/flan-t5-small",
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=prompts,
        config=config,
    )

if __name__ == "__main__":
    main()