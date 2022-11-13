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

url = 'http://65.108.33.71:5000/api_batched'


def reward_fn(samples: List[str]) -> List[float]:
    return requests.post(url, json = {"texts": samples}).json()


def main():

    with open('/root/trlx/examples/prompts.json', 'r') as f:
        prompts = json.load(f)

    model = trlx.train(
        "gpt2",
        reward_fn=reward_fn,
        prompts=prompts[:100],
        eval_prompts=prompts[100:],
    )


if __name__ == "__main__":
    main()