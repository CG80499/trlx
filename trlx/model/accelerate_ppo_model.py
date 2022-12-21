from time import time
from typing import Tuple
import uuid, os, json

import torch
from torchtyping import TensorType

from trlx.data.configs import TRLConfig
from trlx.data.ppo_types import PPORLBatch
from trlx.model import register_model
from trlx.model.accelerate_base_model import AccelerateRLModel
from trlx.model.nn.ppo_models import (
    AdaptiveKLController,
    FixedKLController,
    CausalLMHydraWithValueHead,
    T5HydraWithValueHead,
)
from trlx.pipeline.ppo_pipeline import PPORolloutStorage
from trlx.utils.modeling import logprobs_from_logits
import torch.nn.functional as F
import wandb

import ray


@register_model
class AcceleratePPOModel(AccelerateRLModel):
    def __init__(self, config):
        super().__init__(config)

        if config.train.rollout_logging_dir is not None:
            self.log_rollouts = True
            self.setup_rollout_logging(config)
        else:
            self.log_rollouts = False

        self.store = PPORolloutStorage(self.tokenizer.pad_token_id)

        rollout_loader = self.store.create_loader(
            self.config.train.batch_size, shuffle=True
        )

        self.model, self.opt, self.scheduler, rollout_loader = self.accelerator.prepare(
            self.model, self.opt, self.scheduler, rollout_loader
        )

        self.store.clear_history()
        if config.method.target is not None:
            self.kl_ctl = AdaptiveKLController(
                config.method.init_kl_coef, config.method.target, config.method.horizon
            )
        else:
            self.kl_ctl = FixedKLController(config.method.init_kl_coef)

        self.generate_kwargs = dict(
            config.method.gen_kwargs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

    def get_arch(self, config: TRLConfig):
        return CausalLMHydraWithValueHead(
            config.model.model_path, config.model.num_layers_unfrozen
        )

    def get_model_inputs(
        self,
        query_tensors: TensorType["batch_size", "query_size"],
        response_tensors: TensorType["batch_size", "response_size"],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = torch.cat((query_tensors, response_tensors), dim=1)[
            :, -self.max_length :
        ]
        attention_mask = (
            tokens.not_equal(self.tokenizer.pad_token_id).long().to(tokens.device)
        )
        # For a proper positional encoding in case of left padding
        position_ids = attention_mask.cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask.eq(0), 0)
        return tokens, attention_mask, position_ids

    def loss(self, batch: PPORLBatch):
        # Move `batch` data to `accelerator` device
        query_tensors = batch.query_tensors.to(self.accelerator.device)
        response_tensors = batch.response_tensors.to(self.accelerator.device)
        old_logprobs = batch.logprobs.to(self.accelerator.device)
        old_values = batch.values.to(self.accelerator.device)
        old_rewards = batch.rewards.to(self.accelerator.device)

        response_length = old_rewards.shape[1]

        advantages, returns = self.config.method.get_advantages_and_returns(
            old_values, old_rewards, response_length
        )

        tokens, attention_mask, position_ids = self.get_model_inputs(
            query_tensors, response_tensors
        )

        logits, *_, values_pred = self.model(
            tokens, attention_mask=attention_mask, position_ids=position_ids
        )
        values_pred = values_pred[:, :-1]
        logprobs = logprobs_from_logits(logits[:, :-1, :], tokens[:, 1:])
        attention_mask = attention_mask[:, :-1]

        # Only the response part of the values/logprobs is needed
        start = query_tensors.shape[1] - 1
        end = start + response_length
        logprobs, values_pred, mask = (
            logprobs[:, start:end],
            values_pred[:, start:end],
            attention_mask[:, start:end],
        )

        loss, stats = self.config.method.loss(
            logprobs=logprobs,
            values=values_pred,
            old_logprobs=old_logprobs,
            old_values=old_values,
            advantages=advantages,
            returns=returns,
            mask=mask,
        )
        self.approx_kl = stats["policy/approx_kl"]  # Update kl controller stats
        return loss, stats

    def setup_rollout_logging(self, config):
        # Make rollout logging dir for this run and store config
        exists = os.path.exists(config.train.rollout_logging_dir)
        isdir = os.path.isdir(config.train.rollout_logging_dir)
        assert exists and isdir

        self.run_id = f"run-{uuid.uuid4()}"
        self.rollout_logging_dir = os.path.join(
            config.train.rollout_logging_dir, self.run_id
        )
        os.mkdir(self.rollout_logging_dir)

        with open(os.path.join(self.rollout_logging_dir, "config.json"), "w") as f:
            f.write(json.dumps(config.to_dict(), indent=2))

    def post_epoch_callback(self):
        if self.log_rollouts:
            self.store.export_history(location=self.rollout_logging_dir)
        self.store.clear_history()
        self.orch.make_experience(
            self.config.method.num_rollouts, self.iter_count
        )  # Collect more rollouts for training

    def post_backward_callback(self):
        self.kl_ctl.update(self.approx_kl, n_steps=self.config.train.batch_size)

    def prepare_learning(self):
        eval_dataloader = self.eval_pipeline.create_loader(self.config.train.batch_size)

        train_dataloader = self.store.create_loader(
            self.config.train.batch_size, shuffle=True
        )

        self.train_dataloader, self.eval_dataloader = self.accelerator.prepare(
            train_dataloader, eval_dataloader
        )

        self.n_updates_per_batch = self.config.method.ppo_epochs
        self.total_steps = (
            self.config.train.epochs
            * self.n_updates_per_batch
            * len(self.train_dataloader)
        )
        self.total_steps = min(self.total_steps, self.config.train.total_steps)

@register_model
class T5AcceleratePPOModel(AcceleratePPOModel):

    def __init__(self, config: TRLConfig):
        super().__init__(config)

        self.tokenizer.padding_side = "right" # Left padding not supported

    def get_arch(self, config: TRLConfig):
        return T5HydraWithValueHead(
            config.model.model_path,
        )

    def get_model_inputs(
        self,
        query_tensors: TensorType["batch_size", "query_size"],
        response_tensors: TensorType["batch_size", "response_size"],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = query_tensors[:, :self.max_length]
        attention_mask = (
            input_ids.not_equal(self.tokenizer.pad_token_id).long().to(input_ids.device)
        )
        decoder_input_ids = response_tensors[:, : self.max_length]

        decoder_attention_mask = (
            decoder_input_ids.not_equal(self.tokenizer.pad_token_id)
            .long()
            .to(decoder_input_ids.device)
        )

        return input_ids, attention_mask, decoder_input_ids, decoder_attention_mask

    def loss(self, batch: PPORLBatch):
        # Move `batch` data to `accelerator` device
        query_tensors = batch.query_tensors.to(self.accelerator.device)
        response_tensors = batch.response_tensors.to(self.accelerator.device)
        old_logprobs = batch.logprobs.to(self.accelerator.device)
        old_values = batch.values.to(self.accelerator.device)
        old_rewards = batch.rewards.to(self.accelerator.device)

        response_length = old_rewards.shape[1]

        advantages, returns = self.config.method.get_advantages_and_returns(
            old_values, old_rewards, response_length
        )

        input_ids, attention_mask, decoder_input_ids, decoder_attention_mask = self.get_model_inputs(
            query_tensors, response_tensors
        )

        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask
        )
        logits, values_pred = outputs.logits, outputs.value
        values_pred = values_pred[:, :-1]
        logprobs = logprobs_from_logits(logits[:, :-1, :], decoder_input_ids[:, 1:]) # decoder_input_ids doesn't include the start token so we don't need to shift it
        attention_mask = attention_mask[:, :-1]

        loss, stats = self.config.method.loss(
            logprobs=logprobs,
            values=values_pred,
            old_logprobs=old_logprobs,
            old_values=old_values,
            advantages=advantages,
            returns=returns,
            mask=decoder_attention_mask[:, 1:],
        )
        self.approx_kl = stats["policy/approx_kl"]  # Update kl controller stats
        return loss, stats
    
    def evaluate(self):
        """Samples model on `eval_prompts`, logs stats with `reward_fn` or `metric_fn` if provided"""
        stats = {}
        all_samples = []
        prompts_sizes = []
        generate_time = time()
        prompts_list, responses = [], []
        for prompts in self.eval_dataloader:
            if isinstance(prompts, torch.Tensor):
                attention_mask = (
                    prompts.not_equal(self.tokenizer.pad_token_id)
                    .long()
                    .to(prompts.device)
                )
                samples = self.generate(prompts, attention_mask=attention_mask)
                prompts_list.extend(
                    self.tokenizer.batch_decode(
                        prompts, skip_special_tokens=True
                    )
                )
            else:
                samples = self.generate(**prompts)
                prompts_list.extend(
                    self.tokenizer.batch_decode(
                        prompts["input_ids"], skip_special_tokens=True
                    )
                )

            if isinstance(samples, tuple):
                samples, *_ = samples

            responses.extend(
                self.tokenizer.batch_decode(
                    samples, skip_special_tokens=True
                )
            )


        stats["time/generate"] = time() - generate_time

        if self.accelerator.is_main_process:

            columns_data = [prompts_list, responses]

            columns = ["prompt", "response"]

            str_samples = [f"{prompt} {response}" for prompt, response in zip(prompts_list, responses)]

            # in online setting, compute the reward for validation
            if self.reward_fn:
                rewards = torch.tensor(self.reward_fn(str_samples), dtype=torch.float)
                mean_reward = rewards.mean()
                columns.append("reward")
                columns_data.append(rewards)
                stats["reward/mean"] = mean_reward
                print(f"Mean rewards: {mean_reward}")
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.save()
                    print("=== Saved ===")

            # additionally log any other metrics
            if self.metric_fn:
                metric_time = time()
                metrics = self.metric_fn(str_samples)
                stats["time/metric"] = time() - metric_time

                mean_metrics = {
                    f"metrics/{k}": torch.as_tensor(xs).mean(-1)
                    for k, xs in metrics.items()
                }

                stats.update(mean_metrics)

                for metric, values in metrics.items():
                    columns.append(metric)
                    columns_data.append(values)

            rows = list(zip(*columns_data))
            if not ray.is_initialized():
                stats["samples"] = wandb.Table(columns=columns, rows=rows)

        return stats