#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
DRO Trainer for Contrastive Learning.

@Time    :   2024/04/03 17:38:58
@Author  :   Ma (Ma787639046@outlook.com)
'''

import os
import sys
import math
import json
from itertools import repeat
from collections import UserDict
from typing import Dict, List, Tuple, Optional, Any, Union

import torch
import torch.nn as nn
from torch import nn, Tensor
import transformers
from transformers.trainer_utils import has_length
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.training_args import OptimizerNames
from transformers.optimization import get_scheduler

from .arguments import DistributationRobustOptimizationModelArguments as ModelArguments
from .modeling_group_dro_v2 import DROModelv2, GroupWeights
from ..finetune.trainer import ContrastiveTrainer
from ..scheduler import get_linear_schedule_with_warmup_minlr, get_cosine_schedule_with_warmup_minlr

import logging
logger = logging.getLogger(__name__)

try:
    from grad_cache import GradCache
    _grad_cache_available = True
except ModuleNotFoundError:
    _grad_cache_available = False

def rewrite_logs(d):
    # Here we monkey patch the rewriter for Wandb/TFboard/etc.
    # to make logging of α indivisual groups
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    test_prefix = "test_"
    test_prefix_len = len(test_prefix)
    for k, v in d.items():
        if k.startswith(eval_prefix):
            new_d["eval/" + k[eval_prefix_len:]] = v
        elif k.startswith(test_prefix):
            new_d["test/" + k[test_prefix_len:]] = v
        elif ("α" in k) or ("loss_per_group" in k) or ("channel" in k):
            new_d[k] = v
        else:
            new_d["train/" + k] = v
    return new_d

transformers.integrations.integration_utils.rewrite_logs = rewrite_logs

class DROTrainer(ContrastiveTrainer):
    def __init__(self, *args, **kwargs):
        super(DROTrainer, self).__init__(*args, **kwargs)
        self.curr_dro_step = 0
        self.total_steps = self._get_max_steps()
        self.idx_to_domain_name: Dict[int, str] = self.model.idx_to_domain_name

        if isinstance(self.model, DROModelv2):
            # Create GroupWeights, optimizer, scheduler for DROModelv2
            self.model_args: ModelArguments = self.model.model_args
            self.group_weights = GroupWeights(
                n_groups=self.model_args.n_groups, 
                domain_weights=[self.model_args.domain_weights[self.idx_to_domain_name[idx]] for idx in range(self.model_args.n_groups)],
                ema=self.model_args.ema,
                normalize_weights_on_every_update=self.model_args.normalize_weights_on_every_update,
            ).to(self.args.device)
            self.dro_optimizer = self.create_optimizer_dro(self.model_args, self.group_weights)
            self.dro_scheduler = self.create_scheduler_dro(
                args=self.model_args, 
                num_training_steps=self.total_steps, 
                optimizer=self.dro_optimizer,
                num_warmup_steps=self.model_args.dro_warmup_steps if self.model_args.dro_warmup_steps > 0 else math.ceil(self.total_steps * self.model_args.dro_warmup_ratio),
                min_lr_ratio=self.args.min_lr_ratio,
            )

            # Pose shared pointers to the tensors, for saving the weights mean/ema to json
            self.train_domain_weights_mean = self.group_weights.train_domain_weights_mean
            self.train_domain_weights_ema = self.group_weights.train_domain_weights_ema
            self.train_domain_weights_val = self.group_weights.train_domain_weights_val
        else:
            raise NotImplementedError()
    
    @staticmethod
    def create_optimizer_dro(args: ModelArguments, model: GroupWeights):
        # params = (model.train_domain_weights, )
        params = model.parameters()
        if args.dro_optimizer == OptimizerNames.SGD:
            optimizer = torch.optim.SGD(
                params,
                lr=args.reweight_eta,
                momentum=args.ema if args.weight_ema else 0,
                dampening=1-args.ema if args.weight_ema else 0,
                weight_decay=args.dro_weight_decay,
                maximize=True,      # Gradient assend
            )
        elif args.dro_optimizer == OptimizerNames.ADAMW_TORCH:
            optimizer = torch.optim.AdamW(
                params,
                lr=args.reweight_eta,
                betas=(args.dro_adam_beta1, args.dro_adam_beta2),
                eps=args.dro_epsilon,
                weight_decay=args.dro_weight_decay,
                maximize=True,      # Gradient assend
            )
        elif args.dro_optimizer == OptimizerNames.ADAFACTOR:
            optimizer = transformers.optimization.Adafactor(
                params,
                lr=-args.reweight_eta,  # Gradient assend
                scale_parameter=False,
                relative_step=False,    # Adjust lr manually
            )
        elif args.dro_optimizer == OptimizerNames.ADAGRAD:
            optimizer = torch.optim.Adagrad(
                params,
                lr=args.reweight_eta,
                weight_decay=args.dro_weight_decay,
                maximize=True,      # Gradient assend
            )
        elif args.dro_optimizer == OptimizerNames.RMSPROP:
            optimizer = torch.optim.RMSprop(
                params,
                lr=args.reweight_eta,
                alpha=args.dro_rmsprop_alpha,
                eps=args.dro_epsilon,
                weight_decay=args.dro_weight_decay,
                maximize=True,      # Gradient assend
            )
        else:
            raise NotImplementedError(f"Unsppported dro_optimizer type {args.dro_optimizer}")
        
        return optimizer

    @staticmethod
    def create_scheduler_dro(args: ModelArguments, num_training_steps: int, optimizer: torch.optim.Optimizer, num_warmup_steps: int=0, min_lr_ratio: int=0):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if args.dro_lr_scheduler_type == "linear":
            lr_scheduler = get_linear_schedule_with_warmup_minlr(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                min_lr_ratio=min_lr_ratio,
            )
        elif args.dro_lr_scheduler_type == "cosine":
            lr_scheduler = get_cosine_schedule_with_warmup_minlr(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                min_lr_ratio=min_lr_ratio,
            )
        else:
            lr_scheduler = get_scheduler(
                args.dro_lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        return lr_scheduler
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        model_unwrapped = unwrap_model(model)
        if isinstance(model_unwrapped, DROModelv2):
            inputs["group_weights"] = self.group_weights

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)     # Calculate the gradient (mass) for group weights, Produce robust loss for LM training
        
        if isinstance(model_unwrapped, DROModelv2):
            loss = self.training_step_for_dromodelv2(loss=loss, domain_ids=inputs["domain_ids"])
            
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            from apex import amp
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps / self._dist_loss_scale_factor
    
    def training_step_for_dromodelv2(self, loss, domain_ids):
        self.curr_dro_step += 1
        
        if self.curr_dro_step % self.args.gradient_accumulation_steps == 0:
            self.group_weights.train_domain_weights.grad /= self.args.gradient_accumulation_steps

            logs = {"lr_alpha": self.dro_scheduler.get_last_lr()[0]}

            if self.model_args.dro_apply_grad_norm:
                # Do gradient clip if necessary
                grad_norm_alpha = torch.nn.utils.clip_grad_norm_(self.group_weights.parameters(), self.model_args.dro_max_grad_norm)
                logs["grad_norm_alpha"] = round(grad_norm_alpha.item(), 4) if grad_norm_alpha is not None else 0
            elif self.model_args.dro_apply_grad_clip:
                torch.nn.utils.clip_grad_value_(self.group_weights.parameters(), self.model_args.dro_max_grad_norm)
            
            # Update the group weights!
            # Note that group_weights are using float32
            self.dro_optimizer.step()
            self.dro_scheduler.step()
            self.group_weights.update_train_domain_weights()

            for i in range(self.model_args.n_groups):
                logs[f"α.val/{self.idx_to_domain_name[i]}"] = self.group_weights.train_domain_weights_val[i].item()
            
            for i in range(self.model_args.n_groups):
                logs[f"α.mean/{self.idx_to_domain_name[i]}"] = self.group_weights.train_domain_weights_mean[i].item()
            
            for i in range(self.model_args.n_groups):
                logs[f"α.ema/{self.idx_to_domain_name[i]}"] = self.group_weights.train_domain_weights_ema[i].item()
            
            for i in range(self.model_args.n_groups):
                logs[f"logα/{self.idx_to_domain_name[i]}"] = self.group_weights.train_domain_weights[i].item()
            
            for i in range(self.model_args.n_groups):
                logs[f"logα.grad/{self.idx_to_domain_name[i]}"] = self.group_weights.train_domain_weights.grad[i].item()
            
            self._log_custom(logs)

            # Clear grad for group weights
            self.dro_optimizer.zero_grad()

        domain_ids = self._dist_gather_tensor(domain_ids) if self.args.negatives_x_device else domain_ids  # lm_out.loss, domain_ids: [batch_size * world_size]
        weights_indexed = self.group_weights.train_domain_weights_val[domain_ids].detach()        # Index by domain_ids, produce tensor shape [batch_size]
        weights_indexed /= weights_indexed.sum()                      # Normalize
        robust_loss = (loss * weights_indexed).sum()
        return robust_loss
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        A neat compute_loss that supports customized logging

        """
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # Inject Customised logging behavior, only support Dict outputs
        logs: dict = outputs.get("logs", None) if isinstance(outputs, dict) else None
        self._log_custom(logs)

        return (loss, outputs) if return_outputs else loss
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super(DROTrainer, self)._save(output_dir=output_dir, state_dict=state_dict)

        # Save Train Domain Weights, this function is only executed on rank0
        model = unwrap_model(self.model)
        if isinstance(model, DROModelv2):
            model_args: ModelArguments = self.model.model_args
            total_size = sum(model_args.domain_size.values())
            domain_names = list(self.idx_to_domain_name[i] for i in range(model_args.n_groups))
            
            # Current weights
            domain_weights = {self.idx_to_domain_name[i]: self.train_domain_weights_val[i].item() for i in range(model_args.n_groups)}
            curr_weights = {
                "domain_ids": model_args.domain_to_idx,
                "domain_weights": domain_weights,
                "size": model_args.domain_size,
                "epoch": {_name: domain_weights[_name] * total_size / model_args.domain_size[_name] for _name in domain_names},
            }
            self._save_domain_config(curr_weights, output_dir, "curr_weights")
            
            # Mean weights
            domain_weights = {self.idx_to_domain_name[i]: self.train_domain_weights_mean[i].item() for i in range(model_args.n_groups)}
            mean_weights = {
                "domain_ids": model_args.domain_to_idx,
                "domain_weights": domain_weights,
                "size": model_args.domain_size,
                "epoch": {_name: domain_weights[_name] * total_size / model_args.domain_size[_name] for _name in domain_names},
            }
            self._save_domain_config(mean_weights, output_dir, "mean_weights")
            
            # EMA updated weights (ema default to 0.1, thus this value is similar to current weights)
            domain_weights = {self.idx_to_domain_name[i]: self.train_domain_weights_ema[i].item() for i in range(model_args.n_groups)}
            ema_weights = {
                "domain_ids": model_args.domain_to_idx,
                "domain_weights": domain_weights,
                "size": model_args.domain_size,
                "epoch": {_name: domain_weights[_name] * total_size / model_args.domain_size[_name] for _name in domain_names},
            }
            self._save_domain_config(ema_weights, output_dir, "ema_weights")

            # Dataset Selection (Top k%)
            for percent_ratio in model_args.dro_save_top_percent_uniform_weights:
                n_groups: int = model_args.n_groups * percent_ratio // 100
                _, idxs = torch.topk(self.train_domain_weights_val, n_groups, largest=True)
                idxs = idxs.tolist()
                domain_weights = {self.idx_to_domain_name[i]: 1/n_groups for i in idxs}

                domain_names = list(domain_weights.keys())
                total_size = sum(model_args.domain_size[_name] for _name in domain_names)
                topk_weights = {
                    "domain_ids": {_name: i for i, _name in enumerate(domain_names)},
                    "domain_weights": domain_weights,
                    "size": {_name: model_args.domain_size[_name] for _name in domain_names},
                    "epoch": {_name: domain_weights[_name] * total_size / model_args.domain_size[_name] for _name in domain_names},
                }
                self._save_domain_config(topk_weights, output_dir, f"top{percent_ratio}_weights")
    
    def _save_domain_config(self, config: dict, output_dir: str, save_prefix: str):
        # Save weights by dataset
        category_config_path = os.path.join(output_dir, f"{save_prefix}.json")
        with open(category_config_path, "w") as f:
            json.dump(config, f, indent=4)

    def _get_max_steps(self) -> int:
        args = self.args
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        return max_steps

class DROGCTrainer(DROTrainer):
    def __init__(self, *args, **kwargs):
        logger.info('Initializing Gradient Cache Trainer')
        if not _grad_cache_available:
            raise ValueError(
                'Grad Cache package not available. You can obtain it from https://github.com/luyug/GradCache.')
        super(DROGCTrainer, self).__init__(*args, **kwargs)
        
        assert self.accelerator is not None
        self.gc = GradCache(
            models=[self.model, self.model],
            chunk_sizes=[self.args.gc_q_chunk_size, self.args.gc_p_chunk_size],
            loss_fn=self.gc_loss_fn,
            compute_loss_context_manager=self.compute_loss_context_manager,
            accelerator=self.accelerator,
            get_rep_fn=lambda x: x.p_reps if x.p_reps is not None else x.q_reps,
        )

        if self.args.no_sync_except_last and (not self.args._no_sync_in_gradient_accumulation):
            logger.warning(f"No sync for gradient accumulation is not compable with Deepspeed. Setting `no_sync_except_last` to False.")
            self.args.no_sync_except_last = False
    
    def gc_loss_fn(self, *reps, **loss_kwargs):
        outputs = self.model.compute_loss(*reps, **loss_kwargs)
        loss = outputs.loss
        self._log_custom(outputs.logs)   # Inject Customised logging behavior

        model_unwrapped = unwrap_model(self.model)
        if isinstance(model_unwrapped, DROModelv2):
            loss = self.training_step_for_dromodelv2(loss=loss, domain_ids=loss_kwargs["domain_ids"])

        return loss

    def training_step(self, model: DROModelv2, inputs: Dict[str, any]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        queries, passages = {'query': inputs.pop('query')}, {'passage': inputs.pop('passage')}

        model_unwrapped = unwrap_model(model)
        if isinstance(model_unwrapped, DROModelv2):
            inputs["group_weights"] = self.group_weights

        # Reference Model Forward
        # **Note**: 
        # FSDP Full Shard only hooks model.forward(), thus we cannot forward with non-root
        # functions (e.g. model.lm_ref.forward(), model.lm_ref is non-root node, thus FSDP will 
        # not automatically gather lm_ref parameters.)
        # However, we can bypass this by adding switch into root node forward function (model.forward()).
        # And use it to forward a leaf module. In our DROModel, we add the switch `return_lm_ref_reps_for_inference`.
        # Explicitly setting it to True to forward with model.lm_ref.
        if model.lm_ref is not None:
            with torch.no_grad():
                q_reps_ref_list: List[torch.Tensor] = []
                for curr_qry_input in self.gc.split_inputs(queries, chunk_size=self.args.gc_q_chunk_size):
                    curr_q_reps = model(**curr_qry_input, return_lm_ref_reps_for_inference=True).q_reps
                    q_reps_ref_list.append(curr_q_reps)
                q_reps_ref = torch.cat(q_reps_ref_list)

                p_reps_ref_list: List[torch.Tensor] = []
                for curr_psg_input in self.gc.split_inputs(passages, chunk_size=self.args.gc_p_chunk_size):
                    curr_p_reps = model(**curr_psg_input, return_lm_ref_reps_for_inference=True).p_reps
                    p_reps_ref_list.append(curr_p_reps)
                p_reps_ref = torch.cat(p_reps_ref_list)
            
            # Add reference representations to inputs, will be used in model.compute_loss 
            inputs['q_reps_ref'] = q_reps_ref
            inputs['p_reps_ref'] = p_reps_ref

        self.gc.models = [model, model]
        loss = self.gc(queries, passages, no_sync_except_last=self.args.no_sync_except_last, **inputs)

        return loss / self.args.gradient_accumulation_steps / self._dist_loss_scale_factor
