#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Training arguments.

@Time    :   2024/03/27
@Author  :   Ma (Ma787639046@outlook.com)
'''

from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict
from transformers.training_args import SchedulerType

from ..finetune.arguments import ModelArguments, DataArguments, RetrieverTrainingArguments

@dataclass
class DistributationRobustOptimizationModelArguments(ModelArguments):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    
    # *** Group DRO / iDRO ***
    dro_type: Optional[str] = field(
        default='DROModelv1',
        metadata={
            "help": "Choose among ['DROModelv1', 'DROModelv2']."
        },
    )

    ref_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )
    group_loss_minus_ref_loss: bool = field(default=False, metadata={"help": "Compute `lm_loss - ref_loss`. This is the orignal implemention of Group DRO loss."})
    normalize_group_loss_scale_with_ref_loss: bool = field(default=False, metadata={"help": "Compute excess group loss with `lm_loss / ref_loss`, rather than pure `lm_loss`. This helps to get the right grad scale for each group."})
    p_norm_of_ref_loss: int = field(default=1, metadata={"help": "Scale ref_loss with p-norm. This helps to keep the loss scale of `lm_loss / ref_loss` to be consistent with `lm_loss`"})

    reweight_eta: Optional[float] = field(
        default=0.25,
        metadata={
            "help": "η is the learning rate (reweight_eta) for group weights."
        },
    )
    # eps: Optional[float] = field(
    #     default=1e-4,
    #     metadata={
    #         "help": "eps for smoothing factor of updating train domain weights."
    #     },
    # )

    ema: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "ema for weight and loss smoothing for DRO."
        },
    )
    weight_ema: Optional[bool] = field(
        default=False,
        metadata={
            "help": "whether use ema for weight smoothing."
        },
    )
    normalize_weights_on_every_update: bool = field(default=False, metadata={"help": "Whether to use softmax to normalize the train_domain_weights on every update steps."})

    dro_only_hn: bool = field(default=False, metadata={"help": "Only use hard negatives for loss computation with DRO Optimization."})

    dro_save_top_percent_uniform_weights: Optional[List[int]] = field(
        default_factory=lambda: [20, 30, 40, 50, 60, 70, 80, 90], metadata={"help": "This is a Dataset Selection strategy for DRO Optimization. We save the Top k\% domains by uniforming their weights. Thus we `select` the"
        "Top ratio domains."
        })

    # *** Only used by DRO Optimizer / Scheduler ***
    dro_optimizer: str = field(default="sgd", metadata={"help": "Optimizer used for DROModelv2."})
    dro_weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for optimizers if we apply some."})
    dro_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW/RMSProp optimizer."})

    dro_rmsprop_alpha: float = field(default=0.99, metadata={"help": "Alpha for RMSProp optimizer, which is a smoothing constant (default: 0.99)"})

    dro_adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    dro_adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    
    dro_apply_grad_norm: bool = field(default=False, metadata={"help": "Whether to apply gradient normalization."})
    dro_apply_grad_clip: bool = field(default=False, metadata={"help": "Whether to apply gradient clipping."})
    dro_max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    dro_lr_scheduler_type: Union[SchedulerType, str] = field(
        default="constant",
        metadata={"help": "The scheduler type to use. Default is constant"},
    )
    dro_warmup_ratio: float = field(
        default=0.0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    dro_warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    dro_min_lr_ratio: float = field(default=0, metadata={"help": "Min learning rate ratio for cosine/linear scheduler decay."})

    # Domain config related arguments (Init by training scripts)
    epoch: Optional[Dict[str, float]] = field(default=None, metadata={"help": "Dict of dataset name to number of epoch."})
    domain_weights: Optional[Dict[str, float]] = field(default=None, metadata={"help": "Dict of dataset name to weights."})
    n_groups: Optional[int] = field(default=None, metadata={"help": "Number of groups."})
    domain_to_idx: Optional[Dict[str, int]] = field(default=None, metadata={"help": "Dict of dataset name to its serial number."})
    domain_size: Optional[Dict[str, int]] = field(default=None, metadata={"help": "Dict of dataset name to its size."})


@dataclass
class DistributationRobustOptimizationRetrieverTrainingArguments(RetrieverTrainingArguments):
    loss_reduction: str = field(
        default='none', metadata={"help": "Loss reduction of CrossEntropy Loss. Choose among `mean`, `none`."}
    )
