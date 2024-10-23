import logging
from dataclasses import dataclass
from typing import Dict, Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel

from ..finetune.modeling_encoder import EncoderModel, EncoderPooler, EncoderOutput, encode_wrapper_func, compute_sub_batch_size
from .arguments import (
    DistributationRobustOptimizationModelArguments as ModelArguments,
    DataArguments,
    DistributationRobustOptimizationRetrieverTrainingArguments as TrainingArguments
)

logger = logging.getLogger(__name__)

@dataclass
class DROEncoderOutput(EncoderOutput):
    group_loss: torch.Tensor = None

class GroupWeights(nn.Module):
    def __init__(
            self, 
            n_groups: int,                      # Number of groups
            domain_weights: List[int] = None,   # Pass additional values for domain weight initialization
            ema: float = 0.1,                    # EMA factor for tracing the EMA-averaged domain weights
            normalize_weights_on_every_update: bool = False,      # Whether to use softmax to normalize the train_domain_weights on every update steps
        ):
        super().__init__()
        # Current Value of Domain Weights, initialized with domain_weights from data config file (Updated by Group Loss or Excess Group Loss)
        self.register_buffer("train_domain_weights_val", torch.ones(n_groups, dtype=torch.float) / n_groups)
        if domain_weights is not None:
            self.train_domain_weights_val.data[:] = torch.tensor(domain_weights)
        
        # `train_domain_weights` is the main parameter, updated by an optimizer
        # Other parameters are updated by `self.update_train_domain_weights()`
        # log(α^t) = log(α^(t-1)) + η*group_loss (SGD example)
        self.train_domain_weights = nn.Parameter(torch.log(self.train_domain_weights_val.clone()))
        
        self.register_buffer("train_domain_weights_mean", self.train_domain_weights_val.clone())        # Mean of train_domain_weights
        self.register_buffer("train_domain_weights_ema", self.train_domain_weights_val.clone())         # EMA updated train_domain_weights
        self.update_counter = 0
        self.ema = ema
        self.normalize_weights_on_every_update = normalize_weights_on_every_update
    
    def update_train_domain_weights(self):
        # Update the value, mean and EMA. Call this function after every update of `train_domain_weights`
        self.train_domain_weights_val[:] = nn.functional.softmax(self.train_domain_weights, dim=0)
        if self.normalize_weights_on_every_update:
            self.train_domain_weights.data = torch.log(self.train_domain_weights_val.clone())
        self.train_domain_weights_mean[:] = (self.train_domain_weights_mean * self.update_counter + self.train_domain_weights_val) / (self.update_counter + 1)
        self.train_domain_weights_ema[:] = self.train_domain_weights_ema * (1 - self.ema) + self.train_domain_weights_val * self.ema
        self.update_counter += 1


class DROModelv2(EncoderModel):
    """
    DROModelv2 converts the gradient assend problem to a gradient desend optimization,
    thus this model will do:
    1. Calculate the gradient (mass) for group weights
    This model will NOT preserve anything about group weights, you should pass the weights to the model

    You should call this model in following steps:
    1. optimizer.zero_grad() -> Clear grad for group weights
    2. Call this model
    3. Do gradient clip if necessary, then update the group weights with optimizer.step()
    4. lr_schedulder.step() if necessary
    5. Produce robust loss for LM training
    """
    def __init__(self,
                 lm_q: PreTrainedModel,
                 lm_p: PreTrainedModel,
                 model_args: Optional[ModelArguments] = None,
                 train_args: Optional[TrainingArguments] = None,
                 data_args: Optional[DataArguments] = None,
                 pooler_q: Optional[EncoderPooler] = None,
                 pooler_p: Optional[EncoderPooler] = None,
                 lm_ref: EncoderModel = None,        # Reference Model for Group DRO
                 ):
        super().__init__(lm_q=lm_q, lm_p=lm_p, model_args=model_args, train_args=train_args, data_args=data_args, pooler_q=pooler_q, pooler_p=pooler_p)

        self.model_args = model_args

        self.lm_ref = lm_ref        # Whether to use Reference Model for Providing Baseline Loss
        self.idx_to_domain_name: Dict[int, str] = {_domain_name: _idx for _idx, _domain_name in model_args.domain_to_idx.items()}   # idx -> Domain Name
    
    def forward(
            self, 
            query: Dict[str, Tensor] = None, 
            passage: Dict[str, Tensor] = None,
            ce_scores: torch.Tensor = None,
            domain_ids: torch.Tensor = None,
            group_weights: GroupWeights = None,
            return_lm_ref_reps_for_inference: bool = False,     # Only return lm_ref_reps, this is used for inferencing with lm_ref only
            **kwargs,
        ):
        if not return_lm_ref_reps_for_inference:
            if (self.train_args is not None) and self.train_args.dynamic_subbatch_forward:
                # Wrap the encode function for forward with sub-batch
                q_reps = encode_wrapper_func(self.encode_query, compute_sub_batch_size(query["input_ids"].shape[-1]), query)
                p_reps = encode_wrapper_func(self.encode_passage, compute_sub_batch_size(passage["input_ids"].shape[-1]), passage)
            else:         
                q_reps = self.encode_query(query)
                p_reps = self.encode_passage(passage)
        else:
            q_reps, p_reps = None, None
        
        # Add lm_ref forward
        if self.lm_ref is not None:
            self.lm_ref.eval()
            with torch.no_grad():
                q_reps_ref = self.lm_ref.encode_query(query)
                p_reps_ref = self.lm_ref.encode_passage(passage)
        else:
            q_reps_ref, p_reps_ref = None, None

        # for inference
        if q_reps is None or p_reps is None:
            return EncoderOutput(
                q_reps=q_reps if not return_lm_ref_reps_for_inference else q_reps_ref,
                p_reps=p_reps if not return_lm_ref_reps_for_inference else p_reps_ref
            )

        # for training
        if self.training:
            return self.compute_loss(q_reps=q_reps, p_reps=p_reps, group_weights=group_weights, ce_scores=ce_scores, domain_ids=domain_ids, q_reps_ref=q_reps_ref, p_reps_ref=p_reps_ref, **kwargs)
        # for eval
        else:
            q_reps_eval = q_reps.unsqueeze(1)      # B 1 D
            p_reps_eval = p_reps.view(q_reps.shape[0], -1, q_reps.shape[-1]) # B N D
            scores = self.compute_similarity(q_reps_eval, p_reps_eval).squeeze(1)      # B N
            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps,
                scores=scores
            )
    
    def compute_loss_only_hn(
            self, 
            q_reps: torch.Tensor, 
            p_reps: torch.Tensor, 
            **kwargs,
        ):
        loss = 0.
        logs = dict()   # Customized Logs

        if self.train_args.clloss_coef > 0:
            n_psg = p_reps.shape[0] // q_reps.shape[0]

            q_reps = q_reps.unsqueeze(1)
            p_reps = p_reps.view(q_reps.shape[0], n_psg, p_reps.shape[-1])

            scores = self.compute_similarity(q_reps, p_reps) / self.train_args.temperature
            scores = scores.squeeze(1)

            target = torch.zeros(scores.shape[0], device=scores.device, dtype=torch.long) # [0]
            clloss = self.cross_entropy(scores, target) * self.train_args.clloss_coef
            
            loss += clloss
            logs['clloss'] = round(clloss.item(), 4) if self.train_args.loss_reduction == 'mean' else round(clloss.mean().item(), 4)
        
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
            logs=logs,
        )
    
    def compute_loss(
            self, 
            q_reps: torch.Tensor, 
            p_reps: torch.Tensor, 
            group_weights: GroupWeights,
            ce_scores: torch.Tensor = None,
            domain_ids: torch.Tensor = None,
            q_reps_ref: torch.Tensor = None,        # Reference Representations for computing baseline losses
            p_reps_ref: torch.Tensor = None,
            **kwargs,
        ):
        """ Compute Contrastive Loss and/or Distilation Loss
            GradCache separates the forward & contrastive loss computation, thus
            this function is separated from model.forward. And this function is 
            shared with model.forward & GradCache.compute_loss
        """
        if self.model_args.dro_only_hn:
            lm_out = self.compute_loss_only_hn(q_reps=q_reps, p_reps=p_reps, ce_scores=ce_scores, **kwargs)
        else:
            lm_out = super().compute_loss(q_reps=q_reps, p_reps=p_reps, ce_scores=ce_scores, **kwargs)

        loss_ref = None
        if (q_reps_ref is not None) and (p_reps_ref is not None):
            with torch.no_grad():
                if self.model_args.dro_only_hn:
                    loss_ref: torch.Tensor = self.compute_loss_only_hn(q_reps=q_reps_ref, p_reps=p_reps_ref, ce_scores=ce_scores, **kwargs).loss
                else:
                    loss_ref: torch.Tensor = super().compute_loss(q_reps=q_reps_ref, p_reps=p_reps_ref, ce_scores=ce_scores, **kwargs).loss

        return self.compute_group_dro_loss(lm_out=lm_out, domain_ids=domain_ids, group_weights=group_weights, loss_ref=loss_ref)
    
    def compute_group_dro_loss(
            self, 
            lm_out: EncoderOutput, 
            domain_ids: torch.Tensor, 
            group_weights: GroupWeights,        # Group Weights should be preserved, updated outside the model. You should move it to GPU before using
            loss_ref: Optional[torch.Tensor] = None, 
            **kwargs
        ):
        # ** Interleave update Group Weights Gradient & Robust Loss for Model **

        # ** Step1: Update Geadient of Group Weights
        # Note that lm_out.loss, loss_ref, domain_ids are required to be gathered tensors.
        domain_ids = self._dist_gather_tensor(domain_ids)   # lm_out.loss, domain_ids: [batch_size * world_size]
        
        if self.train_args.negatives_x_device:
            # Scale back to the true loss scale, because of all gather
            loss_lm_scaled = lm_out.loss / self.world_size
            loss_ref_scaled = loss_ref / self.world_size if loss_ref is not None else None
        else:
            loss_lm_scaled = self._dist_gather_tensor(lm_out.loss)
            loss_ref_scaled = self._dist_gather_tensor(loss_ref) if loss_ref is not None else None

        with torch.no_grad():
            # Compute per-domain (excess) losses for each domain
            loss_proxy: torch.Tensor = loss_lm_scaled.clone()
            if loss_ref_scaled is not None:
                if self.model_args.group_loss_minus_ref_loss:
                    loss_proxy -= loss_ref_scaled
                
                if self.model_args.normalize_group_loss_scale_with_ref_loss:
                    # Compute excess group loss with `lm_loss / ref_loss`, rather than pure `lm_loss`. 
                    # This helps to get the right grad scale for each group.
                    loss_ref_clamped = loss_ref_scaled.clone()
                    if self.model_args.p_norm_of_ref_loss > 0:
                        loss_ref_clamped = nn.functional.normalize(loss_ref_clamped, p=self.model_args.p_norm_of_ref_loss, dim=0)

                    loss_ref_clamped[loss_ref_clamped < 1e-4] = 1   # Avoid division by zero
                    loss_proxy /= loss_ref_clamped            

            # Ensure the per-domain (excess) losses >= 0.
            loss_proxy = torch.clamp(loss_proxy, min=0.)
            # Mean Reduce the losses by domain_ids
            one_vec = loss_proxy.new_ones(domain_ids.shape[0])
            zero_vec = loss_proxy.new_zeros(self.model_args.n_groups)
            num_of_each_group = zero_vec.scatter_reduce(0, index=domain_ids, src=one_vec, reduce="sum")
            num_of_each_group[num_of_each_group == 0] = 1     # Fill the zero region with 1, to avoid division of zero

            group_loss = zero_vec.scatter_reduce(0, index=domain_ids, src=loss_proxy, reduce="sum") / num_of_each_group
        
        # Update gradient of group weights here:
        if group_weights.train_domain_weights.grad is None:
            group_weights.train_domain_weights.grad = group_loss
        else:
            group_weights.train_domain_weights.grad += group_loss

        with torch.no_grad():
            # Log Group LM Loss
            for i in range(self.model_args.n_groups):
                lm_out.logs[f"logα.grad_ori/{self.idx_to_domain_name[i]}"] = group_loss[i].item()

            lm_loss_per_group = zero_vec.scatter_reduce(0, index=domain_ids, src=loss_lm_scaled, reduce="sum") / num_of_each_group
            for i in range(self.model_args.n_groups):
                lm_out.logs[f"lm_loss_per_group/{self.idx_to_domain_name[i]}"] = lm_loss_per_group[i].item()
            
            if loss_ref_scaled is not None:
                ref_loss_per_group = zero_vec.scatter_reduce(0, index=domain_ids, src=loss_ref_scaled, reduce="sum") / num_of_each_group
                for i in range(self.model_args.n_groups):
                    lm_out.logs[f"ref_loss_per_group/{self.idx_to_domain_name[i]}"] = ref_loss_per_group[i].item()

        return lm_out

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            data_args: DataArguments,
            **hf_kwargs,
    ):
        """ Build EncoderModel for Training """
        model = super().build(model_args=model_args, train_args=train_args, data_args=data_args, **hf_kwargs)
        model.lm_ref = EncoderModel.load(
                        model_name_or_path=model_args.ref_model_name_or_path,
                        model_args=model_args,
                        **hf_kwargs
                    ).eval() if model_args.ref_model_name_or_path is not None else None
        return model
