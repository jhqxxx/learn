'''
Author: jhq
Date: 2025-04-05 17:40:56
LastEditTime: 2025-04-05 20:26:25
Description: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DeepSeekV2MLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
    
    def forward(self, x):
        x = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return x

class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_exports = config.n_routed_exports
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        
        self.inference_norm = self.norm_topk_prob and (self.top_k > 1)
        self.use_group_limited = (self.topk_method == "group_limited_greedy")
        
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.n_routed_exports, self.gating_dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    @torch.inference_mode()
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, h)
        logits = F.linear(hidden_states, self.weight)
        scores = logits.softmax(dim=-1)
        
        # ????没看懂
        if self.use_group_limited:
            group_scores = scores.view(bsz*seq_len, self.n_group, -1).max(dim=-1).values
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
            scores_mask = group_mask.unsqueeze(-1).expand(-1, -1, self.n_routed_exports // self.n_group).reshape(bsz*seq_len, -1)
            scores = scores.masked_fill(~scores_mask.bool(), 0.0)
        
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        if self.inference_norm:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        else:
            topk_weight = topk_weight * self.routed_scaling_factor
        
        return topk_idx, topk_weight, None
            