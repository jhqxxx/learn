'''
Author: jhq
Date: 2025-04-05 13:19:50
LastEditTime: 2025-04-05 14:00:09
Description: 
'''
import torch
def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token_sorted_idx = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, dim=-1, index=next_token_sorted_idx)
    
    return next_token