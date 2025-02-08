import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 1. patch embedding
def image2patch_naive(image, patch_size, weight):
    '''
    image: shape(bs, channel, h, w)
    patch_size: int
    weight: shape(patch_depth, model_dim_c)
    return: shape(bs, num_patch, model_dim_c)
    '''
    patch = F.unfold(image, kernel_size=(patch_size, patch_size), 
                    stride=(patch_size, patch_size)).transpose(-1, -2) # bs, num_patch, patch_depth
    patch_embedding = patch @ weight
    return patch_embedding

def image2patch_conv(image, kernel, stride):
    '''
    image: shape(bs, channel, h, w)
    kernel: shape(out_channel, in_channe/groups, patch_size, patch_size)
    return: shape(bs, num_patch, model_dim_c)
    '''
    conv_out = F.conv2d(image, kernel, stride=stride)
    bs, oc, oh, ow = conv_out.shape
    patch_embedding = conv_out.reshape((bs, oc, oh*ow)).transpose(-1, -2)  # bs,num_patch, model_dim_c
    return patch_embedding


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, model_dim, num_head) -> None:
        super(MultiHeadSelfAttention, self).__init__()
        self.num_head = num_head
        self.proj_linear_layer = nn.Linear(model_dim, 3*model_dim)
        self.final_linear_layer = nn.Linear(model_dim, model_dim)
    
    def forward(self, input, additive_mask=None):
        bs, seqlen, model_dim = input.shape
        num_head = self.num_head
        head_dim = model_dim // num_head
        proj_output = self.proj_linear_layer(input)  # [bs, seqlen, 3*model_dim]
        proj_output = proj_output.reshape(bs, seqlen, 3*num_head, head_dim).transpose(1, 2)
        q, k, v = proj_output.chunk(3, dim=1)
        q = q.reshape(bs*num_head, seqlen, head_dim)
        k = k.reshape(bs*num_head, seqlen, head_dim)
        v = v.reshape(bs*num_head, seqlen, head_dim)
        if additive_mask is None:
            attn_prob = F.softmax(torch.bmm(q, k.transpose(-2, -1))/math.sqrt(head_dim), dim=-1)
        else:
            additive_mask = additive_mask.tile((num_head, 1, 1))
            attn_prob = F.softmax(torch.bmm(q, k.transpose(-2, -1))/math.sqrt(head_dim)+additive_mask, dim=-1)
        output = torch.bmm(attn_prob, v)  # (bs*num_head, seqlen, head_dim)
        output = output.reshape(bs, num_head, seqlen, head_dim).transpose(1, 2)
        output = output.reshape(bs, seqlen, model_dim)
        
        output = self.final_linear_layer(output)
        return attn_prob, output


def window_multi_head_self_attention(patch_embedding, mhsa, window_size=4, num_head=2):
    num_patch_in_window = window_size * window_size
    bs, num_patch, patch_depth = patch_embedding.shape
    image_height = image_width = int(math.sqrt(num_patch))

    patch_embedding = patch_embedding.transpose(-1, -2)
    patch = patch_embedding.reshape(bs, patch_depth, image_height, image_width)
    window = F.unfold(patch, kernel_size=(window_size, window_size),
                        stride=(window_size, window_size)).transpose(-1, -2)  # (bs, num_window, window_depth)
    bs, num_window, patch_depth_in_window = window.shape
    window = window.reshape(bs*num_window, patch_depth, num_patch_in_window).transpose(-1, -2)
    attn_prob, output = mhsa(window)  # (bs*num_window, num_patch_in_window, patch_depth)
    output = output.reshape(bs, num_window, num_patch_in_window, patch_depth)
    return output


def window2image(msa_output):
    bs, num_window, num_patch_in_window, patch_depth = msa_output.shape
    window_size = int(math.sqrt(num_patch_in_window))
    image_height = int(math.sqrt(num_window)) * window_size
    image_width = image_height

    msa_output = msa_output.reshape(bs, int(math.sqrt(num_window)),
                                        int(math.sqrt(num_window)),
                                        window_size,
                                        window_size,
                                        patch_depth)
    msa_output = msa_output.transpose(2, 3)
    image = msa_output.reshape(bs, image_height*image_width, patch_depth)
    image = image.transpose(-1, -2).reshape(bs, patch_depth, image_height, image_width)
    return image


def build_mask_for_shifted_wmsa(batch_size, image_height, image_width, window_size):
    index_matrix = torch.zeros(image_height, image_width)
    for i in range(image_height):
        for j in range(image_width):
            row_times = (i+window_size//2) // window_size
            col_times = (j+window_size//2) // window_size
            index_matrix[i, j] = row_times*(image_height//window_size) + col_times + i
    rolled_index_matrix = torch.roll(index_matrix, shifts=(-window_size//2, -window_size//2), dims=(0, 1))
    rolled_index_matrix = rolled_index_matrix.unsqueeze(0).unsqueeze(0)  # [bs, ch, h, w]

    c = F.unfold(rolled_index_matrix, kernel_size=(window_size, window_size),
                stride=(window_size, window_size)).transpose(-1, -2)
    c = c.tile(batch_size, 1, 1)    # (bs, num_window, num_patch_in_window)
    bs, num_window, num_patch_in_window = c.shape
    c1 = c.unsqueeze(-1)  
    c2 = (c1 - c1.transpose(-1, -2)) == 0
    valid_matrix = c2.to(torch.float32)
    additive_mask = (1-valid_matrix)*(-1e9)
    additive_mask = additive_mask.reshape(bs*num_window, num_patch_in_window, num_patch_in_window)
    return additive_mask

def shift_window(w_msa_output, window_size, shift_size, generate_mask=False):
    bs, num_window, num_patch_in_window, patch_depth = w_msa_output.shape
    w_msa_output = window2image(w_msa_output)
    bs, patch_depth, image_height, image_width = w_msa_output.shape
    rolled_w_mas_output = torch.roll(w_msa_output, shifts=(shift_size, shift_size), dims=(2, 3))
    shifted_w_msa_input = rolled_w_mas_output.reshape(bs, patch_depth,
                                                    int(math.sqrt(num_window)),
                                                    window_size,
                                                    int(math.sqrt(num_window)),
                                                    window_size)
    shifted_w_msa_input = shifted_w_msa_input.transpose(3, 4)
    shifted_w_msa_input = shifted_w_msa_input.reshape(bs, patch_depth, num_window*num_patch_in_window)
    shifted_w_msa_input = shifted_w_msa_input.transpose(-1, -2)
    shift_window = shifted_w_msa_input.reshape(bs, num_window, num_patch_in_window, patch_depth)

    if generate_mask:
        additive_mask = build_mask_for_shifted_wmsa(bs, image_height, image_width, window_size)
    else:
        additive_mask = None
    
    return shift_window, additive_mask

def shift_window_multi_head_self_attention(w_msa_output, mhsa, window_size=4, num_head=2):
    bs, num_window, num_patch_in_window, patch_depth = w_msa_output.shape
    shifted_w_msa_input, additive_mask = shift_window(w_msa_output, window_size, 
                                                    shift_size=window_size//2, generate_mask=True)
    shifted_w_msa_input = shifted_w_msa_input.reshape(bs*num_window, num_patch_in_window, patch_depth)
    attn_prob, output = mhsa(shifted_w_msa_input, additive_mask=additive_mask)
    output = output.reshape(bs, num_window, num_patch_in_window, patch_depth)
    output, _ = shift_window(output, window_size, shift_size=window_size//2, generate_mask=False)
    return output


class PatchMerging(nn.Module):
    def __init__(self, model_dim, merge_size, output_depth_scale=0.5) -> None:
        super(PatchMerging, self).__init__()
        self.merge_size = merge_size
        self.proj_layer = nn.Linear(model_dim*merge_size*merge_size,
                            int(model_dim*merge_size*merge_size*output_depth_scale))
    
    def forward(self, input):
        bs, num_window, num_patch_in_window, patch_depth = input.shape
        window_size = int(math.sqrt(num_patch_in_window))
        iuput = window2image(input)
        merged_window = F.unfold(input, kernel_size=(self.merge_size, self.merge_size),
                                stride=(self.merge_size, self.merge_size)).transpose(-1, -2)
        merged_window = self.proj_layer(merged_window)
        return merged_window


class SwinTransformerBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()