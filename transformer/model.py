'''
Author: jhq
Date: 2025-04-03 13:28:23
LastEditTime: 2025-04-03 21:25:23
Description: 
'''
import torch
from torch import nn

class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionEmbedding, self).__init__()
        
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False        
        pos = torch.arange(0, max_len, device=device)
        print(pos.shape)
        pos = pos.float().unsqueeze(dim=1)
        print(pos.shape)
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        print(_2i.shape)
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model))) 
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        print("encoding.shape: ",self.encoding.shape)
        
    def forward(self, x):
        seq_len = x.shape[1]
        return self.encoding[:seq_len, :]

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)
    

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, drop_prob, device):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionEmbedding(d_model, max_len, device)
        self.dropout = nn.Dropout(drop_prob)
    
    def forward(self, x):
        tok_emb = self.tok_emb(x)
        print("tok_emb shape", tok_emb.shape)
        pos_emb = self.pos_emb(x)
        print(pos_emb[:, :10])
        print("pos_emb shape", pos_emb.shape)
        return self.dropout(tok_emb + pos_emb)

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, Q, K, V, mask=None):
        K_T = K.transpose(-1, -2) # (B, seq_len, d_model) -> (B, d_model, seq_len)
        d_k = Q.shape[-1]
        scores = torch.matmul(Q, K_T) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32)) # (B, seq_len, seq_len)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weight = self.softmax(scores)
        output = torch.matmul(attn_weight, V)  # (B, seq_len, seq_len) * (B, seq_len, d_model) = (B, seq_len, d_model)
        return output, attn_weight

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_head = num_heads
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)
        output, attn_weight = self.attention(q, k, v, mask)
        concat_tensor = self.concat(output)
        mha_output = self.fc(concat_tensor)
        return mha_output
    
    def split(self, tensor):
        batch_size, seq_len, d_model = tensor.shape
        d_tensor = d_model // self.n_head
        split_tensor = tensor.view(batch_size, seq_len, self.n_head, d_tensor).transpose(1, 2)
        return split_tensor
    
    def concat(self, tensor):
        batch_size, n_head, seq_len, d_tensor = tensor.shape
        d_model = d_tensor * n_head
        concat_tensor = tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return concat_tensor

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        print(mean.shape)
        var = x.var(dim=-1, keepdim=True)
        print(var.shape)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out

class PostionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, drop_prob=0.1):
        super(PostionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(drop_prob)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)        
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, drop_prob=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        self.dropout2 = nn.Dropout(drop_prob)
        self.ffn = PostionWiseFeedForward(d_model, d_ff)
    
    def forward(self, x, mask=None):
        mha_output = self.mha(x, x, x, mask)
        output = x + self.dropout1(mha_output)
        output = self.norm1(output)
        x = self.ffn(output)
        output = output + self.dropout1(x)
        output = self.norm2(output)
        return output

class Encoder(nn.Module):
    def __init__(self, voc_size, seq_len, d_model, d_ff, n_head, n_layers, drop_prob=0.1, device='cpu'):
        super(Encoder, self).__init__()
        self.emb = TransformerEmbedding(voc_size, seq_len, d_model, drop_prob, device)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_head, d_ff, drop_prob) for _ in range(n_layers)])
    
    def forward(self, x, mask=None):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, drop_prob=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)
        
        self.ffn = PostionWiseFeedForward(d_model, d_ff)
        self.dropout3 = nn.Dropout(drop_prob)
        self.norm3 = LayerNorm(d_model)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        mha_output1 = self.mha1(x, x, x, tgt_mask)
        output = x + self.dropout1(mha_output1)
        output = self.norm1(output)
        if enc_output is not None:
            mha_output2 = self.mha2(output, enc_output, enc_output, src_mask)
            output = output + self.dropout2(mha_output2)
            output = self.norm2(output)
        ffn_output = self.ffn(output)
        output = output + self.dropout3(ffn_output)
        output = self.norm3(output)
        return output
        

class Decoder(nn.Module):
    def __init__(self, voc_size, seq_len, d_model, d_ff, n_head, n_layers, drop_prob=0.1, device='cpu'):
        super(Decoder, self).__init__()
        self.emb = TransformerEmbedding(voc_size, seq_len, d_model, drop_prob, device)
        self.layers = nn.ModuleList(DecoderLayer(d_model, n_head, d_ff, drop_prob) for _ in range(n_layers))
        self.linear = nn.Linear(d_model, voc_size)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        x = self.linear(x)
        x = self.softmax(x)
        return x

class Transformer(nn.Module):
    def __init__(self, src_pad_idx, tgt_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, seq_len, d_model, d_ff, n_head, n_layers, drop_prob=0.1, device='cpu'):
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        self.encoder = Encoder(enc_voc_size, seq_len, d_model, d_ff, n_head, n_layers, drop_prob, device)
        self.decoder = Decoder(dec_voc_size, seq_len, d_model, d_ff, n_head, n_layers, drop_prob, device)
    
    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        src_tgt_mask = self.make_pad_mask(trg, trg, self.tgt_pad_idx, self.src_pad_idx)
        trg_mask = self.make_pad_mask(trg, trg, self.tgt_pad_idx, self.tgt_pad_idx) * self.make_no_peak_mask(trg, trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_tgt_mask)
        return output
    
    # mask没看懂，
    # 而且encoder部分是不需要mask的
    # decoder也只有第一个attention需要mask
    def make_pad_mask(self, q, k, q_pad_idx, k_pad_idx):
        len_q, len_k = q.shape[1], k.shape[1]
        k = k.ne(k_pad_idx).unsqueeze(1).unsqueeze(2)
        k = k.repeat(1, 1, len_q, 1)
        q = q.ne(q_pad_idx).unsqueeze(1).unsqueeze(3)
        q = q.repeat(1, 1, 1, len_k)
        mask = k & q
        return mask
    def make_no_peak_mask(self, q, k):
        len_q, len_k = q.shape[1], k.shape[1]
        mask = torch.tril(torch.ones(len_q, len_k).type(torch.BoolTensor).to(self.device))
        return mask
        
# embedding = torch.randn(20, 5, 10)

# layer_norm1 = nn.LayerNorm(10)
# out1 = layer_norm1(embedding)
# print(out1.shape)
# layer_norm2 = LayerNorm(10)
# out2 = layer_norm2(embedding)
# print(out2.shape)
# print(torch.allclose(out1, out2, rtol=0.1, atol=0.01))


# embedding = TransformerEmbedding(vocab_size=100, max_len=50, d_model=512, drop_prob=0.1, device='cpu')

# x = torch.randint(0, 100, (2, 50))
# print("input shape:", x.shape)
# out = embedding(x)
# print("output shape:", out.shape)
# x = torch.randn(1, 50, 512)

# print(x.shape)