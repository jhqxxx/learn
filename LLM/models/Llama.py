import torch
import torch.nn as nn
import torch.nn.functional as F

class LlamaRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps
        
    def forward(self, x):
        # rms_x = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.variance_epsilon)
        # x = self.scale * x / rms_x
        # 如果x是float16，则需要先转为float32再计算，然后再转为float16
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = torch.mean(x**2, dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.scale * x.to(input_dtype)

class FFNSwishGLU(nn.Module):           
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.linear2 = nn.Linear(in_dim, hidden_dim, bias=False)
        # self.swish = 
        self.linear3 = nn.Linear(hidden_dim, in_dim, bias=False)
        
    
    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        x = F.silu(x1) * x2
        x = self.linear3(x)
        return x
        

# RoPE
def compute_theta(dim, base=10000, device='cpu'):
    if dim % 2 != 0:
        print("dim must be even")
    i = torch.arange(0, (dim//2), dtype=torch.float32, device=device)
    theta_i = base ** (-2 * i / dim)
    return theta_i

def precompute_freqs_cis(dim, seq_len, theta=10000, device='cpu'):
    theta = compute_theta(dim, base=theta, device=device)
    m = torch.arange(seq_len, device=device)
    m_theta = torch.outer(m, theta)  # m^T * theta,外积 [seq_len, dim//2]
    freqs_cis = torch.polar(torch.ones_like(m_theta), m_theta)  # 构建复数
    # out = abs*cos(angle) + abs*sin(angle)*j, 实部为cos，虚部为sin
    return freqs_cis

def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert ndim >= 2
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), "freqs_cis.shape: {}, x.shape: {}".format(freqs_cis.shape, x.shape)
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape) # [1, seq_len, dim//2]

def apply_rotary_emb(xq, xk, freqs_cis, device='cpu'):
    xq_reshape = xq.reshape(*xq.shape[:-1], -1, 2) # [batch, seq_len, dim] -> [batch, seq_len, dim//2, 2]
    xk_reshape = xk.reshape(*xk.shape[:-1], -1, 2)
    xq_complex = torch.view_as_complex(xq_reshape)  # 复数形式张量 [batch, seq_len, dim//2]
    xk_complex = torch.view_as_complex(xk_reshape)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_complex)
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(3)  # [batch, seq_len, dim//2, 2] -> [batch, seq_len, dim]
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


    