import torch
import torch.nn as nn
import torch.nn.functional as F

# step1 convert image to embedding vector sequence
def img2emb_naive(img, patch_size, weight):
    # images size: bs*channel*h*w
    patch = F.unfold(img, kernel_size=patch_size, stride=patch_size).transpose(-1, -2)
    patch_embedding = patch @ weight
    return patch_embedding

def img2emb_conv(img, kernel, stride):
    conv_output = F.conv2d(img, kernel, stride=stride) # bs*oc*oh*ow
    bs, oc, oh, ow = conv_output.shape
    patch_embedding = conv_output.reshape((bs, oc, oh*ow)).transpose(-1, -2)
    return patch_embedding

# test code for img2emb
bs, ic, img_h, img_w = 1, 3, 8, 8
patch_size = 4
model_dim = 8
max_num_token = 16
num_classes = 10
label = torch.randint(10, (bs,))

patch_depth = patch_size * patch_size * ic
image = torch.randn(bs, ic, img_h, img_w)
weight = torch.randn(patch_depth, model_dim) # model_dim是输出通道数， patch_depth是卷积核面积乘以通道数
patch_embedding_naive = img2emb_naive(image, patch_size, weight)

kernel = weight.transpose(0, 1).reshape((-1, ic, patch_size, patch_size))    # oc*ic*kh*kw
patch_embedding_conv = img2emb_conv(image, kernel, patch_size)

print(patch_embedding_naive.size())
print(patch_embedding_conv.size())

# step2 prepend CLS token embedding
cls_token_embedding = torch.randn(bs, 1, model_dim, requires_grad=True)
token_embedding = torch.cat([cls_token_embedding, patch_embedding_conv], dim=1)

# step3 add position embedding
position_embedding_table = torch.randn(max_num_token, model_dim, requires_grad=True)
seq_len = token_embedding.shape[1]
position_embedding = torch.tile(position_embedding_table[:seq_len], [token_embedding.shape[0], 1, 1])
token_embedding += position_embedding

# step4  pass embedding to transformer Encoder
encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=8)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
print(token_embedding.shape)
out = transformer_encoder(token_embedding)

# step5 do classification
cls_token_output = out[:, 0, :]
linear_layer = nn.Linear(model_dim, num_classes)
logits = linear_layer(cls_token_output)
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, label)
print(loss)