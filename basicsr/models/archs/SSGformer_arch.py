import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
try:
    from basicsr.models.archs.arch_util import ResidualBlock
except: pass

#########################################################################
Conv2d = nn.Conv2d
##########################################################################
## Rearrange
def to_2d(x):
    return rearrange(x, 'b c h w -> b (h w c)')

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

##########################################################################
## Layer Norm
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) #* self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) #* self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type="WithBias"):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## SSGformer module
# Dual-scale Gated Feed-Forward Network (DGFF)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv_5 = Conv2d(hidden_features//4, hidden_features//4, kernel_size=5, stride=1, padding=2, groups=hidden_features//4, bias=bias)
        self.dwconv_dilated2_1 = Conv2d(hidden_features//4, hidden_features//4, kernel_size=3, stride=1, padding=2, groups=hidden_features//4, bias=bias, dilation=2)
        self.p_unshuffle = nn.PixelUnshuffle(2)
        self.p_shuffle = nn.PixelShuffle(2)

        self.project_out = Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.p_shuffle(x)

        x1, x2 = x.chunk(2, dim=1)
        x1 = self.dwconv_5(x1)
        x2 = self.dwconv_dilated2_1( x2 )
        x = F.mish( x2 ) * x1
        x = self.p_unshuffle(x)
        x = self.project_out(x)
        return x

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)

class RefinementAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(RefinementAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
  
class RefinementBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(RefinementBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = RefinementAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# Feature-Grouped Attention (Channel Attention)
class FGA_C(nn.Module):
    def __init__(self, dim, num_heads, bias, ifBox=True):
        super(FGA_C, self).__init__()
        
        self.factor = num_heads
        self.ifBox = ifBox
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = Conv2d(dim, dim*5, kernel_size=1, bias=bias)
        self.qkv_dwconv = Conv2d(dim*5, dim*5, kernel_size=3, stride=1, padding=1, groups=dim*5, bias=bias)
        self.project_out = Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        self.sa_para = nn.Parameter(torch.ones(dim, 1))
        self.ca_para = nn.Parameter(torch.zeros(dim, 1))

    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw//factor+1)*factor-hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad
    
    def unpad(self, x, t_pad):
        _, _, hw = x.shape
        return x[:,:,t_pad[0]:hw-t_pad[1]]

    def softmax_1(self, x, dim=-1):
        logit = x.exp()
        logit  = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit

    def normalize(self, x):
        mu = x.mean(-2, keepdim=True)
        sigma = x.var(-2, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) #* self.weight + self.bias
    
    def in_group_attention(self, q, k, v):
        b, c = q.shape[:2]
        
        q, t_pad = self.pad(q, self.factor)
        k, t_pad = self.pad(k, self.factor)
        v, t_pad = self.pad(v, self.factor)

        hw = q.shape[-1] // self.factor
        shape_ori = "b (head c) (factor hw)"
        shape_tar = "b head (factor c) hw"
        
        q = rearrange(q, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        k = rearrange(k, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        v = rearrange(v, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax_1(attn, dim=-1)
        
        out = (attn @ v)
        out = rearrange(out, '{} -> {}'.format(shape_tar, shape_ori), factor=self.factor, hw=hw, b=b, head=self.num_heads)
        out = self.unpad(out, t_pad)
        return out

    def cross_group_attention(self, q, k, v):
        b, c = q.shape[:2]
        
        q, t_pad = self.pad(q, self.factor)
        k, t_pad = self.pad(k, self.factor)
        v, t_pad = self.pad(v, self.factor)

        hw = q.shape[-1] // self.factor
        
        shape_ori = "b (head c) (factor hw)"
        shape_tar = "b (head c) hw factor" # decompose group (self.factor)
        q = rearrange(q, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        v = rearrange(v, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        
        # Similarity Finder
        if self.factor != 1:
            group_v_mean = v.mean(-2).permute(0,2,1)      # "b (head c) factor"
            
            cos_sim = torch.cosine_similarity(group_v_mean.unsqueeze(2), group_v_mean.unsqueeze(1), dim=-1)
            _, sim_idx = cos_sim.topk(2,1)
            sim_idx_ = sim_idx[:,1]
            q = torch.gather(q, -1, index=sim_idx_[:,None,None,:].repeat(1,c,hw,1))
        
        shape_ori = "b (head c) hw factor"
        shape_tar = "b head (factor c) hw"
        q = rearrange(q, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        v = rearrange(v, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        
        shape_ori = "b (head c) (factor hw)"
        shape_tar = "b head (factor c) hw"
        k = rearrange(k, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax_1(attn, dim=-1)
        
        out = (attn @ v)
        out = rearrange(out, '{} -> {}'.format(shape_tar, shape_ori), factor=self.factor, hw=hw, b=b, head=self.num_heads)
        out = self.unpad(out, t_pad)
        return out

    def forward(self, data):
        
        feats, idx = data
        b,c,h,w = feats.shape

        feats = feats * idx
        idx_ = idx.reshape(b,1,-1).sort(-1)[1]
        
        # reshape & sorting
        feats = torch.gather(feats.reshape(b,c,-1), -1, index=idx_.repeat(1,c,1)).reshape(b,c,h,w)
        
        # qkv
        qkv = self.qkv_dwconv(self.qkv(feats))
        q1,k1,q2,k2,v = qkv.chunk(5, dim=1)
        q1,k1,q2,k2,v = q1.reshape(b,c,-1), k1.reshape(b,c,-1), q2.reshape(b,c,-1), k2.reshape(b,c,-1), v.reshape(b,c,-1)
       
        # In-Group Attention
        sa_out = self.in_group_attention(q1, k1, v)
        
        # Cross-Group Attention
        ca_out = self.cross_group_attention(q2, k2, v)
        
        out = sa_out * self.sa_para + ca_out * self.ca_para
        out = torch.scatter(out, 2, idx_.repeat(1,c,1), out).view(b,c,h,w)
        out = self.project_out(out)
        return (out, idx) 

# Feature-Grouped Attention (Spatial Attention)
class FGA_S(nn.Module):
    def __init__(self, dim, num_heads, num_factor, bias):
        super(FGA_S, self).__init__()
        
        self.factor = num_factor

        self.qkv = Conv2d(dim, dim*5, kernel_size=1, bias=bias)
        self.qkv_dwconv = Conv2d(dim*5, dim*5, kernel_size=3, stride=1, padding=1, groups=dim*5, bias=bias)
        self.project_out = Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        self.sa_para = nn.Parameter(torch.ones(dim, 1))
        self.ca_para = nn.Parameter(torch.zeros(dim, 1))
        
    def ia_pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw//factor+1)*factor-hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad
    
    def ia_unpad(self, x, t_pad):
        _, _, hw = x.shape
        return x[:,:,t_pad[0]:hw-t_pad[1]]

    def softmax_1(self, x, dim=-1):
        logit = x.exp()
        logit  = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit

    def normalize(self, x):
        mu = x.mean(-2, keepdim=True)
        sigma = x.var(-2, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5)
    
    def in_group_attention(self, q, k, v):
        b, c = q.shape[:2]
        
        q, t_pad = self.ia_pad(q, self.factor)
        k, t_pad = self.ia_pad(k, self.factor)
        v, t_pad = self.ia_pad(v, self.factor)

        hw = q.shape[-1] // self.factor
        shape_ori = "b c (factor hw)"
        shape_tar = "b factor hw c"
        
        q = rearrange(q, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, c=c)
        k = rearrange(k, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, c=c)
        v = rearrange(v, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, c=c)
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax_1(attn, dim=-1)
        
        out = (attn @ v)
        out = rearrange(out, '{} -> {}'.format(shape_tar, shape_ori), factor=self.factor, hw=hw, b=b, c=c)
        out = self.ia_unpad(out, t_pad)
        return out
    
    def cross_group_attention(self, q, k, v):
        b, c = q.shape[:2]
        
        q, t_pad = self.ia_pad(q, self.factor)
        k, t_pad = self.ia_pad(k, self.factor)
        v, t_pad = self.ia_pad(v, self.factor)

        hw = q.shape[-1] // self.factor
        shape_ori = "b c (factor hw)"
        shape_tar = "b factor hw c"
        
        q = rearrange(q, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, c=c)
        k = rearrange(k, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, c=c)
        v = rearrange(v, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, c=c)
        
        # Group Selector
        if hw != 1:
            group_v_mean = v.mean(-2)
            cos_sim = torch.cosine_similarity(group_v_mean.unsqueeze(2), group_v_mean.unsqueeze(1), dim=-1)
            _, sim_idx = cos_sim.topk(2,1)
            sim_idx_ = sim_idx[:,1]
            q = torch.gather(q, 1, index=sim_idx_[:,:,None,None].repeat(1,1,hw,c))
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax_1(attn, dim=-1)
        
        out = (attn @ v)
        out = rearrange(out, '{} -> {}'.format(shape_tar, shape_ori), factor=self.factor, hw=hw, b=b, c=c)
        out = self.ia_unpad(out, t_pad)
        return out

    def forward(self, data):
        feats, idx = data
        b,c,h,w = feats.shape
        
        feats = feats * idx
        idx = idx.reshape(b,1,-1).sort(-1)[1]
        
        # qkv
        qkv = self.qkv_dwconv(self.qkv(feats))
        q1,k1,q2,k2,v = qkv.chunk(5, dim=1)
        
        # reshape & sorting
        q1 = torch.gather(q1.reshape(b,c,-1), -1, index=idx.repeat(1,c,1))
        k1 = torch.gather(k1.reshape(b,c,-1), -1, index=idx.repeat(1,c,1))
        q2 = torch.gather(q2.reshape(b,c,-1), -1, index=idx.repeat(1,c,1))
        k2 = torch.gather(k2.reshape(b,c,-1), -1, index=idx.repeat(1,c,1))
        v = torch.gather(v.reshape(b,c,-1), -1, index=idx.repeat(1,c,1))
        
        # In-Group Attention Branch
        ia_out = self.in_group_attention(q1, k1, v)
        
        # Cross-Group Attention Branch
        ca_out = self.cross_group_attention(q2, k2, v)
        
        out = ia_out * self.sa_para + ca_out * self.ca_para        
        out = torch.scatter(out, 2, idx.repeat(1,c,1), out).view(b,c,h,w)
        out = self.project_out(out)
        return out

# Spatial Grouping Transformer Block (Channel Attention)
class SGTB_C(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(SGTB_C, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = FGA_C(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, data):
        feats, idx = data
        out, idx = self.attn((self.norm1(feats), idx))
        feats = feats + out
        feats = feats + self.ffn(self.norm2(feats))
        return (feats, idx)

# Spatial Grouping Transformer Block (Spatial Attention)
class SGTB_S(nn.Module):
    def __init__(self, dim, num_heads, num_factor, ffn_expansion_factor, bias, LayerNorm_type):
        super(SGTB_S, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = FGA_S(dim, num_heads, num_factor, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, data):
        feats, idx = data
        out = self.attn((self.norm1(feats), idx))
        feats = feats + out
        feats = feats + self.ffn(self.norm2(feats))
        return (feats, idx)

##########################################################################
# embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
# Spectral-based Decomposition Prompt
class SDP(nn.Module):
    def __init__(self, in_c, dim, num_prompt, bias=False):
        super(SDP, self).__init__()
        
        self.dim = dim
        
        self.conv_skip_img = nn.Conv2d(in_c, dim, kernel_size=1)
        self.conv_skip_svd = nn.Conv2d(1, dim, kernel_size=1)
        
        # Sobel
        self.conv_gaussian = nn.Conv2d(in_c, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_sobel = nn.Conv2d(dim*2, dim, kernel_size=3, stride=1, padding=1, bias=False)
        
        # SVD
        self.conv_svd1 = nn.Conv2d(in_c+1, in_c+1, kernel_size=11, stride=1, padding=5, padding_mode='reflect', bias=False)
        self.conv_svd2 = nn.Conv2d(in_c+1, in_c+1, kernel_size=3,  stride=1, padding=1, padding_mode='reflect', bias=False, groups=in_c+1)
        self.conv_svd3 = nn.Conv2d(in_c+1, dim-1, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False)
        
        # Spectral Feature Fusion Module
        self.linear_sobel = nn.Linear(dim, num_prompt, bias)
        self.prompt_sobel = nn.Parameter(torch.randn(num_prompt, 1, 1, dim))
        self.conv_svd4 = nn.Conv2d(dim, num_prompt, kernel_size=3, stride=1, padding=1, bias=False)
        self.prompt_svd = nn.Parameter(torch.randn(num_prompt, dim))
        
        self.attention = LinearAttention(dim*2, num_prompt, 1)
        self.pool_max = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=False)
        self.linear2 = nn.Linear(dim, dim//2, bias)
        self.conv_mask = nn.Conv2d(dim*2, 1, kernel_size=7, stride=1, padding=3, padding_mode='reflect', bias=False)
        self.conv4 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.shuffle = torch.nn.PixelShuffle(2)
        self.conv_unshuffle_sobel = nn.Conv2d(dim//4, dim, kernel_size=1)
        self.conv_unshuffle_svd = nn.Conv2d(dim//4, dim, kernel_size=1)
        
    def convolve(self, image, kernel):
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(self.dim, 1, 1, 1).to(image.device)
        output = F.conv2d(image, kernel, padding=kernel.size(-1) // 2, groups=self.dim)
        return output
    
    def sobel_kernels(self):
        Kx = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=torch.float32)
        Ky = torch.tensor([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]], dtype=torch.float32)
        return Kx, Ky
    
    def sobel(self, x):
        x = self.conv_gaussian(x) # replacing guassuian kernel
        
        Kx, Ky = self.sobel_kernels()
        Gx = self.convolve(x, Kx) + 1e-8
        Gy = self.convolve(x, Ky) + 1e-8
        
        edge = torch.hypot(Gx, Gy)
        theta = torch.atan2(Gy, Gx) / 3.1416

        return edge, theta
        
    def forward(self, inp_img, feats, svd):

        b,_,h,w = feats.shape

        # interpolation
        img_svd_cat = torch.cat([inp_img, svd], dim=1)
        img_svd_cat = F.interpolate(img_svd_cat, (h,w), mode='bilinear')
        img = img_svd_cat[:,:3]
        svd = img_svd_cat[:,3:]
        
        # pool
        img_pool = self.conv_skip_img(img.clone())
        svd_pool = self.conv_skip_svd(svd.clone())
        
        # Sobel Feature
        edge, theta = self.sobel(img)
        
        # Sobel Refinement Block
        sobel = self.conv_sobel(torch.cat([edge, theta], dim=1))
        sobel = rearrange(sobel, 'b c h w -> b (h w) c', b=b, h=h, w=w)
        sobel = self.linear_sobel(sobel)
        sobel = F.softmax(sobel,-1)
        sobel = rearrange(sobel, 'b (h w) c -> b c h w', b=b, h=h, w=w)
        sobel = self.prompt_sobel.unsqueeze(0) * sobel.unsqueeze(-1)
        sobel = sobel.sum(1).permute(0,3,1,2)

        # SVD Refinement Block
        svd_1 = self.conv_svd1(img_svd_cat)
        svd_2 = self.conv_svd2(img_svd_cat)
        svd = torch.cat([self.conv_svd3(svd_1 + svd_2), svd], dim=1)
        svd = self.conv_svd4(svd)
        svd = rearrange(svd, 'b c h w -> b (h w) c', b=b, h=h, w=w)
        svd = svd @ self.prompt_svd
        svd = rearrange(svd, 'b (h w) c -> b c h w', b=b, h=h, w=w)

        x = torch.cat([sobel, svd], dim=1)
        
        original_list = torch.arange(0, self.dim * 2)
        reordered_list = torch.stack([original_list[:self.dim], original_list[self.dim:]], dim=1).flatten().tolist()
        
        # Multi-Head Linear Attention
        x = x[:, reordered_list]
        x = self.attention(x)
        sobel, svd = torch.chunk(x, 2, dim=1)
        
        # Sobel
        sobel = self.shuffle(sobel)
        sobel = self.conv_unshuffle_sobel(sobel)
        sobel_gap = F.adaptive_avg_pool2d(sobel, (1,1))
        
        # SVD
        svd = self.shuffle(svd)
        svd = self.conv_unshuffle_svd(svd)

        x = torch.cat([sobel, svd], dim=1)
        x = self.pool_max(x)
        sobel, svd = torch.chunk(x, 2, dim=1)

        sobel = img_pool * sobel * sobel_gap
        svd = svd_pool * svd

        x = torch.cat([sobel, svd], dim=1)
        mask = self.conv_mask(x)
        return x, mask

##########################################################################
# SSGformer
class SSGformer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_ca_blocks = [4,6,6,8], 
        num_sa_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        num_factors = [2,4,8,16],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False,      ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        patch_size = 16,
        num_prompt = 8,
        dim_prompt = 16,
    ):

        super(SSGformer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level_sa1 = nn.Sequential(*[SGTB_S(dim=dim, num_heads=heads[0], num_factor=num_factors[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_sa_blocks[0])])
        self.encoder_level_ca1 = nn.Sequential(*[SGTB_C(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_ca_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level_sa2 = nn.Sequential(*[SGTB_S(dim=int(dim*2**1), num_heads=heads[1], num_factor=num_factors[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_sa_blocks[1])])
        self.encoder_level_ca2 = nn.Sequential(*[SGTB_C(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_ca_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level_sa3 = nn.Sequential(*[SGTB_S(dim=int(dim*2**2), num_heads=heads[2], num_factor=num_factors[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_sa_blocks[2])])
        self.encoder_level_ca3 = nn.Sequential(*[SGTB_C(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_ca_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent_sa = nn.Sequential(*[SGTB_S(dim=int(dim*2**3), num_heads=heads[3], num_factor=num_factors[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_sa_blocks[3])])
        self.latent_ca = nn.Sequential(*[SGTB_C(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_ca_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level_sa3 = nn.Sequential(*[SGTB_S(dim=int(dim*2**2), num_heads=heads[2], num_factor=num_factors[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_sa_blocks[2])])
        self.decoder_level_ca3 = nn.Sequential(*[SGTB_C(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_ca_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level_sa2 = nn.Sequential(*[SGTB_S(dim=int(dim*2**1), num_heads=heads[1], num_factor=num_factors[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_sa_blocks[1])])
        self.decoder_level_ca2 = nn.Sequential(*[SGTB_C(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_ca_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level_sa1 = nn.Sequential(*[SGTB_S(dim=int(dim*2**1), num_heads=heads[0], num_factor=num_factors[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_sa_blocks[0])])
        self.decoder_level_ca1 = nn.Sequential(*[SGTB_C(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_ca_blocks[0])])
        
        self.refinement = nn.Sequential(*[RefinementBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        # skip embed
        self.SDP1 = SDP(3, dim_prompt, num_prompt)
        self.SDP2 = SDP(3, dim_prompt, num_prompt)
        self.SDP3 = SDP(3, dim_prompt, num_prompt)
        self.SDP4 = SDP(3, dim_prompt, num_prompt)
        
        self.reduce_chan_level_1 = Conv2d(int(dim*2**0)+dim_prompt*2, int(dim*2**0), kernel_size=1, bias=bias)
        self.reduce_chan_level_2 = Conv2d(int(dim*2**1)+dim_prompt*2, int(dim*2**1), kernel_size=1, bias=bias)
        self.reduce_chan_level_3 = Conv2d(int(dim*2**2)+dim_prompt*2, int(dim*2**2), kernel_size=1, bias=bias)
        self.reduce_chan_level_4 = Conv2d(int(dim*2**3)+dim_prompt*2, int(dim*2**3), kernel_size=1, bias=bias)

        self.output = Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        

    def forward(self, inp_img, svd, filename=None):
        
        inp_img = inp_img / 255.0
        
        inp_enc_level1 = self.patch_embed(inp_img)

        skip_enc_level1, mask1 = self.SDP1(inp_img, inp_enc_level1, svd)
        inp_enc_level1 = self.reduce_chan_level_1(torch.cat([inp_enc_level1, skip_enc_level1], 1))
        out_enc_level1, _ = self.encoder_level_ca1((inp_enc_level1, mask1))
        out_enc_level1, _ = self.encoder_level_sa1((out_enc_level1, mask1))

        inp_enc_level2 = self.down1_2(out_enc_level1)
        skip_enc_level2, mask2 = self.SDP2(inp_img, inp_enc_level2, svd)
        inp_enc_level2 = self.reduce_chan_level_2(torch.cat([inp_enc_level2, skip_enc_level2], 1))
        out_enc_level2, _ = self.encoder_level_ca2((inp_enc_level2, mask2))
        out_enc_level2, _ = self.encoder_level_sa2((out_enc_level2, mask2))

        inp_enc_level3 = self.down2_3(out_enc_level2)
        skip_enc_level3, mask3 = self.SDP3(inp_img, inp_enc_level3, svd)
        inp_enc_level3 = self.reduce_chan_level_3(torch.cat([inp_enc_level3, skip_enc_level3], 1))
        out_enc_level3, _ = self.encoder_level_ca3((inp_enc_level3, mask3)) 
        out_enc_level3, _ = self.encoder_level_sa3((out_enc_level3, mask3)) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        skip_enc_level4, mask4 = self.SDP4(inp_img, inp_enc_level4, svd)
        inp_enc_level4 = self.reduce_chan_level_4(torch.cat([inp_enc_level4, skip_enc_level4], 1))
        latent, _ = self.latent_ca((inp_enc_level4, mask4)) 
        latent, _ = self.latent_sa((latent, mask4)) 
        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3, _ = self.decoder_level_ca3((inp_dec_level3, mask3)) 
        out_dec_level3, _ = self.decoder_level_sa3((out_dec_level3, mask3)) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2, _ = self.decoder_level_ca2((inp_dec_level2, mask2)) 
        out_dec_level2, _ = self.decoder_level_sa2((out_dec_level2, mask2)) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1, _ = self.decoder_level_ca1((inp_dec_level1, mask1))
        out_dec_level1, _ = self.decoder_level_sa1((out_dec_level1, mask1))
        
        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1)
            
        return torch.clamp((out_dec_level1 + inp_img)*255.0 ,0,255)
