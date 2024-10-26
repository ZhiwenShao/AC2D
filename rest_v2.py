# ------------------------------------------------------------
# Written by Qing-Long Zhang
# Modified by Zhiwen Shao
# ------------------------------------------------------------

import torch
import torch.nn as nn

import math

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model


class Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio + 1, stride=sr_ratio, padding=sr_ratio // 2, groups=dim)
            self.sr_norm = nn.LayerNorm(dim, eps=1e-6)

        self.up = nn.Sequential(
            nn.Conv2d(dim, sr_ratio * sr_ratio * dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.PixelShuffle(upscale_factor=sr_ratio)
        )
        self.up_norm = nn.LayerNorm(dim, eps=1e-6)

        self.proj = nn.Linear(dim, dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.sr_norm(x)

        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        identity = v.transpose(-1, -2).reshape(B, C, H // self.sr_ratio, W // self.sr_ratio)
        identity = self.up(identity).flatten(2).transpose(1, 2)
        x = self.proj(x + self.up_norm(identity))
        return x


class Attention_withRestrict(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio + 1, stride=sr_ratio, padding=sr_ratio // 2, groups=dim)
            self.sr_norm = nn.LayerNorm(dim, eps=1e-6)

        self.up = nn.Sequential(
            nn.Conv2d(dim, sr_ratio * sr_ratio * dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.PixelShuffle(upscale_factor=sr_ratio)
        )
        self.up_norm = nn.LayerNorm(dim, eps=1e-6)

        self.proj = nn.Linear(dim, dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.sr_norm(x)

        kv = self.kv(x)
        kv = kv.reshape(B, -1, 2, self.num_heads, C // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_vector = attn.reshape(attn.size(0), 1, attn.size(1)*attn.size(2)*attn.size(3))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        identity = v.transpose(-1, -2).reshape(B, C, H // self.sr_ratio, W // self.sr_ratio)
        identity = self.up(identity).flatten(2).transpose(1, 2)
        x = self.proj(x + self.up_norm(identity))
        return x, attn_vector


class Block(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio=1, drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, sr_ratio=sr_ratio)

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))  # pre_norm
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Block_withRestrict(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio=1, drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention_withRestrict(dim, num_heads=num_heads, sr_ratio=sr_ratio)

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, H, W):
        new_x, attn_vector = self.attn(self.norm1(x), H, W)
        x = x + self.drop_path(new_x)  # pre_norm
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn_vector


class PA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))


class Stem(nn.Module):
    def __init__(self, in_dim=3, out_dim=96, patch_size=2):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(out_dim, eps=1e-6)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class ConvStem(nn.Module):
    def __init__(self, in_ch=3, out_ch=48, patch_size=2, with_pos=True):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        stem = []
        in_dim, out_dim = in_ch, out_ch // 2
        for i in range(2):
            stem.append(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=i+1, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(out_dim))
            stem.append(nn.ReLU(inplace=True))
            in_dim, out_dim = out_dim, out_dim * 2

        stem.append(nn.Conv2d(in_dim, out_ch, kernel_size=1, stride=1))
        self.proj = nn.Sequential(*stem)

        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

        self.norm = nn.LayerNorm(out_ch, eps=1e-6)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.with_pos:
            x = self.pos(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, out_ch=96, patch_size=2, with_pos=True):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=patch_size + 1, stride=patch_size, padding=patch_size // 2)

        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

        self.norm = nn.LayerNorm(out_ch, eps=1e-6)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.with_pos:
            x = self.pos(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class DataAttentionLayer(nn.Module):
    def __init__(self, input_channel, d_model, output_channel):
        super().__init__()

        self.input_channel = input_channel
        self.d_model = d_model
        self.output_channel = output_channel

        # subject attention
        self.kmap = nn.Linear(self.input_channel, self.d_model)
        self.qmap = nn.Linear(self.input_channel, self.d_model)
        self.xmap = nn.Linear(self.input_channel, self.output_channel)
        self.smap = nn.Linear(self.input_channel, self.output_channel)

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = scores.softmax(dim=-1)
        # p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, x, data_infos, i=None):

        K_data, V_data = data_infos
        if i is not None:
            K_data, V_data = K_data[i,:].unsqueeze(0), V_data[i,:].unsqueeze(0)
        else:
            K_data, V_data = K_data.unsqueeze(0), V_data.unsqueeze(0)

        K_data = self.kmap(K_data)
        Q_data = self.qmap(x)

        data_embedding, _ = self.attention(Q_data, K_data, V_data)
        x = self.xmap(x) + self.smap(data_embedding)

        return x


class ResTV2_ac2d(nn.Module):
    def __init__(self, in_chans=3, au_num=12, embed_dims=[48, 96, 192],
                 num_heads=[1, 2, 4], drop_path_rate=0.,
                 depths=[2, 2, 2], sr_ratios=[4, 2, 1], causal_dim=512):
        super().__init__()
        self.depths = depths

        self.stem = ConvStem(in_chans, embed_dims[0], patch_size=2)
        self.patch_2 = PatchEmbed(embed_dims[0], embed_dims[1], patch_size=2)
        self.aus_patch_3 = nn.ModuleList([
            PatchEmbed(embed_dims[1], embed_dims[2], patch_size=2)
            for j in range(au_num)])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.stage1 = nn.ModuleList([
            Block(embed_dims[0], num_heads[0], sr_ratios[0], dpr[cur + i])
            for i in range(depths[0])
        ])

        cur += depths[0]
        self.stage2 = nn.ModuleList([
            Block(embed_dims[1], num_heads[1], sr_ratios[1], dpr[cur + i])
            for i in range(depths[1])
        ])

        cur += depths[1]

        #multiple AU branches
        self.aus_stage3_start = nn.ModuleList([
            nn.ModuleList([
                Block(embed_dims[2], num_heads[2], sr_ratios[2], dpr[cur + i])
                for i in range(depths[2]-2)
            ])
            for j in range(au_num)])

        self.aus_stage3_interm = nn.ModuleList([
            Block_withRestrict(embed_dims[2], num_heads[2], sr_ratios[2], dpr[cur + depths[2]-2])
            for j in range(au_num)])

        self.aus_stage3_end = nn.ModuleList([
            Block(embed_dims[2], num_heads[2], sr_ratios[2], dpr[cur + depths[2]-1])
            for j in range(au_num)])

        # final norm layer
        self.aus_norm = nn.ModuleList([
            nn.LayerNorm(embed_dims[2], eps=1e-6)
            for j in range(au_num)])

        # classification head
        self.aus_avg_pool = nn.ModuleList([
            nn.AdaptiveAvgPool2d((1, 1))
            for j in range(au_num)])

        # causal
        self.aus_data_attention = nn.ModuleList([
            DataAttentionLayer(
            embed_dims[2], causal_dim,
            embed_dims[2])
            for j in range(au_num)])

        self.aus_head = nn.ModuleList([
            nn.Linear(embed_dims[2], 1)
            for j in range(au_num)])

        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    def forward(self, input):
        x, data_infos = input
        B, _, H, W = x.shape
        x, (H, W) = self.stem(x)

        # stage 1
        for blk in self.stage1:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)

        # stage 2
        x, (H, W) = self.patch_2(x)
        for blk in self.stage2:
            x = blk(x, H, W)
        shared_feat = x.permute(0, 2, 1).reshape(B, -1, H, W)

        # stage 3
        for i in range(len(self.aus_patch_3)):
            x, (H, W) = self.aus_patch_3[i](shared_feat)
            for blk in self.aus_stage3_start[i]:
                x = blk(x, H, W)
            x, attn_vector = self.aus_stage3_interm[i](x, H, W)
            x = self.aus_stage3_end[i](x, H, W)

            x = self.aus_norm[i](x)

            x = x.permute(0, 2, 1).reshape(B, -1, H, W)
            x = self.aus_avg_pool[i](x).flatten(1)
            feature = x.unsqueeze(1)

            if data_infos:
                x = self.aus_data_attention[i](x, data_infos, i)

            au_output = self.aus_head[i](x)
            if i == 0:
                aus_output = au_output
                aus_attention = attn_vector
                aus_feature = feature
            else:
                aus_output = torch.cat((aus_output, au_output), 1)
                aus_attention = torch.cat((aus_attention, attn_vector), 1)
                aus_feature = torch.cat((aus_feature, feature), 1)

        return aus_feature, aus_attention, aus_output


@register_model
def restv2_tiny_ac2d(pretrained=False, **kwargs):  # 82.3|4.7G|24M -> |3.92G|30.37M   4.5G|30.33M
    model = ResTV2_ac2d(embed_dims=[64, 64, 64], depths=[1, 6, 3], **kwargs)
    return model



