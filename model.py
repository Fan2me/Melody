import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from einops import rearrange
import config


def get_mamba_config(d_in):
    if d_in <= 16:
        return 8, 1
    elif d_in <= 32:
        return 16, 1.5
    else:
        return 32, 2


class AttentionWeight(nn.Module):
    def __init__(self, channel, kernel_size=7):
        super(AttentionWeight, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size=1)
        self.conv2 = nn.Conv1d(
            channel,
            channel,
            kernel_size,
            padding=padding,
            groups=channel,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_weight = torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )
        x_weight = self.conv1(x_weight).squeeze(1)
        x_weight = self.sigmoid(self.bn(self.conv2(x_weight)))
        x_weight = x_weight.unsqueeze(1)

        return x * x_weight


class IIA(nn.Module):

    def __init__(self, channel):
        super(IIA, self).__init__()
        self.attention_h = AttentionWeight(channel)
        self.attention_w = AttentionWeight(channel)

    def forward(self, x):
        x_f_permuted = x.permute(0, 3, 1, 2).contiguous()
        x_f_attended = self.attention_h(x_f_permuted)
        x_f = x_f_attended.permute(0, 2, 3, 1).contiguous()
        x_t_permuted = x.permute(0, 2, 1, 3).contiguous()
        x_t_attended = self.attention_w(x_t_permuted)
        x_t = x_t_attended.permute(0, 2, 1, 3).contiguous()
        return x + x_f + x_t

class Conformity(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conformity, self).__init__()
        self.IIA = IIA(in_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        x = self.IIA(x)
        x = self.conv(x)

        return x


class CrossDimMambaBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        d_in = d_model // 2
        self.mamba_proj = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_in), nn.SiLU(), nn.Dropout(0.2)
        )
        d_state, expand = get_mamba_config(d_in)
        self.freq_mamba = Mamba(d_model=d_in, d_state=d_state, d_conv=4, expand=expand)
        self.temp_mamba = Mamba(d_model=d_in, d_state=32, d_conv=4, expand=expand)
        self.mamba_output = nn.Linear(d_in, d_model)
        self.iia = Conformity(d_model * 2, d_model)

    def forward(self, x):
        B, C, F, T = x.shape
        x_freq_attn = x
        x_time_attn = x

        x_freq_in = rearrange(x_freq_attn, "b c f t -> (b t) f c")
        x_freq_in = self.mamba_proj(x_freq_in)
        x_freq_out = self.freq_mamba(x_freq_in)
        x_freq_out = self.mamba_output(x_freq_out)
        x_freq = rearrange(x_freq_out, "(b t) f c -> b c f t", b=B)

        x_time_in = rearrange(x_time_attn, "b c f t -> (b f) t c")
        x_time_in = self.mamba_proj(x_time_in)
        x_time_out = self.temp_mamba(x_time_in)
        x_time_out = self.mamba_output(x_time_out)
        x_time = rearrange(x_time_out, "(b f) t c -> b c f t", b=B)
        x_fuse = torch.cat([x_freq, x_time], dim=1)
        x_fuse = self.iia(x_fuse)
        output = x + x_fuse
        return output


class OctaveMaxPool(nn.Module):
    def __init__(self, octaves=6, pitches_per_octave=12, channel=64):
        super().__init__()
        self.octaves = octaves
        self.pitches_per_octave = pitches_per_octave

    def forward(self, x):
        O, P = self.octaves, self.pitches_per_octave
        x_reshaped = rearrange(x, "b c (o p) t -> b c o p t", o=O, p=P)
        tone_fused, idx = x_reshaped.max(dim=2)
        return tone_fused, idx


class OctaveMaxUnpool(nn.Module):
    def __init__(self, octaves=6, pitches_per_octave=12):
        super().__init__()
        self.octaves = octaves
        self.pitches_per_octave = pitches_per_octave

    def forward(self, tone_out, idx):
        B, C, P, T = tone_out.shape
        O = self.octaves
        idx = idx.unsqueeze(2)
        tone_out = tone_out.unsqueeze(2)
        restored = torch.zeros(B, C, O, P, T, device=tone_out.device)
        restored.scatter_(2, idx, tone_out)
        restored = rearrange(restored, "b c o p t -> b c (o p) t")

        return restored


class MambaNet(nn.Module):

    def __init__(self):
        super().__init__()
        C = config.encoder_channels
        self.enc1_block = nn.Sequential(
            nn.Conv2d(3, C[0], kernel_size=5, padding=2),
            nn.BatchNorm2d(C[0]),
            nn.SiLU(),
            CrossDimMambaBlock(
                C[0],
            ),
        )
        self.pool1 = nn.MaxPool2d(config.pool1_kernel, return_indices=True)

        self.enc2_block = nn.Sequential(
            nn.Conv2d(C[0], C[1], kernel_size=5, padding=2),
            nn.BatchNorm2d(C[1]),
            nn.SiLU(),
            CrossDimMambaBlock(
                C[1],
            ),
        )
        self.pool2 = OctaveMaxPool(channel=C[1])

        self.enc3_block = nn.Sequential(
            nn.Conv2d(C[1], C[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(C[2]),
            nn.SiLU(),
            CrossDimMambaBlock(
                C[2],
            ),
        )

        self.dec3_block = nn.Sequential(
            nn.Conv2d(C[2], C[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(C[1]),
            nn.SiLU(),
            CrossDimMambaBlock(
                C[1],
            ),
        )

        self.up_pool2 = OctaveMaxUnpool()
        self.dec2_block = nn.Sequential(
            nn.Conv2d(C[1], C[0], kernel_size=5, padding=2),
            nn.BatchNorm2d(C[0]),
            nn.SiLU(),
            CrossDimMambaBlock(
                C[0],
            ),
        )
        self.up_pool1 = nn.MaxUnpool2d(config.pool1_kernel)
        self.dec1_block = nn.Sequential(
            nn.Conv2d(C[0], C[0], kernel_size=5, padding=2),
            nn.BatchNorm2d(C[0]),
            nn.SiLU(),
            CrossDimMambaBlock(
                C[0],
            ),
        )

        self.fine_proj = nn.Sequential(
            nn.BatchNorm2d(C[0]),
            nn.Conv2d(C[0], C[0], kernel_size=5, padding=2),
            nn.SiLU(),
            nn.BatchNorm2d(C[0]),
            nn.Conv2d(C[0], 1, kernel_size=5, padding=2),
            nn.SiLU(),
        )

        self.no_melody_proj = nn.Sequential(
            nn.BatchNorm2d(C[2]),
            nn.Conv2d(C[2], 1, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, None)),
        )

    def forward(self, x):
        c1 = self.enc1_block(x)
        c1_pool, idx = self.pool1(c1)
        c2 = self.enc2_block(c1_pool)
        c2_pool, idx2 = self.pool2(c2)
        c3 = self.enc3_block(c2_pool)
        silence_mask = self.no_melody_proj(c3).squeeze(1)
        u3 = self.dec3_block(c3)
        u3 = c2_pool + u3
        u2 = self.up_pool2(u3, idx2) 
        u2 = c2 + u2
        u2 = self.dec2_block(u2)
        u1 = self.up_pool1(u2, idx) 
        u1 = c1 + u1
        u1 = self.dec1_block(u1) 
        fine_out = self.fine_proj(u1).squeeze(1) 
        final_output = torch.cat([silence_mask, fine_out], dim=1)
        return final_output.softmax(dim=1)
