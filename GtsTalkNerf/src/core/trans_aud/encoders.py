# coding: utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class AudEncoder(nn.Module):
    def __init__(self, out_dim=128, model_name='WAV2VEC2_ASR_BASE_960H'):
        super(AudEncoder, self).__init__()

        model = getattr(torchaudio.pipelines, model_name).get_model()
        feature_dim = model.encoder.feature_projection.projection.weight.shape[0]
        model.aux = nn.Linear(feature_dim + 6, out_dim)

        for param in model.feature_extractor.parameters():
            param.requires_grad = False
        self.model = model

    def forward(self, waveforms, formant):
        # wav: 1, 640*L    formant: 1, L, 6
        x, _ = self.model.feature_extractor(waveforms, None)
        length = formant.shape[1]
        if isinstance(length, int) and length > 0:
            x = x.transpose(-1, -2)
            x = F.interpolate(x, length, align_corners=True, mode='linear')
            x = x.transpose(-1, -2).contiguous()

        x = self.model.encoder(x, None)
        return self.model.aux(torch.cat([x, formant], -1))

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        self.model.feature_extractor.train(False)
        self.model.encoder.train(mode)
        self.model.aux.train(mode)
        return self


class lmkEncoder(nn.Module):
    def __init__(self,
                 out_dim: int = 128,
                 lmk_dim: int = 3,
                 emb_dim: int = 512,
                 hid_dim: int = 2048,
                 dropout: float = 0.1,
                 layer_num: int = 2,
                 ):
        super(lmkEncoder, self).__init__()
        self.embeder = nn.Conv1d(lmk_dim, emb_dim, 5, dilation=2)

        self.classifer_token = nn.Parameter(torch.zeros(1, 1, emb_dim))

        x = torch.arange(1 + int((37 + 2 * 0 - 2 * (5 - 1) - 1) / 1 + 1))
        y = x[:, None] * torch.exp(-torch.linspace(0, 1, emb_dim // 2 + 1) * math.log(1000))
        pe = torch.stack([y, y + torch.pi / 2], -1).view(1, x.size(0), -1)
        self.pos_emb = nn.Parameter(torch.sin(pe))

        self.dropout = nn.Dropout(dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(emb_dim, 16, hid_dim, dropout=dropout, activation=F.gelu, batch_first=True),
            layer_num,
            nn.LayerNorm(emb_dim),
        )

        self.fc = nn.Linear(emb_dim, out_dim)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, k) or (3, k) or (B, L, 3, k)
        Output:
            (B, feat_dim) or (1, feat_dim) or (B, L, feat_dim)
        """
        if x.ndim == 2:
            x = x.unsqueeze(0)
        sh = x.shape
        if x.ndim == 4:
            x = x.view(-1, *sh[-2:])
        # x: B, 3, 37
        x = self.embeder(x).permute(0, 2, 1)  # B,(k-2*(5-1)),512
        cls = self.classifer_token.expand(x.size(0), 1, x.size(2))
        x = torch.cat([cls, x], 1)  # add cls_token
        x = x + self.pos_emb[:, :x.size(1), :x.size(2)]  # B,(k+1-2*(5-1)),512
        x = self.dropout(x)

        x = self.transformer_encoder(x)[:, 0]  # B, 512
        x = self.fc(x)  # B, 128
        if len(sh) == 4:
            x = x.view(*sh[:2], x.size(1))
        return x
