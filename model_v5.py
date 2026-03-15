"""
TransFuser-v5: Lightweight cross-attention fusion with frozen backbones.

Key changes vs baseline (model2_seq.py):
1. Frozen ResNet backbones (only conv1 of lidar/radar adapts)
2. GPT layers reduced: 8 → 2 per scale (4 scales × 2 layers = 8 total vs 32)
3. Cross-attention fusion token replaces element-wise sum
4. Contrastive alignment head (InfoNCE) for cross-modal consistency
5. Dropout increased to 0.3 by default

Estimated trainable params: ~8-10M (vs 78M baseline)
"""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class ImageCNN(nn.Module):
    def __init__(self, c_dim, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.features = models.resnet34(pretrained=True)
        self.features.fc = nn.Sequential()

    def forward(self, inputs):
        c = 0
        for x in inputs:
            if self.normalize:
                c += self.features(normalize_imagenet(x))
            else:
                c += self.features(x)
        return c


def normalize_imagenet(x):
    x = x.clone()
    x[:, 0] = (x[:, 0] / 255.0 - 0.485) / 0.229
    x[:, 1] = (x[:, 1] / 255.0 - 0.456) / 0.224
    x[:, 2] = (x[:, 2] / 255.0 - 0.406) / 0.225
    return x


class LidarEncoder(nn.Module):
    def __init__(self, num_classes=512, in_channels=2):
        super().__init__()
        self._model = models.resnet18(pretrained=True)
        self._model.fc = nn.Sequential()
        _tmp = self._model.conv1
        self._model.conv1 = nn.Conv2d(
            in_channels, out_channels=_tmp.out_channels,
            kernel_size=_tmp.kernel_size, stride=_tmp.stride,
            padding=_tmp.padding, bias=_tmp.bias)

    def forward(self, inputs):
        features = 0
        for lidar_data in inputs:
            features += self._model(lidar_data)
        return features


# ── Lightweight Self-Attention (same structure, fewer layers) ──────────────
class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True),
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, n_embd, n_head, block_exp, n_layer,
                 vert_anchors, horz_anchors, seq_len,
                 embd_pdrop, attn_pdrop, resid_pdrop, config):
        super().__init__()
        self.n_embd = n_embd
        self.seq_len = seq_len
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        self.config = config
        self.pos_emb = nn.Parameter(
            torch.zeros(1, (config.n_views + 2) * seq_len * vert_anchors * horz_anchors + 2, n_embd))
        self.drop = nn.Dropout(embd_pdrop)
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_exp, attn_pdrop, resid_pdrop)
            for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.block_size = seq_len
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, image_tensor, lidar_tensor, radar_tensor, gps):
        bz = lidar_tensor.shape[0] // self.seq_len
        h, w = lidar_tensor.shape[2:4]
        image_tensor = image_tensor.view(bz, self.config.n_views * self.seq_len, -1, h, w)
        lidar_tensor = lidar_tensor.view(bz, self.seq_len, -1, h, w)
        radar_tensor = radar_tensor.view(bz, self.seq_len, -1, h, w)
        token_embeddings = torch.cat(
            [image_tensor, lidar_tensor, radar_tensor], dim=1
        ).permute(0, 1, 3, 4, 2).contiguous()
        token_embeddings = token_embeddings.view(bz, -1, self.n_embd)
        token_embeddings = torch.cat([token_embeddings, gps], dim=1)
        x = self.drop(self.pos_emb + token_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        pos_out = x[:, (self.config.n_views + 2) * self.seq_len * self.vert_anchors * self.horz_anchors:, :]
        x = x[:, :(self.config.n_views + 2) * self.seq_len * self.vert_anchors * self.horz_anchors, :]
        x = x.view(bz, (self.config.n_views + 2) * self.seq_len,
                    self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        nv = self.config.n_views
        sl = self.seq_len
        img_out = x[:, :nv * sl].contiguous().view(bz * nv * sl, -1, h, w)
        lid_out = x[:, nv * sl:(nv + 1) * sl].contiguous().view(bz * sl, -1, h, w)
        rad_out = x[:, (nv + 1) * sl:].contiguous().view(bz * sl, -1, h, w)
        return img_out, lid_out, rad_out, pos_out


# ── Cross-Attention Fusion Token ──────────────────────────────────────────
class CrossAttentionFusion(nn.Module):
    """Learnable fusion token attends to all modality features via cross-attention."""
    def __init__(self, d_model=512, n_heads=4, dropout=0.3):
        super().__init__()
        self.fusion_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, modality_tokens):
        """
        Args:
            modality_tokens: (B, N_tokens, d_model) — concatenated modality features
        Returns:
            fused: (B, d_model)
        """
        bz = modality_tokens.size(0)
        q = self.fusion_token.expand(bz, -1, -1)  # (B, 1, d)
        attn_out, _ = self.cross_attn(self.ln1(q), self.ln1(modality_tokens), modality_tokens)
        q = q + attn_out
        q = q + self.ffn(self.ln2(q))
        return q.squeeze(1)  # (B, d_model)


# ── Encoder with frozen backbones + lightweight transformers ──────────────
class EncoderV5(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.avgpool = nn.AdaptiveAvgPool2d((config.vert_anchors, config.horz_anchors))
        self.image_encoder = ImageCNN(512, normalize=True)
        self.lidar_encoder = LidarEncoder(num_classes=512, in_channels=1)
        if config.add_velocity:
            self.radar_encoder = LidarEncoder(num_classes=512, in_channels=2)
        else:
            self.radar_encoder = LidarEncoder(num_classes=512, in_channels=1)

        self.vel_emb1 = nn.Linear(2, 64)
        self.vel_emb2 = nn.Linear(64, 128)
        self.vel_emb3 = nn.Linear(128, 256)
        self.vel_emb4 = nn.Linear(256, 512)

        # Key change: n_layer from config (default 2 instead of 8)
        n_layer = config.n_layer
        self.transformer1 = GPT(n_embd=64, n_head=config.n_head,
            block_exp=config.block_exp, n_layer=n_layer,
            vert_anchors=config.vert_anchors, horz_anchors=config.horz_anchors,
            seq_len=config.seq_len, embd_pdrop=config.embd_pdrop,
            attn_pdrop=config.attn_pdrop, resid_pdrop=config.resid_pdrop, config=config)
        self.transformer2 = GPT(n_embd=128, n_head=config.n_head,
            block_exp=config.block_exp, n_layer=n_layer,
            vert_anchors=config.vert_anchors, horz_anchors=config.horz_anchors,
            seq_len=config.seq_len, embd_pdrop=config.embd_pdrop,
            attn_pdrop=config.attn_pdrop, resid_pdrop=config.resid_pdrop, config=config)
        self.transformer3 = GPT(n_embd=256, n_head=config.n_head,
            block_exp=config.block_exp, n_layer=n_layer,
            vert_anchors=config.vert_anchors, horz_anchors=config.horz_anchors,
            seq_len=config.seq_len, embd_pdrop=config.embd_pdrop,
            attn_pdrop=config.attn_pdrop, resid_pdrop=config.resid_pdrop, config=config)
        self.transformer4 = GPT(n_embd=512, n_head=config.n_head,
            block_exp=config.block_exp, n_layer=n_layer,
            vert_anchors=config.vert_anchors, horz_anchors=config.horz_anchors,
            seq_len=config.seq_len, embd_pdrop=config.embd_pdrop,
            attn_pdrop=config.attn_pdrop, resid_pdrop=config.resid_pdrop, config=config)

        # Cross-attention fusion replaces element-wise sum
        self.cross_fusion = CrossAttentionFusion(
            d_model=512, n_heads=4, dropout=config.resid_pdrop)

    def forward(self, image_list, lidar_list, radar_list, gps):
        if self.image_encoder.normalize:
            image_list = [normalize_imagenet(img) for img in image_list]

        bz, _, h, w = lidar_list[0].shape
        img_ch = image_list[0].shape[1]
        lid_ch = lidar_list[0].shape[1]
        rad_ch = radar_list[0].shape[1]
        self.config.n_views = len(image_list) // self.config.seq_len

        image_tensor = torch.stack(image_list, dim=1).view(
            bz * self.config.n_views * self.config.seq_len, img_ch, h, w)
        lidar_tensor = torch.stack(lidar_list, dim=1).view(
            bz * self.config.seq_len, lid_ch, h, w)
        radar_tensor = torch.stack(radar_list, dim=1).view(
            bz * self.config.seq_len, rad_ch, h, w)

        # ResNet feature extraction (frozen layers handled externally)
        image_features = self.image_encoder.features.conv1(image_tensor)
        image_features = self.image_encoder.features.bn1(image_features)
        image_features = self.image_encoder.features.relu(image_features)
        image_features = self.image_encoder.features.maxpool(image_features)

        lidar_features = self.lidar_encoder._model.conv1(lidar_tensor)
        lidar_features = self.lidar_encoder._model.bn1(lidar_features)
        lidar_features = self.lidar_encoder._model.relu(lidar_features)
        lidar_features = self.lidar_encoder._model.maxpool(lidar_features)

        radar_features = self.radar_encoder._model.conv1(radar_tensor)
        radar_features = self.radar_encoder._model.bn1(radar_features)
        radar_features = self.radar_encoder._model.relu(radar_features)
        radar_features = self.radar_encoder._model.maxpool(radar_features)

        # Scale 1: 64-dim
        image_features = self.image_encoder.features.layer1(image_features)
        lidar_features = self.lidar_encoder._model.layer1(lidar_features)
        radar_features = self.radar_encoder._model.layer1(radar_features)
        ie1 = self.avgpool(image_features)
        le1 = self.avgpool(lidar_features)
        re1 = self.avgpool(radar_features)
        ge1 = self.vel_emb1(gps)
        io1, lo1, ro1, go1 = self.transformer1(ie1, le1, re1, ge1)
        image_features = image_features + F.interpolate(io1, scale_factor=8, mode='bilinear')
        lidar_features = lidar_features + F.interpolate(lo1, scale_factor=8, mode='bilinear')
        radar_features = radar_features + F.interpolate(ro1, scale_factor=8, mode='bilinear')

        # Scale 2: 128-dim
        image_features = self.image_encoder.features.layer2(image_features)
        lidar_features = self.lidar_encoder._model.layer2(lidar_features)
        radar_features = self.radar_encoder._model.layer2(radar_features)
        ie2 = self.avgpool(image_features)
        le2 = self.avgpool(lidar_features)
        re2 = self.avgpool(radar_features)
        ge2 = self.vel_emb2(go1)
        io2, lo2, ro2, go2 = self.transformer2(ie2, le2, re2, ge2)
        image_features = image_features + F.interpolate(io2, scale_factor=4, mode='bilinear')
        lidar_features = lidar_features + F.interpolate(lo2, scale_factor=4, mode='bilinear')
        radar_features = radar_features + F.interpolate(ro2, scale_factor=4, mode='bilinear')

        # Scale 3: 256-dim
        image_features = self.image_encoder.features.layer3(image_features)
        lidar_features = self.lidar_encoder._model.layer3(lidar_features)
        radar_features = self.radar_encoder._model.layer3(radar_features)
        ie3 = self.avgpool(image_features)
        le3 = self.avgpool(lidar_features)
        re3 = self.avgpool(radar_features)
        ge3 = self.vel_emb3(go2)
        io3, lo3, ro3, go3 = self.transformer3(ie3, le3, re3, ge3)
        image_features = image_features + F.interpolate(io3, scale_factor=2, mode='bilinear')
        lidar_features = lidar_features + F.interpolate(lo3, scale_factor=2, mode='bilinear')
        radar_features = radar_features + F.interpolate(ro3, scale_factor=2, mode='bilinear')

        # Scale 4: 512-dim
        image_features = self.image_encoder.features.layer4(image_features)
        lidar_features = self.lidar_encoder._model.layer4(lidar_features)
        radar_features = self.radar_encoder._model.layer4(radar_features)
        ie4 = self.avgpool(image_features)
        le4 = self.avgpool(lidar_features)
        re4 = self.avgpool(radar_features)
        ge4 = self.vel_emb4(go3)
        io4, lo4, ro4, go4 = self.transformer4(ie4, le4, re4, ge4)
        image_features = image_features + io4
        lidar_features = lidar_features + lo4
        radar_features = radar_features + ro4

        # Pool to vectors
        image_features = self.image_encoder.features.avgpool(image_features)
        image_features = torch.flatten(image_features, 1).view(bz, self.config.n_views * self.config.seq_len, -1)
        lidar_features = self.lidar_encoder._model.avgpool(lidar_features)
        lidar_features = torch.flatten(lidar_features, 1).view(bz, self.config.seq_len, -1)
        radar_features = self.radar_encoder._model.avgpool(radar_features)
        radar_features = torch.flatten(radar_features, 1).view(bz, self.config.seq_len, -1)
        gps_features = go4

        # Concatenate all modality tokens: (B, 5+5+5+2, 512)
        all_tokens = torch.cat([image_features, lidar_features, radar_features, gps_features], dim=1)

        # Cross-attention fusion instead of sum
        fused = self.cross_fusion(all_tokens)  # (B, 512)
        return fused, image_features, lidar_features, radar_features


class TransFuserV5(nn.Module):
    """
    v5: Lightweight TransFuser with cross-attention fusion + contrastive head.
    """
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.config = config
        self.encoder = EncoderV5(config).to(device)

        # Beam classifier
        self.join = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.resid_pdrop),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(config.resid_pdrop),
            nn.Linear(128, 64),
        ).to(device)

        # Contrastive projection heads (for InfoNCE alignment)
        self.proj_img = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(True), nn.Linear(256, 128)).to(device)
        self.proj_lid = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(True), nn.Linear(256, 128)).to(device)
        self.proj_rad = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(True), nn.Linear(256, 128)).to(device)

    def forward(self, image_list, lidar_list, radar_list, gps):
        fused, img_feat, lid_feat, rad_feat = self.encoder(
            image_list, lidar_list, radar_list, gps)
        beam_logits = self.join(fused)

        # Pool modality features for contrastive loss: mean over temporal dim
        img_pooled = img_feat.mean(dim=1)  # (B, 512)
        lid_pooled = lid_feat.mean(dim=1)
        rad_pooled = rad_feat.mean(dim=1)

        z_img = F.normalize(self.proj_img(img_pooled), dim=-1)
        z_lid = F.normalize(self.proj_lid(lid_pooled), dim=-1)
        z_rad = F.normalize(self.proj_rad(rad_pooled), dim=-1)

        return beam_logits, z_img, z_lid, z_rad

    def predict(self, image_list, lidar_list, radar_list, gps):
        """Inference-only: returns beam logits only."""
        fused, _, _, _ = self.encoder(image_list, lidar_list, radar_list, gps)
        return self.join(fused)


def freeze_backbone(model, freeze_layers='all'):
    """Freeze ResNet backbone layers. Only conv1 of lidar/radar stays trainable.
    Also sets frozen BN layers to eval mode to prevent running stats drift."""
    # Freeze image encoder (ResNet34)
    for name, param in model.encoder.image_encoder.features.named_parameters():
        if freeze_layers == 'all':
            param.requires_grad = False
        elif freeze_layers == 'early':
            if any(k in name for k in ['conv1', 'bn1', 'layer1', 'layer2']):
                param.requires_grad = False

    # Freeze lidar encoder (ResNet18) — keep conv1 trainable (different input channels)
    for name, param in model.encoder.lidar_encoder._model.named_parameters():
        if 'conv1' in name:
            continue  # keep conv1 trainable (1-channel input)
        if freeze_layers == 'all':
            param.requires_grad = False
        elif freeze_layers == 'early':
            if any(k in name for k in ['bn1', 'layer1', 'layer2']):
                param.requires_grad = False

    # Freeze radar encoder (ResNet18) — keep conv1 trainable
    for name, param in model.encoder.radar_encoder._model.named_parameters():
        if 'conv1' in name:
            continue
        if freeze_layers == 'all':
            param.requires_grad = False
        elif freeze_layers == 'early':
            if any(k in name for k in ['bn1', 'layer1', 'layer2']):
                param.requires_grad = False

    # Set frozen BN layers to eval mode to prevent running stats drift
    _freeze_bn(model.encoder.image_encoder.features, freeze_layers)
    _freeze_bn(model.encoder.lidar_encoder._model, freeze_layers)
    _freeze_bn(model.encoder.radar_encoder._model, freeze_layers)


def _freeze_bn(module, freeze_layers):
    """Set BN layers in frozen parts to eval mode."""
    for name, m in module.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if freeze_layers == 'all':
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
            elif freeze_layers == 'early':
                if any(k in name for k in ['bn1', 'layer1', 'layer2']):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False


class FrozenBNMixin:
    """Call this after model.train() to re-freeze BN layers."""
    @staticmethod
    def freeze_bn_eval(model):
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                if not any(p.requires_grad for p in m.parameters()):
                    m.eval()
