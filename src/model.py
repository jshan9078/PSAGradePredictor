import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50


# ---------------------------
# Utilities: Adapt first conv to 6 channels (LAB + Gx + Gy + Lap)
# ---------------------------
def _inflate_first_conv(m: nn.Conv2d, in_ch: int) -> nn.Conv2d:
    """
    Take a pretrained conv1 (3->64) and expand its input channels to in_ch by
    copying/averaging the RGB weights across new channels. Keeps fan-in scale.
    """
    assert m.in_channels in (1, 3)
    w = m.weight  # [out, in, k, k]
    if in_ch == m.in_channels:
        return m
    new = nn.Conv2d(in_ch, m.out_channels, kernel_size=m.kernel_size,
                    stride=m.stride, padding=m.padding, bias=(m.bias is not None))
    with torch.no_grad():
        if m.in_channels == 3:
            # average across RGB and tile
            mean = w.mean(dim=1, keepdim=True)  # [out,1,k,k]
            new_w = mean.repeat(1, in_ch, 1, 1) * (3.0 / in_ch)  # keep variance similar
        else:  # rare path (in=1 -> in_ch)
            new_w = w.repeat(1, in_ch, 1, 1) / in_ch
        new.weight.copy_(new_w)
        if m.bias is not None:
            new.bias.copy_(m.bias)
    return new


# ---------------------------
# CBAM: Channel + Spatial Attention (lightweight)
# ---------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]
        avg = F.adaptive_avg_pool2d(x, 1).flatten(1)  # [B,C]
        mx = F.adaptive_max_pool2d(x, 1).flatten(1)   # [B,C]
        w = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).unsqueeze(-1).unsqueeze(-1)
        return x * w


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7, use_rim_mask: bool = False):
        super().__init__()
        padding = (kernel_size - 1) // 2
        # If using rim mask, we concatenate it -> 3 channels instead of 2
        in_channels = 3 if use_rim_mask else 2
        self.use_rim_mask = use_rim_mask
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor, rim_mask: torch.Tensor = None) -> torch.Tensor:
        # Along channel dim: avg+max compressions
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)

        if self.use_rim_mask and rim_mask is not None:
            # Paper §5.3: Append binary rim mask to emphasize borders
            attn_input = torch.cat([avg, mx, rim_mask], dim=1)
        else:
            attn_input = torch.cat([avg, mx], dim=1)

        attn = torch.sigmoid(self.conv(attn_input))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_k: int = 7, use_rim_mask: bool = False):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_k, use_rim_mask)
        self.use_rim_mask = use_rim_mask

    def forward(self, x: torch.Tensor, rim_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x, rim_mask)
        return x


# ---------------------------
# Encoder wrappers
# ---------------------------
def build_resnet_encoder(depth: int = 18, in_ch: int = 6, pretrained: bool = True) -> nn.Module:
    if depth == 18:
        m = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    elif depth == 34:
        m = resnet34(weights="IMAGENET1K_V1" if pretrained else None)
    elif depth == 50:
        m = resnet50(weights="IMAGENET1K_V1" if pretrained else None)
    else:
        raise ValueError(f"ResNet depth {depth} not supported. Choose from [18, 34, 50].")

    # Replace conv1 to accept 6 channels
    m.conv1 = _inflate_first_conv(m.conv1, in_ch)
    # Strip classification head; keep features up to global pool
    feature_dim = m.fc.in_features
    m.fc = nn.Identity()
    return m, feature_dim


# ---------------------------
# Main Model
# ---------------------------
class DualBranchPSA(nn.Module):
    """
    Dual-branch architecture for PSA card grading.

    Front: ResNet-18/34/50 (default: ResNet-18, lighter for less critical features)
    Back : ResNet-18/34/50 (default: ResNet-34, deeper for damage/centering detection)
           + CBAM on high-level feature maps with optional rim mask
    Fusion: weighted concat z = [λ h_b ; (1−λ) h_f]
    Heads:
      - grade logits: 10-way classification
      - edge damage: sigmoid scalar
      - centering: 2-dim regression (cx, cy) from back branch only

    Supports flexible ResNet depths for experimentation:
    - Default: front_depth=18, back_depth=34 (baseline)
    - High-capacity: front_depth=34, back_depth=50 (if baseline QWK < 0.85)
    """
    def __init__(
        self,
        lambda_fusion: float = 0.7,
        in_channels: int = 6,
        front_depth: int = 18,
        back_depth: int = 34,
        pretrained: bool = True,
        dropout: float = 0.2,
        hidden: int = 512,
        use_rim_mask: bool = True,
        rim_mask_ratio: float = 0.07,
    ):
        super().__init__()
        assert 0.0 < lambda_fusion < 1.0
        self.lambda_fusion = lambda_fusion
        self.use_rim_mask = use_rim_mask
        self.rim_mask_ratio = rim_mask_ratio

        # Encoders
        self.front_enc, d_f = build_resnet_encoder(front_depth, in_channels, pretrained)
        self.back_enc, d_b = build_resnet_encoder(back_depth, in_channels, pretrained)

        # Store embedding dimensions for heads
        self.d_b = d_b
        self.d_f = d_f

        # Lightweight attention on the back branch: attach after the last residual block
        # We'll hook it by wrapping the last layer (layer4) with CBAM -> pool.
        # The channels at layer4 output depend on resnet depth:
        # - ResNet-18/34: 512 channels
        # - ResNet-50: 2048 channels
        layer4_channels = 2048 if back_depth == 50 else 512
        self.back_cbam = CBAM(channels=layer4_channels, reduction=16, spatial_k=7, use_rim_mask=use_rim_mask)

        # Cached rim mask (created on first forward pass)
        self.register_buffer('rim_mask', None)

        # Fusion MLP
        fused_dim = d_b + d_f
        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        # Heads
        self.head_grade = nn.Linear(hidden, 10)
        self.head_edge = nn.Linear(hidden, 1)
        # Paper §5.4.4: Centering head uses back branch only (h_b)
        self.head_center = nn.Linear(d_b, 2)

        # Init
        nn.init.zeros_(self.head_edge.bias)
        nn.init.zeros_(self.head_center.bias)

    def _create_rim_mask(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Create or retrieve cached rim mask for CBAM spatial attention."""
        if self.rim_mask is None or self.rim_mask.shape[-2:] != (height, width):
            mask = torch.zeros((1, 1, height, width), dtype=torch.float32, device=device)
            rim_h = int(height * self.rim_mask_ratio)
            rim_w = int(width * self.rim_mask_ratio)

            # Mark outer rim as 1
            mask[:, :, :rim_h, :] = 1  # top
            mask[:, :, -rim_h:, :] = 1  # bottom
            mask[:, :, :, :rim_w] = 1  # left
            mask[:, :, :, -rim_w:] = 1  # right

            self.rim_mask = mask
        return self.rim_mask

    # ---- custom forward for back to insert CBAM on high-level maps ----
    def _forward_back_feature(self, x: torch.Tensor) -> torch.Tensor:
        # follow torchvision ResNet forward with cbam before pooling
        m = self.back_enc
        x = m.conv1(x); x = m.bn1(x); x = m.relu(x); x = m.maxpool(x)
        x = m.layer1(x); x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)

        # Apply CBAM with optional rim mask
        if self.use_rim_mask:
            _, _, h, w = x.shape
            rim_mask = self._create_rim_mask(h, w, x.device)
            # Expand to batch size
            rim_mask = rim_mask.expand(x.size(0), -1, -1, -1)
            x = self.back_cbam(x, rim_mask)
        else:
            x = self.back_cbam(x)

        x = m.avgpool(x)  # [B, C, 1, 1]
        x = torch.flatten(x, 1)  # [B, C]
        return x

    def _forward_front_feature(self, x: torch.Tensor) -> torch.Tensor:
        m = self.front_enc
        x = m.conv1(x); x = m.bn1(x); x = m.relu(x); x = m.maxpool(x)
        x = m.layer1(x); x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)
        x = m.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(
        self,
        front: torch.Tensor,   # [B, 6, H, W]
        back: torch.Tensor,    # [B, 6, H, W]
        return_probs: bool = False,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        h_f = self._forward_front_feature(front)   # [B, d_f]
        h_b = self._forward_back_feature(back)     # [B, d_b]

        # Asymmetric weighting by λ on embeddings (not simple scalar mult before concat;
        # we scale then concat so the MLP "sees" the weighting directly.)
        z = torch.concat([self.lambda_fusion * h_b, (1.0 - self.lambda_fusion) * h_f], dim=1)
        z = self.fuse(z)

        logits = self.head_grade(z)                         # [B, 10]
        edge_logit = self.head_edge(z).squeeze(-1)          # [B]
        # Paper §5.4.4: Centering uses back branch only (geometric info)
        center = self.head_center(h_b)                      # [B, 2]

        out = {"logits": logits, "edge_logit": edge_logit, "center": center}

        if return_probs:
            probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
            out["probs"] = probs
        return out
