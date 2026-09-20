"""
U-Net architecture for road segmentation.

Chosen for Part 1 because it is simple, well-understood, and computationally
manageable for a first working hackathon pipeline. The model is deliberately
kept plain (no attention blocks, no pretrained backbone) so that a more
sophisticated architecture (e.g. DeepLabV3+) can be swapped in later without
touching the dataset, training, or inference code -- all of which depend
only on the common interface: a model that maps (B, in_channels, H, W) to
(B, out_channels, H, W) logits.
"""

from typing import List

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two consecutive (Conv2d -> BatchNorm -> ReLU) blocks.

    This is the basic building block used in both the encoder and decoder
    of the U-Net.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """A compact U-Net for binary road segmentation.

    Architecture:
        Encoder: progressively downsamples the input, doubling channel
            depth at each stage, extracting increasingly abstract spatial
            features via max pooling.
        Bottleneck: the deepest, most compressed representation of the
            image, sitting between encoder and decoder.
        Skip connections: feature maps from each encoder stage are
            concatenated into the matching decoder stage, preserving
            fine-grained spatial detail (e.g. thin road edges) that would
            otherwise be lost to downsampling.
        Decoder: progressively upsamples back to the original resolution,
            fusing skip-connection features with upsampled features at
            each stage.
        Segmentation head: a final 1x1 convolution that maps decoder
            features to `out_channels` raw logits (no sigmoid applied --
            downstream loss/inference functions handle activation).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: List[int] = (64, 128, 256, 512),
    ) -> None:
        super().__init__()
        features = list(features)

        # --- Encoder ---
        self.encoder_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_channels = in_channels
        for feature in features:
            self.encoder_blocks.append(DoubleConv(prev_channels, feature))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_channels = feature

        # --- Bottleneck ---
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # --- Decoder ---
        self.upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        reversed_features = list(reversed(features))
        prev_channels = features[-1] * 2
        for feature in reversed_features:
            self.upconvs.append(
                nn.ConvTranspose2d(prev_channels, feature, kernel_size=2, stride=2)
            )
            # After upsampling, the feature map is concatenated with the
            # matching encoder skip connection, doubling channel count.
            self.decoder_blocks.append(DoubleConv(feature * 2, feature))
            prev_channels = feature

        # --- Segmentation head ---
        # 1x1 conv projecting to the desired number of output classes
        # (1 for binary road-vs-background, as raw logits).
        self.segmentation_head = nn.Conv2d(features[0], out_channels, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming initialization for conv layers, standard for ReLU nets."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        # Encoder pass: store pre-pool features for skip connections.
        for encoder_block, pool in zip(self.encoder_blocks, self.pools):
            x = encoder_block(x)
            skip_connections.append(x)
            x = pool(x)

        # Bottleneck.
        x = self.bottleneck(x)

        # Decoder pass: upsample, concatenate matching skip connection,
        # then refine with a DoubleConv block. Skip connections are
        # consumed in reverse (deepest encoder stage pairs with the
        # shallowest decoder stage).
        skip_connections = skip_connections[::-1]
        for idx, (upconv, decoder_block) in enumerate(zip(self.upconvs, self.decoder_blocks)):
            x = upconv(x)
            skip = skip_connections[idx]
            if x.shape[-2:] != skip.shape[-2:]:
                # Guard against off-by-one size mismatches from odd input
                # dimensions by cropping/padding to match the skip tensor.
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = decoder_block(x)

        return self.segmentation_head(x)


def build_model(in_channels: int = 3, out_channels: int = 1) -> UNet:
    """Factory function used by training/inference to instantiate the model.

    Centralizing model construction here makes it a single point of change
    if the architecture is swapped (e.g. for DeepLabV3+) in the future.
    """
    return UNet(in_channels=in_channels, out_channels=out_channels)
