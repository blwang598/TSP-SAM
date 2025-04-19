import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Type, Union, Sequence, Optional
import numpy as np
from .common import LayerNorm2d

def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.initialize_se()
    def initialize_se(self):
        for b_layer in self.fc:
            if type(b_layer) == nn.Linear:
                torch.nn.init.xavier_normal_(b_layer.weight)    

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class UnetResSEBlock(nn.Module):
    """
    A skip-connection based module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.
    """

    def __init__(
            self,    
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            norm_name: Union[Tuple, str] = "instance",
            act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            dropout=None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.lrelu = nn.LeakyReLU(0.01)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.norm3 = nn.InstanceNorm2d(out_channels)
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True

        self.se = SELayer(channel=out_channels)

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.se(out)
        if self.downsample:
            residual = self.conv3(residual)
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out


class PreUpBlock(nn.Module):
    def __init__(
            self,    
            in_channels: int,
            out_channels: int,
            upsample_kernel_size: int,
            norm_name: Union[Tuple, str] = 'instance',
            num_layer: int = 1,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            stride: convolution stride.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
        """

        super().__init__()

        self.input_channels = in_channels
        self.output_channels = out_channels

        self.block_init = UnetResSEBlock(    
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

        self.residual_block = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=upsample_kernel_size,
                        stride=upsample_kernel_size),
                    UnetResSEBlock(    
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        norm_name=norm_name,
                    ),
                )
                for i in range(num_layer)
            ]
        )

    def forward(self, x):
        x = self.block_init(x)    
        for blk in self.residual_block:
            x = blk(x)
        return x


class UpBlock(nn.Module):    
    def __init__(
            self,    
            in_channels: int,
            out_channels: int,
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str] = 'instance',
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.transp_conv = nn.ConvTranspose2d(
            in_channels=self.input_channels,
            out_channels=self.output_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride
            )

        self.res_block = UnetResSEBlock(    
            self.output_channels + self.output_channels,
            self.output_channels,
            kernel_size=3,
            stride=1,    
        )

    def forward(self, inp, skip):    
        inp = self.transp_conv(inp)    
        out = torch.cat((inp, skip), dim=1)    
        out = self.res_block(out)    
        return out


class TSPGenerator(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,    
            act_layer: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        kernel_size = [7, 5, 3, 3, 1]
        self.conv1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=kernel_size[0], stride=2, padding=kernel_size[0]//2)
        self.conv2 = nn.Conv2d(in_channels//2, in_channels//4, kernel_size=kernel_size[1], stride=2, padding=kernel_size[1]//2)
        self.conv3 = nn.Conv2d(in_channels//4, in_channels//4, kernel_size=kernel_size[2], stride=2, padding=kernel_size[2]//2)
        self.conv4 = nn.Conv2d(in_channels//4, in_channels//4, kernel_size=kernel_size[3], stride=2, padding=kernel_size[3]//2)
        self.conv5 = nn.Conv2d(in_channels//4, out_channels, kernel_size=kernel_size[4], stride=1, padding=0)
        self.norm1 = nn.InstanceNorm2d(in_channels//2)
        self.norm2 = nn.InstanceNorm2d(in_channels//4)
        self.norm3 = nn.InstanceNorm2d(in_channels//4)
        self.norm4 = nn.InstanceNorm2d(in_channels//4)
        self.norm5 = nn.InstanceNorm2d(out_channels)
        self.act = act_layer()

    def forward(self, fused_embedding):
        x = fused_embedding
        out = self.act(self.norm1(self.conv1(x)))    
        out = self.act(self.norm2(self.conv2(out)))    
        out = self.act(self.norm3(self.conv3(out)))    
        out = self.act(self.norm4(self.conv4(out)))    
        out = self.norm5(self.conv5(out))    
        tsp_tokens = out.view(out.shape[0], self.out_channels, -1)
        return tsp_tokens


class FuseTSPG(nn.Module):
    def __init__(
            self,
            endoder_transformer_dim: int = 768,
            upsample_transformer_dim: int = 256,
            sam_features_length: int = 3,
    ):
        super().__init__()
        self.sam_features_length = sam_features_length    
        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.seg_outputs = []
        self.encoder_embed_size = [int(upsample_transformer_dim // 2 ** i) for i in range(sam_features_length)]    
        self.in_channels_list = [320, 160, 128]
        self.num_layers_list = [0, 1, 1]

        for d in range(self.sam_features_length):    
    
            in_channels = self.in_channels_list[d]    
            out_channels = self.encoder_embed_size[d]    
            num_layer = self.num_layers_list[d]    
            upsample_kernel_size = 2    
            self.conv_blocks_context.append(
                PreUpBlock(    
                    in_channels=in_channels,
                    out_channels=out_channels,
                    upsample_kernel_size=upsample_kernel_size,    
                    num_layer=num_layer
                )
            )

        for d in range(self.sam_features_length - 1):
            in_channels = self.encoder_embed_size[d]
            out_channels = self.encoder_embed_size[d + 1]
            upsample_kernel_size = 2
            self.conv_blocks_localization.append(
                UpBlock(    
                    in_channels=in_channels,
                    out_channels=out_channels,
                    upsample_kernel_size=upsample_kernel_size
                )
            )

        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(
                nn.Conv2d(
                    self.conv_blocks_localization[ds].output_channels, 1,
                    1, 1, 0, 1, 1, False
                )
            )
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.tspg_fusion = TSPGenerator(
            in_channels=self.encoder_embed_size[-1],
            out_channels=3,
        )    
        self.apply(_init_vit_weights)

    def forward(
            self,
            image_embeddings_blocks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:    
        skips = []
        for d in range(len(self.conv_blocks_context)):    
            embed = self.conv_blocks_context[d](image_embeddings_blocks[-(d + 1)])    
            skips.append(embed)    
    
    
        for u in range(len(self.conv_blocks_localization)):
            if u == 0:
                enc_x = skips[0]    
                dec_x = skips[1]    
            else:
                dec_x = skips[u + 1]    
            enc_x = self.conv_blocks_localization[u](enc_x, dec_x)
        task_prompts = self.tspg_fusion(enc_x)    
        return task_prompts

