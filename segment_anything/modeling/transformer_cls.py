  
  
  
  

import torch
from torch import Tensor, nn

import math
from typing import Tuple, Type

from .common import MLPBlock



class Classifiy_TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        self.layers.append(
            TwoWayAttentionBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                activation=activation,
                depth=self.depth,
                attention_downsample_rate=attention_downsample_rate,
                )
            )  
  
  
  


    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,  
        token_cls_bl_embedding: Tensor,
        token_cls_fh_embedding: Tensor,
        token_cls_sd_embedding: Tensor,
        bbox_prompt_embedding: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """  
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)  
        image_pe = image_pe.flatten(2).permute(0, 2, 1)  

        queries = image_embedding  
        keys_bbox_token = bbox_prompt_embedding  
  
        keys_cls_bl_token = token_cls_bl_embedding  
        keys_cls_fh_token = token_cls_fh_embedding
        keys_cls_sd_token = token_cls_sd_embedding  
        for layer in self.layers:
            keys_bl, keys_fh, keys_sd = layer(
                queries=queries,
                keys_bbox_token=keys_bbox_token,
                keys_cls_bl_token=keys_cls_bl_token,
                keys_cls_fh_token=keys_cls_fh_token,
                keys_cls_sd_token=keys_cls_sd_token,
                query_pe=image_pe,
                key_pe=bbox_prompt_embedding,
            )
        return keys_bl, keys_fh, keys_sd


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        depth: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.depth = depth  
        self.self_attn = Attention(embedding_dim, num_heads)
        self.self_attn_bl = Attention(embedding_dim, num_heads)
        self.self_attn_fh = Attention(embedding_dim, num_heads)
        self.self_attn_sd = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)  
        self.cross_attn_image_to_token = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)  
  
  
  
        self.cross_attn_token_to_image_bl = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.cross_attn_token_to_image_fh = Attention(embedding_dim, num_heads,
                                                      downsample_rate=attention_downsample_rate)
        self.cross_attn_token_to_image_sd = Attention(embedding_dim, num_heads,
                                                      downsample_rate=attention_downsample_rate)
        self.norm4 = nn.LayerNorm(embedding_dim)

    def forward(
            self,
            queries: Tensor,
            keys_bbox_token: Tensor,  
            keys_cls_bl_token: Tensor,
            keys_cls_fh_token: Tensor,
            keys_cls_sd_token: Tensor,
            query_pe: Tensor,
            key_pe: Tensor,
    ) -> Tuple[Tensor, Tensor]:  
        k_bbox = keys_bbox_token  
        attn_out = self.self_attn(q=k_bbox, k=k_bbox, v=k_bbox)
        keys_bbox = k_bbox + attn_out  
        keys_bbox = self.norm1(keys_bbox)  
  
  
        q = queries + query_pe  
        k = keys_bbox + key_pe  
        attn_out = self.cross_attn_image_to_token(q=q, k=k, v=keys_bbox)  
        queries = queries + attn_out
        queries = self.norm2(queries)
        q = queries + query_pe  
        k_bl = keys_cls_bl_token  
        k_fh = keys_cls_fh_token  
        k_sd = keys_cls_sd_token  
        attn_out_k_bl = self.self_attn_bl(q=k_bl, k=k_bl, v=k_bl)  
        attn_out_k_fh = self.self_attn_fh(q=k_fh, k=k_fh, v=k_fh)  
        attn_out_k_sd = self.self_attn_sd(q=k_sd, k=k_sd, v=k_sd)  
        k_bl = k_bl + attn_out_k_bl
        k_fh = k_fh + attn_out_k_fh
        k_sd = k_sd + attn_out_k_sd
        attn_out_bl = self.cross_attn_token_to_image_bl(q=k_bl, k=q, v=queries)  
        attn_out_fh = self.cross_attn_token_to_image_fh(q=k_fh, k=q, v=queries)
        attn_out_sd = self.cross_attn_token_to_image_sd(q=k_sd, k=q, v=queries)
        keys_bl = k_bl + attn_out_bl
        keys_fh = k_fh + attn_out_fh
        keys_sd = k_sd + attn_out_sd
        keys_bl = self.norm4(keys_bl)  
        keys_fh = self.norm4(keys_fh)  
        keys_sd = self.norm4(keys_sd)  

        return keys_bl, keys_fh, keys_sd  


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,  
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape  
        x = x.transpose(1, 2)  
        return x.reshape(b, n_tokens, n_heads * c_per_head)  

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:  
        q = self.q_proj(q)  
        k = self.k_proj(k)
        v = self.v_proj(v)  
        q = self._separate_heads(q, self.num_heads)  
        k = self._separate_heads(k, self.num_heads)  
        v = self._separate_heads(v, self.num_heads)  
        _, _, _, c_per_head = q.shape  
        attn = q @ k.permute(0, 1, 3, 2)  
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)  
        out = attn @ v
        out = self._recombine_heads(out)  
        out = self.out_proj(out)

        return out  
