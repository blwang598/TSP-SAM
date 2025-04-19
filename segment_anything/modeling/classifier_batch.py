import torch
from torch import nn


from typing import List, Tuple, Type


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


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        num_class: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp_dim = mlp_dim
        self.num_class = num_class
        self.create_bottleneck_layer()
        self.create_classifier(depth=2)
        self.act = act()

    def create_bottleneck_layer(self):
        bottleneck_layer_list = [
            nn.Linear(self.embedding_dim, int(2*self.embedding_dim)),
            nn.BatchNorm1d(int(2*self.embedding_dim)),
            nn.ReLU(),   
        ]
        self.bottleneck_layer = nn.Sequential(*bottleneck_layer_list)
        self.initialize_bottleneck()

    def create_classifier(self, depth):
        layer_list = []   
        input_size = int(2*self.embedding_dim)
        width = input_size // depth
        for i in range(depth):
            layer_list.append(nn.Linear(input_size, width))
            layer_list.append(nn.ReLU())   
            input_size = width

        layer_list.append(nn.Linear(width, self.num_class))
        self.classifier = nn.Sequential(*layer_list)
        self.initialize_classifier()

    def initialize_classifier(self):
        self.xavier_initialization(self.classifier)

    def xavier_initialization(self, layers):
        for layer in layers:
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_normal_(layer.weight)
                layer.bias.data.fill_(0.0)


    def initialize_bottleneck(self):
        self.xavier_initialization(self.bottleneck_layer)


    def forward(self, x: torch.Tensor):
        x = self.bottleneck_layer(x)
        x_out = x
        x = self.classifier(x)
        return x, x_out

class CombinedClassifier_batch_fore(nn.Module):
    def __init__(
            self,
            transformer: nn.Module,
            transformer_dim: int,
    ):
        super().__init__()
        self.cls_token_bingli = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        self.cls_token_fenhua = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        self.cls_token_shendu = nn.Parameter(torch.zeros(1, 1, transformer_dim))   
   
   
        self.transformer = transformer
        self.transformer_dim = transformer_dim   
        nn.init.trunc_normal_(self.cls_token_bingli, std=0.02)
        nn.init.trunc_normal_(self.cls_token_fenhua, std=0.02)
        nn.init.trunc_normal_(self.cls_token_shendu, std=0.02)   
    def forward(self, image_embeddings, image_pe, task_prompt_embeddings, sparse_prompt_embeddings, dense_prompt_embeddings):   
   
        task_prompt_token_bl = task_prompt_embeddings[:, 0, :]   
        task_prompt_token_fh = task_prompt_embeddings[:, 1, :]   
        task_prompt_token_sd = task_prompt_embeddings[:, 2, :]   
        task_prompt_token_bl = task_prompt_token_bl.unsqueeze(1)
        task_prompt_token_fh = task_prompt_token_fh.unsqueeze(1)   
        task_prompt_token_sd = task_prompt_token_sd.unsqueeze(1)
        cls_token_bingli = self.cls_token_bingli.expand(image_embeddings.shape[0], -1, -1)   
        cls_token_fenhua = self.cls_token_fenhua.expand(image_embeddings.shape[0], -1, -1)
        cls_token_shendu = self.cls_token_shendu.expand(image_embeddings.shape[0], -1, -1)
        token_cls_bl = torch.cat([cls_token_bingli, task_prompt_token_bl], dim=1)   
        token_cls_fh = torch.cat([cls_token_fenhua, task_prompt_token_fh], dim=1)
        token_cls_sd = torch.cat([cls_token_shendu, task_prompt_token_sd], dim=1)

        tokens_bbox_prompt = sparse_prompt_embeddings   
        src = image_embeddings + dense_prompt_embeddings   
        pos_src = torch.repeat_interleave(image_pe, sparse_prompt_embeddings.shape[0], dim=0)   

        cls_pred_bl, cls_pred_fh, cls_pred_sd = self.transformer(src, pos_src, token_cls_bl, token_cls_fh, token_cls_sd, tokens_bbox_prompt)   

        return cls_pred_bl[:, 0, :].squeeze(), cls_pred_fh[:, 0, :].squeeze(), cls_pred_sd[:, 0, :].squeeze()   
   
   
   

class CombinedClassifier_batch_aft_bl(nn.Module):
    def __init__(
            self,
            transformer_dim: int,
            num_class: int,
            mlp_dim: int = 256,
            activation: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.mlp_bl = MLPBlock(transformer_dim, mlp_dim, num_class, activation)   

    def forward(self, cls_pred_bingli):
        logits_bl = torch.softmax(self.mlp_bl(cls_pred_bingli)[0], dim=1)
        return logits_bl


class CombinedClassifier_batch_aft_fh(nn.Module):
    def __init__(
            self,
            transformer_dim: int,
            num_class: int=3,
            mlp_dim: int = 256,
            activation: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.mlp_fh = MLPBlock(int(transformer_dim), mlp_dim, num_class, activation)   

    def forward(self, cls_pred_fenhua):
        logits_fh = torch.softmax(self.mlp_fh(cls_pred_fenhua)[0], dim=1)
        return logits_fh


class CombinedClassifier_batch_aft_sd(nn.Module):
    def __init__(
            self,
            transformer_dim: int,
            num_class: int=3,
            mlp_dim: int = 256,
            activation: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.mlp_sd = MLPBlock(int(transformer_dim), mlp_dim, num_class, activation)   

    def forward(self, cls_pred_shendu):
        logits_sd = torch.softmax(self.mlp_sd(cls_pred_shendu)[0], dim=1)
        return logits_sd