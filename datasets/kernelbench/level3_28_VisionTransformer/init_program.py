# EVOLVE-BLOCK-START
from utils import make_match_reference
from typing import Tuple

import torch
from torch import nn


def custom_kernel(data: Tuple[torch.Tensor, int, int, int, int, int, int, int, float, float]) -> torch.Tensor:
    img, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels, dropout, emb_dropout = data
    # Your optimized implementation

# Reference code in PyTorch
class Model(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_classes: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        channels: int = 3,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
    ):
        super().__init__()

        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
            ),
            num_layers=depth,
        )

        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        p = self.patch_size

        x = img.unfold(2, p, p).unfold(3, p, p).reshape(
            img.shape[0], -1, p * p * img.shape[1]
        )
        x = self.patch_to_embedding(x)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)


def ref_kernel(
    data: Tuple[
        torch.Tensor,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        float,
        float,
    ]
) -> torch.Tensor:
    (
        img,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels,
        dropout,
        emb_dropout,
    ) = data

    model = Model(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        channels=channels,
        dropout=dropout,
        emb_dropout=emb_dropout,
    ).to(img.device)

    return model(img)


def generate_input(
    batch_size: int,
    image_size: int,
    patch_size: int,
    num_classes: int,
    dim: int,
    depth: int,
    heads: int,
    mlp_dim: int,
    channels: int,
    dropout: float,
    emb_dropout: float,
    seed: int,
    distribution: str = "uniform",
    dtype: str = "float32",
) -> Tuple[
    torch.Tensor,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    float,
    float,
]:
    """
    Generate inputs for the Vision Transformer task.
    """
    device = "cuda"
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[dtype]

    shape = (batch_size, channels, image_size, image_size)

    if distribution == "uniform":
        img = torch.rand(shape, device=device, dtype=torch_dtype, generator=gen).contiguous()
    elif distribution == "normal":
        img = torch.randn(shape, device=device, dtype=torch_dtype, generator=gen).contiguous()
    elif distribution == "cauchy":
        img = torch.distributions.Cauchy(0, 2).sample(shape).to(
            device=device, dtype=torch_dtype
        ).contiguous()
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    return (
        img,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels,
        dropout,
        emb_dropout,
    )


check_implementation = make_match_reference(ref_kernel, rtol=1e-4, atol=1e-4)
# EVOLVE-BLOCK-END