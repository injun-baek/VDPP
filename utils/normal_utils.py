import torch
import torch.nn.functional as F

def normal_vector(
    img: torch.Tensor,
    normalize_kernel: bool = True,
    scale_xy: float = 1.0,
    scale_z: float = 1.0,
    eps: float = 1e-8
) -> torch.Tensor:
    Ix, Iy = sobel_ix_iy(img, normalize_kernel=normalize_kernel)  # (B,S,1,Y,X)

    nx = -scale_xy * Ix
    ny = -scale_xy * Iy
    nz =  scale_z * torch.ones_like(Ix)

    n = torch.cat([nx, ny, nz], dim=2)  # (B,S,3,Y,X)
    norm = torch.sqrt((n ** 2).sum(dim=2, keepdim=True) + eps)
    n = n / norm
    return n


def sobel_ix_iy(img: torch.Tensor, normalize_kernel: bool = True):
    """
    img: (B, S, 1, Y, X) float tensor
    return : Ix, Iy (B, S, 1, Y, X)
    """
    assert img.dim() == 5 and img.size(2) == 1, "input is expected (B,S,1,Y,X) shape"
    B, S, _, Y, X = img.shape

    kx = torch.tensor([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]], dtype=img.dtype, device=img.device)
    ky = torch.tensor([[1,  2,  1],
                    [0,  0,  0],
                    [-1, -2, -1]], dtype=img.dtype, device=img.device)
    if normalize_kernel:
        kx = kx / 8.0
        ky = ky / 8.0

    kx = kx.view(1, 1, 3, 3)
    ky = ky.view(1, 1, 3, 3)

    x = img.reshape(B * S, 1, Y, X)
    x_pad = F.pad(x, (1, 1, 1, 1), mode='reflect')

    Ix = F.conv2d(x_pad, kx)
    Iy = F.conv2d(x_pad, ky)

    Ix = Ix.view(B, S, 1, Y, X)
    Iy = Iy.view(B, S, 1, Y, X)
    return Ix, Iy