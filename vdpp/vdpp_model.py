# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from utils.normal_utils import normal_vector

import torch
import torch.nn.functional as F
import torch.nn as nn

from vdpp.dinov2 import DINOv2
from vdpp.dpt_temporal import DPTHeadTemporal

def compute_scale_and_shift(prediction, target, mask):
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid]
                  * b_1[valid]) / (det[valid] + 1e-6)
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid]
                  * b_1[valid]) / (det[valid] + 1e-6)

    return x_0, x_1


def make_multiple_of(size, multiple=14):
    return int(((size + multiple - 1) // multiple) * multiple)

class TanhToExp(nn.Module):
    def __init__(self, max_log_scale: float = 0.2):
        super().__init__()
        self.max_log_scale = max_log_scale
    def forward(self, x):
        return torch.exp(torch.tanh(x) * self.max_log_scale)

class ZeroConv(nn.Conv2d):
    def __init__(self, in_ch, out_ch, kernel_size=1, padding=0, bias=True):
        super().__init__(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=bias)
        nn.init.zeros_(self.weight)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)

class GlobalQuantilePool2d(nn.Module):
    def __init__(self, q: float = 0.5):
        super().__init__()
        self.q = q
    def forward(self, x):
        try:
            return torch.quantile(x, self.q, dim=(-2, -1), keepdim=True)
        except TypeError:
            N, C, H, W = x.shape
            return torch.quantile(x.view(N, C, H*W), self.q, dim=-1, keepdim=True).unsqueeze(-1)

class GlobalScaleHead(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, max_log_scale=0.2):
        super().__init__()
        self.feat = nn.Sequential(
            GlobalQuantilePool2d(0.5),
            ZeroConv(in_channels, out_channels, kernel_size=1, padding=0, bias=True)
        )
        self.to_scale = TanhToExp(max_log_scale=max_log_scale)

    def forward(self, x):
        g = self.feat(x)
        s = self.to_scale(g)
        return s

class VDPP(nn.Module):
    def __init__(
        self,
        encoder='vits',
        features=64,
        out_channels=[48, 96, 192, 384],
        use_bn=False,
        use_clstoken=False,
        num_frames=32,
        max_depth = 1.0,
        pe='ape',
    ):
        super(VDPP, self).__init__()

        self.num_frames = num_frames
        self.infer_overlap_size = 4
        self.encoder_size = encoder
        self.features = features
        self.out_channels = out_channels
        self.max_depth = max_depth
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23]
        }
        
        self.pretrained = DINOv2(model_name=self.encoder_size)
        self.scale_head = GlobalScaleHead(in_channels=1, out_channels=1, max_log_scale=1.0)
        self.temporal_head = DPTHeadTemporal(self.pretrained.embed_dim, self.features, use_bn, out_channels=self.out_channels, use_clstoken=use_clstoken, num_frames=num_frames, pe=pe)

    def forward(self, input_depth, downsize=True):
        B, S, H_orig, W_orig = input_depth.shape
        
        input_depth = input_depth / self.max_depth
        
        input_depth_bs1hw = input_depth.unsqueeze(2).flatten(0, 1)
        scale_bs11 = self.scale_head(input_depth_bs1hw).unflatten(0, (B, S)).squeeze(2)
        input_depth = input_depth * scale_bs11
        
        if downsize:
            MIN_SIZE_OF_FRAME = 224
            downsize_height = max(make_multiple_of(H_orig / 2, 14), MIN_SIZE_OF_FRAME)
            downsize_width = max(make_multiple_of(W_orig / 2, 14), MIN_SIZE_OF_FRAME)
            resize_depth = F.interpolate(input_depth, size=(downsize_height, downsize_width), mode='bilinear', align_corners=True)
        else:
            resize_depth = input_depth

        _, _, H, W = resize_depth.shape
        depth_expanded = resize_depth.unsqueeze(2)
        normal = normal_vector(depth_expanded)
        depth_expanded = torch.cat([depth_expanded, normal[:, :, :2, :, :]], dim=2)
            
        depth_reshaped = depth_expanded.view(B * S, 3, H, W)
        
        patch_h, patch_w = H // 14, W // 14
        features = self.pretrained.get_intermediate_layers(depth_reshaped, self.intermediate_layer_idx[self.encoder_size], return_class_token=True)
        depth = self.temporal_head(features, patch_h, patch_w, S)[0]
        depth = F.interpolate(depth, size=(H_orig, W_orig), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        output_depth = depth.squeeze(1).unflatten(0, (B, S))
            
        output_depth = input_depth + output_depth
                        
        return output_depth * self.max_depth
    
    @torch.no_grad()
    def infer_video_depth(self, frames, infer_frame=32, downsize=True):
        B, S, frame_height, frame_width = frames.shape
        if B != 1:  # TODO: support B > 1 cases
            raise Exception("Batch size must be 1 for inference.")
        resize_height = make_multiple_of(frame_height, 14)
        resize_width = make_multiple_of(frame_width, 14)
        
        if (resize_height, resize_width) != (frame_height, frame_width):
            frames = frames.view(B * S, 1, frame_height, frame_width)
            resized_frames = F.interpolate(frames, size=(resize_height, resize_width), mode="bilinear", align_corners=True)
            resized_frames = resized_frames.view(B, S, resize_height, resize_width)
            frames = resized_frames

        stride = infer_frame - self.infer_overlap_size
        mini_batch_sf = 0
        mini_batch_ef = min(infer_frame, S)
        mini_batch_output_depths = self(frames[:, mini_batch_sf:mini_batch_ef, ...], downsize=downsize)
        processed_depths = [mini_batch_output_depths]
        
        while mini_batch_ef < S:
            mini_batch_sf += stride
            mini_batch_ef = min(mini_batch_sf + infer_frame, S)
            mini_batch_frames = frames[:, mini_batch_sf:mini_batch_ef, :, :]
            mini_batch_output_depths = self(mini_batch_frames, downsize=downsize)
            
            prev_batch_overlap = processed_depths[-1][:, -self.infer_overlap_size:, ...]
            scale, shift = compute_scale_and_shift(mini_batch_output_depths[:, :self.infer_overlap_size, ...].flatten(1, 2), prev_batch_overlap.flatten(1, 2), torch.ones_like(prev_batch_overlap.flatten(1, 2)))
            affined_mini_batch_output_depths = mini_batch_output_depths * scale.view(1, 1, 1, 1) +  shift.view(1, 1, 1, 1)

            processed_depths.append(affined_mini_batch_output_depths[:, self.infer_overlap_size:, ...])

        output_depths = torch.cat(processed_depths, dim=1)
        
        if (resize_height, resize_width) != (frame_height, frame_width):
            output_depths = output_depths.view(B * S, 1, resize_height, resize_width)
            output_origin_size_depth = F.interpolate(output_depths, size=(frame_height, frame_width), mode="bilinear", align_corners=True)
            output_origin_size_depth = output_origin_size_depth.view(B, S, frame_height, frame_width)
            output_depths = output_origin_size_depth
        return output_depths
