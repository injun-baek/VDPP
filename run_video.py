import os
import cv2
import glob
import torch
import imageio
import argparse
import numpy as np

from tqdm import tqdm
import matplotlib.cm as cm
from vdpp.vdpp_model import VDPP
from external.dav2.depth_anything_v2.dpt import DepthAnythingV2


def save_video(frames, output_video_path, fps=10, is_depths=False, grayscale=False):
    writer = imageio.get_writer(output_video_path, fps=fps, macro_block_size=1, codec='libx264', ffmpeg_params=['-crf', '18'])
    if is_depths:
        colormap = cm.get_cmap("Spectral_r")(np.linspace(0, 1, 256))[:, :3]
        d_min, d_max = frames.min(), frames.max()
        for i in tqdm(range(frames.shape[0])):
            depth = frames[i]
            depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            depth_vis = (colormap[depth_norm] * 255).astype(np.uint8) if not grayscale else depth_norm
            writer.append_data(depth_vis)
    else:
        for i in tqdm(range(frames.shape[0])):
            writer.append_data(frames[i])

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VDPP')
    
    parser.add_argument('--dav2_model', default='vitl', type=str)
    parser.add_argument('--input-size', type=int, default=0)
    parser.add_argument('--checkpoint', type=str, default='checkpoints/vdpp.pth', help='Path to validation checkpoint file')

    parser.add_argument('--indir', default='./assets/SVD/', type=str)
    parser.add_argument('--outdir', type=str, default='./results/', help='Directory to save the output video')
    
    parser.add_argument('--downsize', default=False, dest='downsize', action='store_true', help='downsize input frames for faster inference')
    parser.add_argument('--grayscale', default=False, dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    dav2_model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    depth_anything = DepthAnythingV2(**dav2_model_configs[args.dav2_model])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.dav2_model}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    vdpp = VDPP(encoder="vits", features=64, out_channels=[48, 96, 192, 384]).to(DEVICE).eval()
    vdpp.load_state_dict(torch.load(args.checkpoint, map_location='cpu'), strict=True)

    filenames = glob.glob(os.path.join(args.indir, '**/*'), recursive=True)
    
    print(f'Found {len(filenames)} videos in the input path.')
    filenames.sort()
    os.makedirs(args.outdir, exist_ok=True)

    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_video = cv2.VideoCapture(filename)
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        frame_count = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if args.input_size == 0:
            inference_size = min(frame_height, frame_width)
        else:
            inference_size = args.input_size
        

        dav2_depths = []
        for i in tqdm(range(frame_count), desc="Processing I2D"):
            ret, raw_frame = raw_video.read()
            if not ret:
                break
            
            raw_frame = raw_frame[:, :, ::-1]  # BGR to RGB
            i2d_depth = depth_anything.infer_image(raw_frame, inference_size)
            i2d_depth = torch.from_numpy(i2d_depth).to(DEVICE)
            i2d_depth = (i2d_depth - i2d_depth.min()) / (i2d_depth.max() - i2d_depth.min())

            dav2_depths.append(i2d_depth)

        i2d_depth = torch.stack(dav2_depths, dim=0).unsqueeze(0)

        with torch.no_grad():
            ours_depth = vdpp.infer_video_depth(i2d_depth, downsize=args.downsize)
            ours_depth = (ours_depth - ours_depth.min()) / (ours_depth.max() - ours_depth.min())
            ours_depth = ours_depth.squeeze(0)
        
        ours_depth = ours_depth.cpu().numpy()
        i2d_depth = i2d_depth.squeeze(0).cpu().numpy()
        
        dav_output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '_i2d.mp4')
        ours_output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '_ours.mp4')
        
        save_video(i2d_depth, dav_output_path, fps=frame_rate, is_depths=True, grayscale=args.grayscale)
        save_video(ours_depth, ours_output_path, fps=frame_rate, is_depths=True, grayscale=args.grayscale)
        
