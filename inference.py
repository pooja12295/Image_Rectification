import argparse
from pathlib import Path

import numpy as np
import scipy.io as scio
import skimage.io as io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from Guided_Rectification_Network_M_model import ClassNet, DecoderNet, EncoderNet

CLASS_NAMES = ['barrel', 'pincushion', 'rotation', 'shear', 'projective', 'wave']

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Lambda(lambda im: im.convert("RGB")),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def load_models(model_dir: Path, device: torch.device):
    model_en = EncoderNet([1, 1, 1, 1, 2])
    model_de = DecoderNet([1, 1, 1, 1, 2])
    model_class = ClassNet()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_en = nn.DataParallel(model_en)
        model_de = nn.DataParallel(model_de)
        model_class = nn.DataParallel(model_class)

    def _load(model, filename):
        state = torch.load(model_dir / filename, map_location=device)
        # Models were trained with DataParallel; strip the "module." prefix if present.
        if any(k.startswith('module.') for k in state.keys()):
            state = {k.replace('module.', '', 1): v for k, v in state.items()}
        model.load_state_dict(state)

    _load(model_en, 'model_en.pkl')
    _load(model_de, 'model_de.pkl')
    _load(model_class, 'model_class.pkl')

    model_en = model_en.to(device)
    model_de = model_de.to(device)
    model_class = model_class.to(device)

    model_en.eval()
    model_de.eval()
    model_class.eval()

    return model_en, model_de, model_class


def rectify_image(disimgs: torch.Tensor, flow_output: torch.Tensor) -> np.ndarray:
    """
    disimgs: (1, 3, H, W) normalized [-1, 1]
    flow_output: (1, 2, H, W) forward flow (u, v)
    """
    _, _, H, W = disimgs.shape
    u = flow_output[:, 0]
    v = flow_output[:, 1]

    # base grid of destination pixel indices
    xs = torch.arange(W, device=disimgs.device).view(1, 1, W).expand(1, H, W)
    ys = torch.arange(H, device=disimgs.device).view(1, H, 1).expand(1, H, W)

    # map to source coords (backward mapping for grid_sample)
    src_x = xs - u
    src_y = ys - v

    grid_x = 2.0 * src_x / max(W - 1, 1) - 1.0
    grid_y = 2.0 * src_y / max(H - 1, 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)  # (1, H, W, 2)

    rectified = F.grid_sample(
        disimgs,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True,
    )
    rectified = rectified.squeeze(0)
    rectified = rectified.mul(0.5).add(0.5).clamp(0, 1)  # denorm to [0,1]
    rectified = rectified.permute(1, 2, 0).cpu().numpy()
    rectified = (rectified * 255.0).round().astype(np.uint8)
    return rectified


def run_inference(input_dir: Path, output_dir: Path, model_dir: Path, save_rectified: bool = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_en, model_de, model_class = load_models(model_dir, device)

    output_dir.mkdir(parents=True, exist_ok=True)
    rectified_dir = output_dir / 'rectified'
    if save_rectified:
        rectified_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    ])

    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    with torch.no_grad():
        for img_path in image_paths:
            img = io.imread(str(img_path))
            disimgs = transform(img).unsqueeze(0).to(device)

            middle = model_en(disimgs)
            flow_output = model_de(middle)
            clas = model_class(middle)

            predicted_idx = int(torch.argmax(clas, dim=1).item())
            predicted_label = CLASS_NAMES[predicted_idx]

            u = flow_output.detach().cpu().numpy()[0][0]
            v = flow_output.detach().cpu().numpy()[0][1]

            save_mat_path = output_dir / f"{img_path.stem}.mat"
            scio.savemat(
                str(save_mat_path),
                {'u': u, 'v': v, 'class_index': predicted_idx, 'class_name': predicted_label},
            )

            if save_rectified:
                rectified = rectify_image(disimgs, flow_output)
                save_img_path = rectified_dir / f"{img_path.stem}_rectified.png"
                io.imsave(str(save_img_path), rectified)
                print(f"Processed {img_path.name} -> {save_mat_path.name}, {save_img_path.name} | predicted: {predicted_label}")
            else:
                print(f"Processed {img_path.name} -> {save_mat_path.name} | predicted: {predicted_label}")


def parse_args():
    parser = argparse.ArgumentParser(description='Run Guided_Rectification_Network inference on a folder of images.')
    parser.add_argument('--input_dir', type=str, default='/home/ubuntu/Image_Rectification/data-files/input_images')
    parser.add_argument('--output_dir', type=str, default='/home/ubuntu/Image_Rectification/data-files/output-images')
    parser.add_argument('--model_dir', type=str, default='/home/ubuntu/Image_Rectification/pretrained-model')
    parser.add_argument('--save_rectified', action='store_true', help='Save rectified images alongside flow.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_inference(Path(args.input_dir), Path(args.output_dir), Path(args.model_dir), save_rectified=args.save_rectified)

