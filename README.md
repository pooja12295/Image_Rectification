# Image_Rectification

## Prerequisites
- Linux or Windows
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## For Inference
1) Install dependencies (from project root):
   - `pip install -r requirements.txt`
2) Place your distorted images in `data-files/input_images/` 
3) Ensure pretrained weights are in `pretrained-model/` (`model_en.pkl`, `model_de.pkl`, `model_class.pkl`).
4) Run:
   - `python3 inference.py` (uses default input/output/model paths).
   - Optional: add `--save_rectified` to also save rectified PNGs under `data-files/output-images/rectified/`.
   python3 inference.py --save_rectified
   - Override paths if needed, e.g. `python3 inference.py --input_dir /path/to/images --output_dir /path/to/output --model_dir /path/to/weights`.



### Resampling
Import `resample.resampling.rectification` function to resample the distorted image by the forward flow.

The distorted image should be a Numpy array with the shape of H\*W\*3 for a color image or H\*W for a greyscale image, the forward flow should be an array with the shape of 2\*H\*W.

The function will return the resulting image and a mask to indicate whether each pixel will converge within the maximum iteration.