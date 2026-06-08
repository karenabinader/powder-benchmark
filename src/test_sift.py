"""
Test SIFT keypoint extraction on a single training image.
This is a sanity check before building the full BOVW pipeline.
Expected output:
  - A reasonable number of keypoints (probably 200-2000)
  - Descriptors shape: (N, 128)
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import cv2
from dataset import PowderDataset

# Load one training image
ds = PowderDataset('data', split='train', transform=None)
img, label = ds[0]

# Convert PIL grayscale to numpy uint8 (what SIFT expects)
img_np = np.array(img)
print(f"Image numpy shape: {img_np.shape}")
print(f"Image dtype: {img_np.dtype}")
print(f"Min pixel: {img_np.min()}, Max pixel: {img_np.max()}")

# Create SIFT detector with default parameters
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(img_np, None)

print(f"\nResults:")
print(f"Number of keypoints detected: {len(keypoints)}")
if descriptors is not None:
    print(f"Descriptors shape: {descriptors.shape}")
    print(f"Descriptor dtype: {descriptors.dtype}")
    print(f"First descriptor (first 10 values): {descriptors[0][:10]}")
else:
    print("WARNING: No descriptors computed!")

# Quick check: are keypoints distributed across the image?
if len(keypoints) > 0:
    xs = [kp.pt[0] for kp in keypoints]
    ys = [kp.pt[1] for kp in keypoints]
    print(f"\nKeypoint x range: {min(xs):.1f} to {max(xs):.1f} (image width = 512)")
    print(f"Keypoint y range: {min(ys):.1f} to {max(ys):.1f} (image height = 512)")