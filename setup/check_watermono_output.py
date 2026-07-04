# check_watermono_output.py
import numpy as np

disp = np.load("data/_depth_test/000002_disp.npy")
depth = np.load("data/_depth_test/000002_depth.npy")

print("=== DISP (scaled disparity) ===")
print("shape:", disp.shape, "dtype:", disp.dtype)
print("min/max/mean:", disp.min(), disp.max(), disp.mean())

print("\n=== DEPTH (1/disp) ===")
print("shape:", depth.shape, "dtype:", depth.dtype)
print("min/max/mean:", depth.min(), depth.max(), depth.mean())

# 상관관계 체크: depth와 disp가 정확히 반비례인지
print("\ncorrelation disp vs 1/depth:", np.corrcoef(disp.flatten(), (1/depth).flatten())[0,1])