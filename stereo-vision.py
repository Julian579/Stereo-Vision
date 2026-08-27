import math
import cv2 as cv
import numpy as np

calib_im1 = {
    "fx": 3979.911,
    "baseline": 193.001,   # mm
    "doffs": 124.343,
    "width": 2964,
    "height": 2000,
    "ndisp": 270,
}
calib_im0 = {
        "fx": 6338.47,
        "baseline": 171.548,   # mm
        "doffs": 479.489,
        "width": 2964,
        "height": 2000,
        "ndisp": 390,
    }

def compute_disparity(left_path, right_path, ndisp, block_size=15):
    left = cv.imread(left_path, cv.IMREAD_GRAYSCALE)
    right = cv.imread(right_path, cv.IMREAD_GRAYSCALE)


    num_disparities = math.ceil(ndisp / 16) * 16

    stereo = cv.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
    disparity = stereo.compute(left, right).astype(np.float32) / 16.0  # fixed-point x16
    return disparity, left.shape


def visualize_disparity(disparity, out_path="disparity_vis.png"):
    disp_vis = np.where(disparity > 0, disparity, 0)  # zero out invalid pixels before scaling
    disp_norm = cv.normalize(disp_vis, None, 0, 255, cv.NORM_MINMAX)
    disp_norm = disp_norm.astype(np.uint8)
    heatmap = cv.applyColorMap(disp_norm, cv.COLORMAP_JET)
    cv.imwrite(out_path, heatmap)
    return heatmap


def disparity_to_depth(disparity, pixel, fx, baseline, doffs):
    x, y = pixel
    d = disparity[y, x]
    if d <= 0:
        return None  # invalid: no match, occlusion, or textureless region
    return (fx * baseline) / (d + doffs)


if __name__ == "__main__":
    disp, shape = compute_disparity("img1\im0.png", "img1\im1.png", ndisp=calib_im0["ndisp"], block_size=15)
    visualize_disparity(disp)

    px = (calib_im0["width"] // 2, calib_im0["height"] // 2)
    depth = disparity_to_depth(disp, px, calib_im0["fx"], calib_im0["baseline"], calib_im0["doffs"])
    print(f"For image 0 Depth at {px}: {depth:.2f} mm" if depth else "Invalid disparity at that pixel")



    disp, shape = compute_disparity("im0\im0.png", "im0\im1.png", ndisp=calib_im1["ndisp"], block_size=15)
    visualize_disparity(disp)

    px = (calib_im1["width"] // 2, calib_im1["height"] // 2)
    depth = disparity_to_depth(disp, px, calib_im1["fx"], calib_im1["baseline"], calib_im1["doffs"])
    print(f"For image 1 Depth at {px}: {depth:.2f} mm" if depth else "Invalid disparity at that pixel")