import numpy as np
import pyrealsense2 as rs
import cv2
import math


def gauss_xy(x, y):
    exp_argument = - ((x * x + y * y) / 2.0)
    return math.exp(exp_argument) / (2 * math.pi)


def gauss_x(x):
    exp_argument = - (x * x / 2.0)
    return math.exp(exp_argument) / math.sqrt(2 * math.pi)


# def jbu():
#     gauss_x = np.vectorize(gauss_x)


frames_to_pass = 2
config = rs.config()
rs.config.enable_device_from_file(config, 'BAG/cup_1_1920.bag')
pipeline = rs.pipeline()
config.enable_all_streams()
# config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
# config.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)
pipeline.start(config)
i = 0
for x in range(frames_to_pass):
    pipeline.wait_for_frames()
frameset = pipeline.wait_for_frames()
align = rs.align(rs.stream.color)
frameset = align.process(frameset)
depth_frame = frameset.get_depth_frame()
color_frame = frameset.get_color_frame()
spatial_filter = rs.spatial_filter()
temporal_filter = rs.temporal_filter()
depth_frame = spatial_filter.process(depth_frame)
depth_frame = temporal_filter.process(depth_frame)
depth = np.asanyarray(depth_frame.get_data())
np.savetxt('Frames/Cup/depth.txt', depth)
depth = depth / 65536.0
color = np.asanyarray(color_frame.get_data())
gray_color = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
gray_color = gray_color / 256.0
gauss_x = np.vectorize(gauss_x)
spatial_gauss = np.array([[gauss_xy(-2, -2), gauss_xy(-2, -1), gauss_xy(-2, 0), gauss_xy(-2, 1), gauss_xy(-2, 2)],
                          [gauss_xy(-1, -2), gauss_xy(-1, -1), gauss_xy(-1, 0), gauss_xy(-1, 1), gauss_xy(-1, 2)],
                          [gauss_xy(0, -2), gauss_xy(0, -1), gauss_xy(0, 0), gauss_xy(0, 1), gauss_xy(0, 2)],
                          [gauss_xy(-1, -2), gauss_xy(-1, -1), gauss_xy(-1, 0), gauss_xy(-1, 1), gauss_xy(-1, 2)],
                          [gauss_xy(-2, -2), gauss_xy(-2, -1), gauss_xy(-2, 0), gauss_xy(-2, 1), gauss_xy(-2, 2)]])

np.savetxt('Frames/Cup/original.txt', depth * 65536)

# try to make color more important
# gray_color = gray_color * 10

new_depth = np.zeros(shape=np.shape(depth))
for i in range(2, len(depth) - 2):
    flag = 0
    for j in range(2, len(depth[i]) - 2):
        color_window = gray_color[i - 2: i + 3, j - 2: j + 3]
        color_window = (gray_color[i][j] - color_window)
        color_window = gauss_x(color_window)
        depth_window = depth[i - 2: i + 3, j - 2: j + 3]
        avg_depth = np.true_divide(depth_window.sum(), (depth_window != 0).sum())
        for k in range(i - 2, i + 3):
            for l in range(j - 2, j + 3):
                depth_window[k - i + 2][l - j + 2] = depth[k][l] if depth[k][l] > 0 else avg_depth
        kernel = color_window * spatial_gauss
        k_p = np.sum(kernel)
        if k_p:
            new_depth[i][j] = np.sum(kernel * depth_window) / k_p
        if not math.isnan(depth[i][j]):
            pass

depth = depth * 65536
new_depth = new_depth * 65536


np.savetxt('Frames/Cup/joint bilateral.txt', new_depth)
blue_color = color[:, :, 0]
green_color = color[:, :, 1]
red_color = color[:, :, 2]
np.savetxt('Frames/Cup/blue color.txt', blue_color)
np.savetxt('Frames/Cup/green color.txt', green_color)
np.savetxt('Frames/Cup/red color.txt', red_color)
