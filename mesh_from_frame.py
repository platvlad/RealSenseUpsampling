import numpy as np
import math


def build_mesh(depth_frame, color_frame, fov, output_file):
    """ fov = (horizontal, vertical)
        color_frame = (red_frame, green_frame, blue_frame) """
    hor, vert = fov
    red_frame, green_frame, blue_frame = color_frame
    right = math.tan(math.radians(hor / 2))
    left = -right
    up = math.tan(math.radians(vert / 2))
    vertices = []
    faces = []
    vert_indices = np.zeros(shape=(1080, 1920))
    counter = 1
    height, width = np.shape(depth_frame)
    for y in range(height):
        for x in range(width):
            y_far = up - (y / 1080.0) * 2 * up
            x_far = left + (x / 1920.0) * 2 * right
            z = depth_frame[y][x]
            if z > 0:
                x_real = x_far * z
                y_real = y_far * z
                vertices.append(np.array([x_real, y_real, z, red_frame[y][x] / 256, green_frame[y][x] / 256,
                                          blue_frame[y][x] / 256]))
                vert_indices[y][x] = counter
                counter += 1
                if x and y:
                    if depth_frame[y][x] > 0 and depth_frame[y - 1][x - 1] > 0 and \
                            depth_frame[y - 1][x] > 0:
                        faces.append((vert_indices[y][x], vert_indices[y - 1][x - 1], vert_indices[y - 1][x]))
                    if depth_frame[y][x] > 0 and depth_frame[y][x - 1] > 0 and \
                            depth_frame[y - 1][x - 1] > 0:
                        faces.append((vert_indices[y][x], vert_indices[y][x - 1], vert_indices[y - 1][x - 1]))
    vertices = np.array(vertices)
    with open(output_file, "w") as output:  
        for vertex in vertices:
            output.write(
                "v " + str(vertex[0]) + " " + str(vertex[1]) + " " + str(vertex[2]) + " " + str(vertex[3]) + " " +
                str(vertex[4]) + " " + str(vertex[5]) + "\n")
        for face in faces:
            output.write("f " + str(int(face[0])) + " " + str(int(face[1])) + " " + str(int(face[2])) + "\n")


depth = np.loadtxt('Frames/Cup/depth.txt')
blue_color = np.loadtxt('Frames/Cup/blue color.txt')
green_color = np.loadtxt('Frames/Cup/green color.txt')
red_color = np.loadtxt('Frames/Cup/red color.txt')
joint_bilateral_depth = np.loadtxt('Frames/Cup/joint bilateral.txt')
fov_hor = 69.4
fov_vert = 42.5
fov = (fov_hor, fov_vert)

color_frame = (blue_color, green_color, red_color)
build_mesh(depth, color_frame, fov, output_file='Meshes/Cup/downsample.obj')
build_mesh(joint_bilateral_depth, color_frame, fov, output_file='Meshes/Cup/joint bilateral.obj')
