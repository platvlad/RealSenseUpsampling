import numpy as np
import math


depth = np.loadtxt('Normals/Cup/depth.txt')
blue_color = np.loadtxt('Normals/Cup/blue_color.txt')
green_color = np.loadtxt('Normals/Cup/green_color.txt')
red_color = np.loadtxt('Normals/Cup/red_color.txt')
normals0 = np.loadtxt('Normals/Cup/new normals0.txt')
normals1 = np.loadtxt('Normals/Cup/new normals1.txt')
normals2 = np.loadtxt('Normals/Cup/new normals2.txt')


def get_depth_by_direction_vector(points, new_points, y, x, direction_vector, right=True):
    if right:
        prev_point = points[y][x - 1]
        prev_new_point = new_points[y][x - 1]
    else:
        prev_point = points[y - 1][x]
        prev_new_point = new_points[y - 1][x]
    a = prev_new_point
    c = direction_vector
    a_len = np.linalg.norm(prev_new_point)
    c_len = np.linalg.norm(direction_vector)
    dot_product_ac = np.dot(-a, c)
    cos_beta = dot_product_ac / (a_len * c_len)
    dot_product_ab = np.dot(prev_point, points[y][x])
    cos_gamma = dot_product_ab / (np.linalg.norm(points[y][x]) * np.linalg.norm(prev_point))
    beta = math.acos(cos_beta)
    gamma = math.acos(cos_gamma)
    alpha = math.pi - beta - gamma
    depth_yx = a_len * math.sin(beta) / math.sin(alpha)
    return depth_yx


def build_mesh(depth_frame, fov, output_file):
    """ fov = (horizontal, vertical)
        color_frame = (red_frame, green_frame, blue_frame) """
    hor, vert = fov
    left = math.tan(math.radians(hor / 2))
    right = -left
    up = math.tan(math.radians(vert / 2))
    vertices = []
    faces = []
    vert_indices = np.zeros(shape=(1080, 1920))
    height, width = np.shape(depth_frame)
    points = np.zeros(shape=(height, width, 3))
    new_points = np.zeros(shape=(height, width, 3))
    down_vectors = np.zeros(shape=(height, width, 3))
    right_vectors = np.zeros(shape=(height, width, 3))
    for y in range(height):
        for x in range(width):
            y_far = up - (y / 1080.0) * 2 * up
            x_far = left + (x / 1920.0) * 2 * right
            z = depth_frame[y][x]
            if z > 0:
                x_real = x_far * z
                y_real = y_far * z
                points[y][x] = [x_real, y_real, z]

    for y in range(height):
        for x in range(width):
            if y == 0:
                if x == 0:
                    new_points[y][x] = points[y][x]
                    if normals0[y][x] or normals1[y][x] or normals2[y][x]:
                        right_length = np.linalg.norm(points[y][x + 1] - points[y][x])
                        normal_vector = np.array([normals0[y][x], normals1[y][x], normals2[y][x]])
                        down_normal_vector = np.array([normals0[y + 1][x], normals1[y + 1][x], normals2[y + 1][x]])
                        right_unnormalized = np.cross(normal_vector, down_normal_vector)
                        if right_unnormalized[0] == 0 and right_unnormalized[1] == 0 and right_unnormalized[2] == 0:
                            right_unnormalized = np.cross(normal_vector, points[y + 1][x] - points[y][x])
                        if right_unnormalized[0] > 0:
                            right_unnormalized = -right_unnormalized
                        right_unnormalized_length = np.linalg.norm(right_unnormalized)
                        right_vectors[y][x] = right_unnormalized * (right_length / right_unnormalized_length)
                        down_length = np.linalg.norm(points[y + 1][x] - points[y][x])
                        right_normal_vector = np.array([normals0[y][x + 1], normals1[y][x + 1], normals2[y][x + 1]])
                        down_unnormalized = np.cross(right_normal_vector, normal_vector)
                        if down_unnormalized[0] == 0 and down_unnormalized[1] == 0 and down_unnormalized[2] == 0:
                            down_unnormalized = np.cross(right_vectors[y][x], normal_vector)
                        if down_unnormalized[1] > 0:
                            down_unnormalized = -down_unnormalized
                        down_unnormalized_length = np.linalg.norm(down_unnormalized)
                        down_vectors[y][x] = down_unnormalized * (down_length / down_unnormalized_length)
                else:
                    if right_vectors[y][x - 1][0] or right_vectors[y][x - 1][1] or right_vectors[y][x - 1][2]:
                        new_points[y][x] = new_points[y][x - 1] + right_vectors[y][x - 1]
                    else:
                        new_points[y][x] = points[y][x]
                    if normals0[y][x] or normals1[y][x] or normals2[y][x]:
                        right_length = np.linalg.norm(points[y][x] - points[y][x - 1])
                        normal_vector = np.array([normals0[y][x], normals1[y][x], normals2[y][x]])
                        down_normal_vector = np.array([normals0[y + 1][x], normals1[y + 1][x], normals2[y + 1][x]])
                        right_unnormalized = np.cross(normal_vector, down_normal_vector)
                        if right_unnormalized[0] == 0 and right_unnormalized[1] == 0 and right_unnormalized[2] == 0:
                            right_unnormalized = np.cross(normal_vector, points[y + 1][x] - points[y][x])
                        if right_unnormalized[0] > 0:
                            right_unnormalized = -right_unnormalized
                        right_unnormalized_length = np.linalg.norm(right_unnormalized)
                        right_vectors[y][x] = right_unnormalized * (right_length / right_unnormalized_length)
                        down_length = np.linalg.norm(points[y + 1][x] - points[y][x])
                        left_normal_vector = np.array([normals0[y][x - 1], normals1[y][x - 1], normals2[y][x - 1]])
                        down_unnormalized = np.cross(left_normal_vector, normal_vector)
                        if down_unnormalized[0] == 0 and down_unnormalized[1] == 0 and down_unnormalized[2] == 0:
                            if x < width - 1:
                                right_normal_vector = \
                                    np.array([normals0[y][x + 1], normals1[y][x + 1], normals2[y][x + 1]])
                                down_unnormalized = np.cross(normal_vector, right_normal_vector)
                            if down_unnormalized[0] == 0 and down_unnormalized[1] == 0 and down_unnormalized[2] == 0:
                                down_unnormalized = np.cross(right_vectors[y][x], normal_vector)
                        if down_unnormalized[1] > 0:
                            down_unnormalized = -down_unnormalized
                        down_unnormalized_length = np.linalg.norm(down_unnormalized)
                        down_vectors[y][x] = down_unnormalized * (down_length / down_unnormalized_length)
            else:
                if x == 0:
                    if down_vectors[y - 1][x][0] or down_vectors[y - 1][x][1] or down_vectors[y - 1][x][2]:
                        new_points[y][x] = new_points[y - 1][x] + down_vectors[y - 1][x]
                    else:
                        new_points[y][x] = points[y][x]
                    if normals0[y][x] or normals1[y][x] or normals2[y][x]:
                        right_length = np.linalg.norm(points[y][x + 1] - points[y][x])
                        normal_vector = np.array([normals0[y][x], normals1[y][x], normals2[y][x]])
                        up_normal_vector = np.array([normals0[y - 1][x], normals1[y - 1][x], normals2[y - 1][x]])
                        right_unnormalized = np.cross(up_normal_vector, normal_vector)
                        if right_unnormalized[0] == 0 and right_unnormalized[1] == 0 and right_unnormalized[2] == 0:
                            if y < height - 1:
                                down_normal_vector = np.array(
                                    [normals0[y + 1][x], normals1[y + 1][x], normals2[y + 1][x]])
                                right_unnormalized = np.cross(normal_vector, down_normal_vector)
                            if right_unnormalized[0] == 0 and right_unnormalized[1] == 0 and right_unnormalized[
                                2] == 0:
                                right_unnormalized = np.cross(normal_vector, points[y][x] - points[y - 1][x])
                        if right_unnormalized[0] > 0:
                            right_unnormalized = -right_unnormalized
                        right_unnormalized_length = np.linalg.norm(right_unnormalized)
                        right_vectors[y][x] = right_unnormalized * (right_length / right_unnormalized_length)
                        down_length = np.linalg.norm(points[y][x] - points[y - 1][x])
                        right_normal_vector = np.array([normals0[y][x + 1], normals1[y][x + 1], normals2[y][x + 1]])
                        down_unnormalized = np.cross(right_normal_vector, normal_vector)
                        if down_unnormalized[0] == 0 and down_unnormalized[1] == 0 and down_unnormalized[2] == 0:
                            down_unnormalized = np.cross(right_vectors[y][x], normal_vector)
                        down_unnormalized_length = np.linalg.norm(down_unnormalized)
                        down_vectors[y][x] = down_unnormalized * (down_length / down_unnormalized_length)
                else:
                    if right_vectors[y][x - 1][0] or right_vectors[y][x - 1][1] or right_vectors[y][x - 1][2]:
                        new_points[y][x] = new_points[y][x - 1] + right_vectors[y][x - 1]
                        if down_vectors[y - 1][x][0] or down_vectors[y - 1][x][1] or down_vectors[y - 1][x][2]:
                            new_points[y][x] += new_points[y - 1][x] + down_vectors[y - 1][x]
                            new_points[y][x] /= 2
                    else:
                        if down_vectors[y - 1][x][0] or down_vectors[y - 1][x][1] or down_vectors[y - 1][x][2]:
                            new_points[y][x] = new_points[y - 1][x] + down_vectors[y - 1][x]
                        else:
                            new_points[y][x] = points[y][x]
                    if normals0[y][x] or normals1[y][x] or normals2[y][x]:
                        right_length = np.linalg.norm(points[y][x] - points[y][x - 1])
                        normal_vector = np.array([normals0[y][x], normals1[y][x], normals2[y][x]])
                        up_normal_vector = np.array([normals0[y - 1][x], normals1[y - 1][x], normals2[y - 1][x]])
                        right_unnormalized = np.cross(up_normal_vector, normal_vector)
                        if right_unnormalized[0] == 0 and right_unnormalized[1] == 0 and right_unnormalized[2] == 0:
                            if y < height - 1:
                                down_normal_vector = np.array(
                                    [normals0[y + 1][x], normals1[y + 1][x], normals2[y + 1][x]])
                                right_unnormalized = np.cross(normal_vector, down_normal_vector)
                            if right_unnormalized[0] == 0 and right_unnormalized[1] == 0 and right_unnormalized[
                                2] == 0:
                                right_unnormalized = np.cross(normal_vector, points[y][x] - points[y - 1][x])
                        if right_unnormalized[0] > 0:
                            right_unnormalized = -right_unnormalized
                        right_unnormalized_length = np.linalg.norm(right_unnormalized)
                        right_vectors[y][x] = right_unnormalized * (right_length / right_unnormalized_length)
                        down_length = np.linalg.norm(points[y][x] - points[y - 1][x])
                        left_normal_vector = np.array([normals0[y][x - 1], normals1[y][x - 1], normals2[y][x - 1]])
                        down_unnormalized = np.cross(left_normal_vector, normal_vector)
                        if down_unnormalized[0] == 0 and down_unnormalized[1] == 0 and down_unnormalized[2] == 0:
                            if x < width - 1:
                                right_normal_vector = \
                                    np.array([normals0[y][x + 1], normals1[y][x + 1], normals2[y][x + 1]])
                                down_unnormalized = np.cross(normal_vector, right_normal_vector)
                            if down_unnormalized[0] == 0 and down_unnormalized[1] == 0 and down_unnormalized[2] == 0:
                                down_unnormalized = np.cross(right_vectors[y][x], normal_vector)
                        if down_unnormalized[1] > 0:
                            down_unnormalized = -down_unnormalized
                        down_unnormalized_length = np.linalg.norm(down_unnormalized)
                        down_vectors[y][x] = down_unnormalized * (down_length / down_unnormalized_length)
    counter = 1
    for y in range(height):
        for x in range(width):
            if new_points[y][x][2]:
                vertices.append(np.array([new_points[y][x][0], new_points[y][x][1], new_points[y][x][2],
                                        red_color[y][x] / 256, green_color[y][x] / 256, blue_color[y][x] / 256]))
                vert_indices[y][x] = counter
                counter += 1
                if x and y:
                    if new_points[y][x][2] > 0 and new_points[y - 1][x - 1][2] > 0 and \
                            new_points[y - 1][x][2] > 0:
                        faces.append((vert_indices[y][x], vert_indices[y - 1][x - 1], vert_indices[y - 1][x]))
                    if new_points[y][x][2] > 0 and new_points[y][x - 1][2] > 0 and \
                            new_points[y - 1][x - 1][2] > 0:
                        faces.append((vert_indices[y][x], vert_indices[y][x - 1], vert_indices[y - 1][x - 1]))
    difference = points - new_points
    vertices = np.array(vertices)
    with open(output_file, "w") as output: 
        for vertex in vertices:
            output.write(
                "v " + str(vertex[0]) + " " + str(vertex[1]) + " " + str(vertex[2]) + " " + str(vertex[3]) + " " +
                str(vertex[4]) + " " + str(vertex[5]) + "\n")
        for face in faces:
            output.write("f " + str(int(face[0])) + " " + str(int(face[1])) + " " + str(int(face[2])) + "\n")


def build_old_mesh(depth_frame, color_frame, fov, output_file):
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


fov_hor = 69.4
fov_vert = 42.5
fov = (fov_hor, fov_vert)

color_frame = (red_color, green_color, blue_color)
build_old_mesh(depth, color_frame,fov, output_file='Meshes/Cup/original.obj')

build_mesh(depth, fov, output_file='Meshes/Cup/with_normals.obj')
