import pyrealsense2 as rs
import numpy as np
import math
import cv2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.optimize import lsq_linear


class Map:
    def __init__(self, depth_map, color_map):
        self.depth_map = depth_map
        self.color_map = color_map
        self.height, self.width = np.shape(depth_map)
        self.fov_hor = 69.4
        self.fov_vert = 42.5
        self.points = self.points_from_depth()
        self.depth_normals = np.zeros(shape=np.shape(color_map))
        self.rgb_clusters = dict()
        self.canonical_normals = Map.polyhedral_normals()
        self.B_structure = None
        self.adjacency_matrix = None
        self.m_matrix = None

    # x from right to left
    # y from down to up
    # z from backward to forward
    def points_from_depth(self):
        left = math.tan(math.radians(self.fov_hor / 2))
        right = -left
        up = math.tan(math.radians(self.fov_vert / 2))
        down = -up
        self.points = np.array([[[(left + (j / self.width) * 2 * right) * self.depth_map[i][j],
                                  (up + (i / self.height) * 2 * down) * self.depth_map[i][j],
                                  self.depth_map[i][j]] for j in range(self.width)] for i in range(self.height)])
        return self.points

    def normals_from_depth(self):
        # self.points_from_depth()
        right_points = self.points[:, 2:]
        left_points = self.points[:, :-2]
        up_points = self.points[:-2, :]
        down_points = self.points[2:, :]
        left_to_right = right_points - left_points
        down_to_up = up_points - down_points
        normals = np.zeros(shape=self.color_map.shape)
        normals[1:-1, 1:-1] = np.cross(left_to_right[1:-1], down_to_up[:, 1:-1])
        non_zero_depth = np.zeros(shape=self.depth_map.shape)
        non_zero_depth[np.where(self.depth_map > 0)] = 1
        norms_of_normals = np.linalg.norm(normals, axis=2)
        # exclude normals on zero pixels and zero pixels neighbours normals
        norms_of_normals = norms_of_normals * non_zero_depth
        norms_of_normals[:, :-1] = norms_of_normals[:, :-1] * non_zero_depth[:, 1:]
        norms_of_normals[:, 1:] = norms_of_normals[:, 1:] * non_zero_depth[:, :-1]
        norms_of_normals[:-1] = norms_of_normals[:-1] * non_zero_depth[1:]
        norms_of_normals[1:] = norms_of_normals[1:] * non_zero_depth[:-1]
        self.depth_normals = np.array([[normals[i][j] / norms_of_normals[i][j]
                                        if norms_of_normals[i][j] > 0 else np.zeros(shape=(3,))
                                        for j in range(self.width)] for i in range(self.height)])
        return self.depth_normals

    def clusterize_rgb(self):
        spatial_radius = 35
        color_radius = 60
        mean_shift_filtered = cv2.pyrMeanShiftFiltering(self.color_map, spatial_radius, color_radius)
        for i in range(len(mean_shift_filtered)):
            for j in range(len(mean_shift_filtered[i])):
                pixel = mean_shift_filtered[i][j]
                if (pixel[0], pixel[1], pixel[2]) not in self.rgb_clusters:
                    self.rgb_clusters[(pixel[0], pixel[1], pixel[2])] = list()
                self.rgb_clusters[(pixel[0], pixel[1], pixel[2])].append([i, j])
        for color_value, pixels in list(self.rgb_clusters.items()):
            if len(pixels) < 500:
                del self.rgb_clusters[color_value]

    def sfs_error(self, cluster_number, i, j, normal):
        nit = np.array([[normal[0], normal[1], normal[2], 1]])
        ni = np.zeros(shape=(4, 1))
        ni[0][0] = normal[0]
        ni[1][0] = normal[1]
        ni[2][0] = normal[2]
        ni[3][0] = 1
        m_red = self.m_matrix[cluster_number][0]
        m_green = self.m_matrix[cluster_number][1]
        m_blue = self.m_matrix[cluster_number][2]
        m_matrix_red = np.array([[m_red[0], m_red[1], m_red[2], m_red[3]],
                                 [m_red[1], m_red[4], m_red[5], m_red[6]],
                                 [m_red[2], m_red[5], m_red[7], m_red[8]],
                                 [m_red[3], m_red[6], m_red[8], m_red[9]]])

        m_matrix_green = np.array([[m_green[0], m_green[1], m_green[2], m_green[3]],
                                   [m_green[1], m_green[4], m_green[5], m_green[6]],
                                   [m_green[2], m_green[5], m_green[7], m_green[8]],
                                   [m_green[3], m_green[6], m_green[8], m_green[9]]])

        m_matrix_blue = np.array([[m_blue[0], m_blue[1], m_blue[2], m_blue[3]],
                                  [m_blue[1], m_blue[4], m_blue[5], m_blue[6]],
                                  [m_blue[2], m_blue[5], m_blue[7], m_blue[8]],
                                  [m_blue[3], m_blue[6], m_blue[8], m_blue[9]]])
        nit_m_red = np.dot(nit, m_matrix_red)
        nit_m_green = np.dot(nit, m_matrix_green)
        nit_m_blue = np.dot(nit, m_matrix_blue)
        nit_m_ni_red = np.dot(nit_m_red, ni)[0][0]
        nit_m_ni_green = np.dot(nit_m_green, ni)[0][0]
        nit_m_ni_blue = np.dot(nit_m_blue, ni)[0][0]
        e_sfs = 0
        e_sfs += (self.color_map[i][j][0] - self.albedos[cluster_number][0] * nit_m_ni_red) * (
                    self.color_map[i][j][0] - self.albedos[cluster_number][0] * nit_m_ni_red)
        e_sfs += (self.color_map[i][j][1] - self.albedos[cluster_number][1] * nit_m_ni_green) * (
                    self.color_map[i][j][1] - self.albedos[cluster_number][1] * nit_m_ni_green)
        e_sfs += (self.color_map[i][j][2] - self.albedos[cluster_number][2] * nit_m_ni_blue) * (
                self.color_map[i][j][2] - self.albedos[cluster_number][2] * nit_m_ni_blue)
        return e_sfs / 3

    def prior_error(self, i, j, normal):
        diff = np.linalg.norm(self.depth_normals[i][j] - normal) ** 2
        return diff

    def norm_error(self, normal):
        return (np.sum(normal * normal) - 1) ** 2

    def enhance_normals(self):
        self.new_normals = np.copy(self.depth_normals)
        cluster_number = 0
        for k, pixel_list in self.rgb_clusters.items():
            for pixel in pixel_list:
                i, j = pixel
                if self.depth_normals[i][j][0] != 0 or self.depth_normals[i][j][1] != 0 or \
                        self.depth_normals[i][j][2] != 0:
                    step = 0.01
                    num_of_iterations = 0
                    to_continue = True
                    while to_continue:
                        up_x = [step, 0, 0]
                        up_y = [0, step, 0]
                        up_z = [0, 0, step]
                        down_x = [-step, 0, 0]
                        down_y = [0, -step, 0]
                        down_z = [0, 0, -step]
                        variants = [self.new_normals[i][j] + up_x, self.new_normals[i][j] + up_y,
                                    self.new_normals[i][j] + up_z,
                                    self.new_normals[i][j] + down_x, self.new_normals[i][j] + down_y,
                                    self.new_normals[i][j] + down_z]
                        variants_sfs_errors = [self.sfs_error(cluster_number, i, j, variant) +
                                           0.1 * self.prior_error(i, j, variant) +
                                            0.05 * self.norm_error(variant) for variant in variants]
                        base_variant_error = self.sfs_error(cluster_number, i, j, self.new_normals[i][j]) + \
                                         0.1 * self.prior_error(i, j, self.new_normals[i][j]) + \
                                         0.05 * self.norm_error(self.new_normals[i][j])
                        if np.min(variants_sfs_errors) < base_variant_error:
                            min_argument = np.argmin(variants_sfs_errors)
                            self.new_normals[i][j] = variants[min_argument]
                        else:
                            to_continue = False
                        num_of_iterations += 1
                        if num_of_iterations % 10 == 0:
                            step *= 2
            print("cluster counter =", cluster_number)
            cluster_number += 1
        return self.new_normals


    def recompute_depth(self):
        self.normals_from_depth()
        print("estimated normals")
        self.clusterize_rgb()
        print("clusterized rgb")
        self.fill_B_structure()
        print("filled B structure")
        self.build_graph()
        print("built graph")
        self.build_tree()
        print("built tree")
        self.compute_albedo()
        print("computed albedo")
        self.estimate_m_matrix()
        print("estimated m_matrix")
        for _ in range(3):
            self.albedo_by_m_matrix()
            self.estimate_m_matrix()
        self.albedo_by_m_matrix()
        return self.enhance_normals()

    def point_cloud_by_normals(self):
        self.new_point_cloud = np.zeros(shape=np.shape(self.points))
        for k, pixel_list in self.rgb_clusters.items():
            for pixel in pixel_list:
                i, j = pixel
                if np.linalg.norm(self.depth_normals[i][j]):
                    self.new_point_cloud[i][j] = self.points[i][j] + self.new_depth_normals[i][j]

    def save_obj(self):
        pass

    @staticmethod
    def polyhedral_normals():
        phi_angles = np.linspace(-math.pi / 2, math.pi / 2, 25)
        psi_angles = np.linspace(-math.pi / 2, math.pi / 2, 25)
        canonical_normals = np.zeros(shape=(25, 25, 3))
        for i in range(len(phi_angles)):
            phi = phi_angles[i]
            for j in range(len(psi_angles)):
                psi = psi_angles[j]
                canonical_normals[i][j] = [math.sin(phi) * math.cos(psi), math.sin(phi) * math.sin(psi), math.cos(phi)]
        return canonical_normals

    def fill_B_structure(self):
        self.B_structure = np.zeros(shape=(len(self.rgb_clusters), int(self.canonical_normals.size / 3), 4))
        B_dict = dict()
        counter = 0
        for pixel_value, pixel_list in list(self.rgb_clusters.items()):
            non_zero_normal_in_cluster = False
            for pixel in pixel_list:
                pixel_normal = self.depth_normals[pixel[0], pixel[1]]
                if np.linalg.norm(pixel_normal) > 0:
                    non_zero_normal_in_cluster = True
                    pixel_normals = np.array([pixel_normal] * len(self.canonical_normals[0]))
                    cross_products_columns = np.cross(self.canonical_normals[0], pixel_normals)
                    cross_product_norms_columns = np.linalg.norm(cross_products_columns, axis=1)
                    best_canonical_normal_column = np.argmin(cross_product_norms_columns)
                    cross_products_rows = \
                        np.cross(self.canonical_normals[:, best_canonical_normal_column], pixel_normals)
                    cross_product_norms_rows = np.linalg.norm(cross_products_rows, axis=1)
                    best_canonical_normal_row = np.argmin(cross_product_norms_rows)
                    best_canonical_normal = 25 * best_canonical_normal_row + best_canonical_normal_column
                    if (counter, best_canonical_normal) not in B_dict:
                        B_dict[(counter, best_canonical_normal)] = []
                    B_dict[(counter, best_canonical_normal)].append(self.color_map[pixel[0], pixel[1]])
            if non_zero_normal_in_cluster:
                counter += 1
        for cell, pixel_list in B_dict.items():
            cluster_num, canonical_normal = cell
            pixel_array = np.array(pixel_list)
            median_pixel_value = np.median(pixel_array, axis=0)
            self.B_structure[cluster_num][canonical_normal][0] = 1
            self.B_structure[cluster_num][canonical_normal][1:] = median_pixel_value

    def build_graph(self):
        self.adjacency_matrix = np.zeros(shape=(len(self.B_structure), len(self.B_structure)))
        for i in range(len(self.adjacency_matrix[0])):
            for j in range(i + 1, len(self.adjacency_matrix[0])):
                first_vertex_normals = self.B_structure[i, :, 0]
                second_vertex_normals = self.B_structure[j, :, 0]
                common_vertex_normals = first_vertex_normals * second_vertex_normals
                num_of_common_normals = len(np.where(common_vertex_normals > 0)[0])
                if num_of_common_normals > 20:
                    self.adjacency_matrix[i][j] = num_of_common_normals

    def build_tree(self):
        self.revert_weights()
        csr_adjacency_matrix = csr_matrix(self.adjacency_matrix)
        self.tree = minimum_spanning_tree(csr_adjacency_matrix)
        self.tree = self.tree.toarray().astype(int)
        maxs = np.amax(self.tree, axis=1)
        maxs2 = np.amax(self.tree, axis=0)
        max_plus = maxs + maxs2

    def revert_weights(self):
        max_weight = np.max(self.adjacency_matrix)
        adjacency_matrix_flag = [[1 if self.adjacency_matrix[i][j] > 0 else 0
                                  for j in range(len(self.adjacency_matrix[i]))]
                                 for i in range(len(self.adjacency_matrix))]
        self.adjacency_matrix = (max_weight + 100 - self.adjacency_matrix) * adjacency_matrix_flag

    def visit_components(self, vertex_num):
        self.visited[vertex_num] = True
        counter = 1
        for j in range(len(self.visited)):
            if (self.tree[vertex_num][j] or self.tree[j][vertex_num]) and not self.visited[j]:
                counter += self.visit_components(j)
        return counter

    def find_max_component(self):
        self.visited = [False] * len(self.tree[0])
        component_sizes = np.ones(shape=(len(self.visited)))
        for i in range(len(self.tree[0])):
            if not self.visited[i]:
                self.visited[i] = True
                for j in range(len(self.visited)):
                    if (self.tree[i][j] or self.tree[j][i]) and not self.visited[j]:
                        component_sizes[i] += self.visit_components(j)
        return np.argmax(component_sizes)

    def visit_albedo(self, i):
        self.visited[i] = True
        for j in range(len(self.visited)):
            if not self.visited[j] and (self.tree[i][j] or self.tree[j][i]):
                self.estimate_albedos_ratio(i, j)
                self.visit_albedo(j)

    def estimate_albedos_ratio(self, i, j):
        i_vertex_normals = self.B_structure[i, :, 0]
        j_vertex_normals = self.B_structure[j, :, 0]
        common_normals = i_vertex_normals * j_vertex_normals
        non_zero_normals = np.where(common_normals > 0)
        i_vertex_non_zero = self.B_structure[i, non_zero_normals, 1:]
        j_vertex_non_zero = self.B_structure[j, non_zero_normals, 1:]
        intensity_ratio = j_vertex_non_zero / i_vertex_non_zero
        ratios = np.median(intensity_ratio, axis=1)
        self.albedos[j] = self.albedos[i] * ratios

    def compute_albedo(self):
        max_component_start = self.find_max_component()
        self.albedos = np.zeros(shape=(len(self.B_structure), 3))
        self.albedos[max_component_start] = np.array([0.5, 0.5, 0.5])
        self.visited = [False] * len(self.visited)
        self.visited[max_component_start] = True
        for i in range(len(self.visited)):
            if not self.visited[i] and (self.tree[max_component_start][i] or self.tree[i][max_component_start]):
                self.estimate_albedos_ratio(max_component_start, i)
                self.visit_albedo(i)

    def estimate_m_matrix(self):
        if self.m_matrix is None:
            self.m_matrix = np.zeros(shape=(len(self.albedos), 3, 10))
        counter = 0
        for k, pixel_list in self.rgb_clusters.items():
            if self.albedos[counter][0] > 0 and self.albedos[counter][1] > 0 and self.albedos[counter][2] > 0:
                A = []
                bs = []
                for pixel in pixel_list:
                    i, j = pixel
                    n1, n2, n3 = self.depth_normals[i][j]
                    if n1 != 0 or n2 != 0 or n3 != 0:
                        bs.append(self.color_map[i][j] / self.albedos[counter])
                        A.append([n1 * n1, 2 * n1 * n2, 2 * n1 * n3, n1,
                                               n2 * n2, 2 * n2 * n3, n2,
                                                            n3 * n3, n3,
                                                                      1])
                if not bs:
                    counter += 1
                    continue
                A = np.array(A)
                bs = np.array(bs)
                red_lighting = lsq_linear(A, bs[:, 0])
                self.m_matrix[counter][0] = red_lighting.x
                green_lighting = lsq_linear(A, bs[:, 1])
                self.m_matrix[counter][1] = green_lighting.x
                blue_lighting = lsq_linear(A, bs[:, 2])
                self.m_matrix[counter][2] = blue_lighting.x
            counter += 1

    def albedo_by_m_matrix(self):
        counter = 0
        for k, pixel_list in self.rgb_clusters.items():
            red_albedo_candidats = list()
            green_albedo_candidats = list()
            blue_albedo_candidats = list()
            for pixel in pixel_list:
                i, j = pixel
                if np.linalg.norm(self.depth_normals[i][j]):
                    nit = np.array([[self.depth_normals[i][j][0], self.depth_normals[i][j][1], self.depth_normals[i][j][2], 1]])
                    ni = np.zeros(shape=(4, 1))
                    ni[0][0] = self.depth_normals[i][j][0]
                    ni[1][0] = self.depth_normals[i][j][1]
                    ni[2][0] = self.depth_normals[i][j][2]
                    ni[3][0] = 1
                    # ni = np.array([[self.depth_normals[i][j][0]], [self.depth_normals[i][j][1]], [self.depth_normals[i][j][2], 1]])
                    m_red = self.m_matrix[counter][0]
                    m_green = self.m_matrix[counter][1]
                    m_blue = self.m_matrix[counter][2]
                    m_matrix_red = np.array([[m_red[0], m_red[1], m_red[2], m_red[3]],
                                             [m_red[1], m_red[4], m_red[5], m_red[6]],
                                             [m_red[2], m_red[5], m_red[7], m_red[8]],
                                             [m_red[3], m_red[6], m_red[8], m_red[9]]])

                    m_matrix_green = np.array([[m_green[0], m_green[1], m_green[2], m_green[3]],
                                             [m_green[1], m_green[4], m_green[5], m_green[6]],
                                             [m_green[2], m_green[5], m_green[7], m_green[8]],
                                             [m_green[3], m_green[6], m_green[8], m_green[9]]])

                    m_matrix_blue = np.array([[m_blue[0], m_blue[1], m_blue[2], m_blue[3]],
                                             [m_blue[1], m_blue[4], m_blue[5], m_blue[6]],
                                             [m_blue[2], m_blue[5], m_blue[7], m_blue[8]],
                                             [m_blue[3], m_blue[6], m_blue[8], m_blue[9]]])
                    nit_m_red = np.dot(nit, m_matrix_red)
                    nit_m_green = np.dot(nit, m_matrix_green)
                    nit_m_blue = np.dot(nit, m_matrix_blue)
                    nit_m_ni_red = np.dot(nit_m_red, ni)[0][0]
                    nit_m_ni_green = np.dot(nit_m_green, ni)[0][0]
                    nit_m_ni_blue = np.dot(nit_m_blue, ni)[0][0]
                    if nit_m_ni_red > 0 and nit_m_ni_green > 0 and nit_m_ni_blue > 0:
                        red_albedo_candidats.append(self.color_map[i][j][0] / nit_m_ni_red)
                        green_albedo_candidats.append(self.color_map[i][j][1] / nit_m_ni_green)
                        blue_albedo_candidats.append(self.color_map[i][j][2] / nit_m_ni_blue)
            if red_albedo_candidats:
                red_albedo = np.median(red_albedo_candidats)
                green_albedo = np.median(green_albedo_candidats)
                blue_albedo = np.median(blue_albedo_candidats)
                self.albedos[counter] = [red_albedo, green_albedo, blue_albedo]
            counter += 1


frames_to_pass = 2
config = rs.config()
rs.config.enable_device_from_file(config,
                                  'cup_1_1920.bag')
pipeline = rs.pipeline()
config.enable_all_streams()
pipeline.start(config)
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
color = np.asanyarray(color_frame.get_data())

map = Map(depth, color)

new_depth_normals = map.recompute_depth()
np.savetxt('Normals/cup/new normals0.txt', new_depth_normals[:, :, 0])
np.savetxt('Normals/cup/new normals1.txt', new_depth_normals[:, :, 1])
np.savetxt('Normals/cup/new normals2.txt', new_depth_normals[:, :, 2])
old_points = map.points
new_points = np.zeros(shape=np.shape(old_points))
for k, pixel_list in map.rgb_clusters.items():
    for pixel in pixel_list:
        i, j = pixel
        if np.linalg.norm(map.depth_normals[i][j]):
            new_points[i][j] = old_points[i][j] + new_depth_normals[i][j]
with open('Normals/cup/normals_added.obj', "w") as output:  # a
    for row in new_points:
        for vertex in row:
            output.write(
                "v " + str(vertex[0]) + " " + str(vertex[1]) + " " + str(vertex[2]) + "\n")
np.savetxt('Normals/cup/old_depth.txt', depth)
np.savetxt('Normals/cup/red_color.txt', color[:, :, 0])
np.savetxt('Normals/cup/green_color.txt', color[:, :, 1])
np.savetxt('Normals/cup/blue_color.txt', color[:, :, 2])
