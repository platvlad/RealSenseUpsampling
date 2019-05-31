# mean error of flat .obj momdel

import numpy as np

filename = "PLY/Planes/table_20cm.obj"

flag = True
z_coords = list()
with open(filename) as file:
    for line in file:
        words = line.split()
        if len(words) > 0:
            if words[0] == 'v':
                z = float(words[3])
                z_coords.append(z)

z_coords = np.array(z_coords)
avg = sum(z_coords) / float(len(z_coords))
print("Avg distance =", avg)
diffs = abs(avg - z_coords)
print("Max diff =", max(diffs))
mean_error = sum(diffs) / float(len(diffs))
print("mean error =", mean_error)
no_noise = z_coords[np.where(diffs < 0.04)]
print("Another version:")
avg = sum(no_noise) / float(len(no_noise))
print("Avg distance =", avg)
diffs = abs(avg - no_noise)
print("Max diff =", max(diffs))
mean_error = sum(diffs) / float(len(diffs))
print("mean error =", mean_error)

