# fill rate of flat .obj model

import pandas as pd
import numpy as np

filename = "CSV/20_180_depth_metrics.csv"
csv_data = pd.read_csv(filename)
print(csv_data)
rms_plane_fit = csv_data.loc[:, csv_data.columns == 'Plane Fit RMS Error %']
rms_plane_fit = np.array([value[0] for value in rms_plane_fit.values])
rms_plane_fit = rms_plane_fit[np.where(rms_plane_fit < 0.02)[0]]
fill_rate = csv_data.loc[:, csv_data.columns == 'Fill-Rate %']
fill_rate = np.array([value[0] for value in fill_rate.values])
fill_rate = fill_rate[np.where(fill_rate > 50)[0]]
fill_rate_avg = np.average(fill_rate)
avg = np.average(rms_plane_fit)
print(fill_rate)
print(fill_rate_avg)
print(rms_plane_fit)
print(avg)
