import matplotlib.pyplot as plt
from astropy.io import ascii
import numpy as np
from astropy.coordinates import Angle
from astropy import units as u

data = ascii.read('./hu_et_al/deciphering.dat')

data2 = ascii.read('./hu_et_al/decoding.dat', delimiter=';')
# Match name formatting with 'data'
d2_names = [_.replace('_', '') for _ in list(data2['Cluster'].value)]

theta = []
for i, cl in enumerate(data['Name']):
    if cl in d2_names:
        j = d2_names.index(cl)
        q_delta = abs(data['qall'][i] - data2['qall'][j])
        if q_delta > 5 and q_delta < 85:
            theta.append(data2['qall'][j])
            print("{}: {:.2f}, {:.2f} --> {:.2f}".format(cl, data['qall'][i], data2['qall'][j], data['qall'][i] - data2['qall'][j]))
        else:
            theta.append(data['qall'][i])
    else:
        print("Not in new table:", cl)
        theta.append(data['qall'][i])
theta = np.deg2rad(theta)

lon, lat, a_all = data['lon'], data['lat'], data['aall']
# theta = data['qall']
# # To radians
# theta = np.deg2rad(theta)

plt.subplot(121)
delta_b = a_all * np.tan(theta)
xmin, xmax = lon - a_all, lon + a_all
ymin, ymax = lat - delta_b, lat + delta_b
plt.plot((xmin, xmax), (ymin, ymax), alpha=0.8, zorder=3)
plt.grid(c='lightgray', zorder=0)
plt.ylim([-40, 40])
plt.xlim([0, 360])
plt.xticks(np.arange(0, 390, 30))
plt.xlabel('l')
plt.ylabel('b')

plt.subplot(122)
l_ecc = data['eall'] / data['eall'].max()
r = 30 * l_ecc
x_1 = (np.cos(theta) * r / 2)
y_1 = (np.sin(theta) * r / 2)
plt.plot([lon - x_1, lon + x_1], [lat - y_1, lat + y_1], alpha=0.8, zorder=3)
plt.grid(c='lightgray', zorder=0)
plt.ylim([-40, 40])
plt.xlim([0, 360])
plt.xticks(np.arange(0, 390, 30))
plt.xlabel('l')
plt.ylabel('b')

plt.show()
