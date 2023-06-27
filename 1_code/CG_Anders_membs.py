
import numpy as np
import pandas as pd


"""
Helper script. Count the number (percentage) of members for the two files
'table1' and 'members'. The first file contains the data of stars with P>0.5.
The second file contains all the members with P>0.1.
"""

data_P50 = pd.read_csv("../0_data/cantat_gaudin_anders_2020/table1.csv")

N_memb_P50 = data_P50['N'].values
for N in (50, 100, 200):
    msk = N_memb_P50 < N
    print(N, msk.sum() / N_memb_P50.size)

#
# Read file data will all the members
# data_P_all = pd.read_fwf(
#     "../0_data/cantat_gaudin_anders_2020/members.dat",
#     colspecs=[(1,  21), (23,  43), (45,  63), (65,  85), (87, 108), (110, 131),
#     (133, 152), (154, 175), (177, 196), (198, 219), (221, 240), (242, 265),
#     (267, 287), (289, 302), (304, 317), (319, 332), (334, 347), (349, 362),
#     (364, 377), (379, 392), (394, 407), (409, 422), (424, 437), (439, 442),
#     (444, 453), (455, 467), (469, 473), (476, 493)])
data_P_all = pd.read_csv("../0_data/cantat_gaudin_anders_2020/members.csv")

unq_clusts = np.array(list(set(data_P_all['Cluster'])))

N_unq_clusts = []
for cl in unq_clusts:
    msk = data_P_all['Cluster'] == cl
    N_unq_clusts.append(msk.sum())
N_unq_clusts = np.array(N_unq_clusts)

for N in (50, 100, 200):
    msk = N_unq_clusts < N
    print(N, msk.sum() / N_unq_clusts.size)
