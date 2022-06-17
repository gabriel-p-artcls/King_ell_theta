
# NAME TO BE DETERMINED

This project began on June 2022.



## TODO

1. Process large files 



## Input data

### Cantat-Gaudin & Anders (2020)

1481 clusters (435833 stars with P>0.01); G_max=18

Average number of members

For members identified as those with P>0.5 (table1.csv):
 N<50 : 26.5%
 N<100: 57.1%
 N<200: 80.9%

For all the stars selected as members P>0.01 (members.csv):
 N<50 :  7.7% (115)
 N<100: 25.7% (380)
 N<200: 53.1% (787)
 N<300: 69.6% (1031)

Mean   = 294 stars per cluster (435833/1481)
Median = 184 stars per cluster
Max    = 3646 stars per cluster (NGC_7789)
Min    = 14 stars per cluster (DBSB_21)


### Cantat-Gaudin et al. (2020)

2017 clusters (234129 stars with P>0.7); parameters for 1867.
Mean = 125 stars per cluster (234129/1867)

* r>30'  : 138 clusters
* d<1kpc : 245 clusters

* Sol processes 1634 clusters with r<30' & d>1kpc
* I process 383 clusters: 138 with r>30' and 245 with d<1kpc

1. What is the G limit used?
  The hard limit is 18, but there are a 15 stars up to ~19.6. These belong to
  Hyades/Melotte 25 (9) and Melotte 111 (6)
2. How were very large clusters processed?
  They were not processed with UPMASK. This is explained in Sect 2.1.


### Synthetic data

Tested 4 sets of synthetic King profiles to test SVD vs ASteCA with the following parameters:

 0. outl_perc=(10, 25); r_max_outl=1.5
 1. outl_perc=(5, 20) ; r_max_outl=1.
 2. outl_perc=(5, 10) ; r_max_outl=1.
 3. outl_perc=5       ; r_max_outl=1.

For all the sets these parameters are fixed: N_memb=200; ell_min=0.2, ell_max=0.8; CI=0.5.

Results of this analysis: not even in the most favorable run (3) is the SVD
method able to match the performance of ASteCA for either of the fitted
parameters.