
# NAME TO BE DETERMINED

This project began on June 2022.



## TODO

1. Process large files 


## Cantat-Gaudin data

* Cantat-Gaudin & Anders (2020):

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


* Cantat-Gaudin et al. (2020)

2017 clusters (234129 stars with P>0.7); parameters for 1867.

Mean = 125 stars per cluster (234129/1867)


## Synthetic data

Tested 4 sets of synthetic King profiles to test SVD vs ASteCA with the
following parameters:

 0. outl_perc=(10, 25); r_max_outl=1.5
 1. outl_perc=(5, 20) ; r_max_outl=1.
 2. outl_perc=(5, 10) ; r_max_outl=1.
 3. outl_perc=5       ; r_max_outl=1.

For all the sets these parameters are fixed: N_memb=200; ell_min=0.2,
ell_max=0.8; CI=0.5.

Results of this analysis: not even in the most favorable run (3) is the SVD
method able to match the performance of ASteCA for either of the fitted
parameters.


## Input data

Sol 1613 processed with `rad=10*r_50`, and the following filters applied:

0. Only for frames with >1000 stars
1. Plx filter: +/- 0.25 (using CG20 mean values)
2. PMs filter: +/- 1 (using CG20 mean values)
3. Center (lon, lat) in ASteCA fixed to CG20 values
4. Auto ASteCA radius
