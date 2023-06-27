
import numpy as np
from scipy.spatial.distance import cdist
from astropy.table import Table
import pinky

# Fixed parameters
xmax, ymax = 4, 4.
cl_cent = (0, 0)
rt_fix = 1.
N_grid = 5
N_repeat = 2
N_clust = 200
CI = 0.5
ell_min, ell_max = 0.2, 0.8

# Range percentage of outliers
outl_perc = (4.95, 5.05)
# Number of 'rt_fix' that delimitate the area where outliers are generated
r_max_outl = 1.

# Assign a number to the run to differentiate from other runs
run = 3
out_path = f"../0_data/KP_synth_{run}/"


def main():
    """
    Generate synthetic clusters with elongated and rotated King profiles.
    """
    print("Generating {} synthetic files".format(N_grid**3))

    area_cl = np.pi * rt_fix**2
    N_fl, x_fl, y_fl = estimateNfield(
        N_clust, CI, area_cl, xmax, ymax)
    _id_fl = ["1" + str(_) for _ in range(len(x_fl))]

    for kcp in np.linspace(.04, .99, N_grid):
        rc = rt_fix / 10 ** kcp
        for ell in np.linspace(ell_min, ell_max, N_grid):
            for theta in np.linspace(-np.pi / 2., np.pi / 2., N_grid):
                print(ell, theta)

                for _ in range(N_repeat):

                    x_cl, y_cl = invTrnsfSmpl_ellip(
                        cl_cent, rc, rt_fix, ell, theta, N_clust)

                    # Add outliers
                    N_outl, x_cl, y_cl = addOutliers(x_cl, y_cl)

                    _id_cl = ["2" + str(_ + N_fl) for _ in range(len(x_cl))]

                    # Add field stars
                    x, y = np.array(x_cl + x_fl), np.array(y_cl + y_fl)

                    # Filler data
                    N = len(x)
                    Gmag = np.random.uniform(12, 20, N)
                    e_Gmag = np.random.uniform(0.0001, 0.003, N)
                    BP_RP = np.random.uniform(0, 0.3, N)
                    e_BP_RP = np.random.uniform(0.0001, 0.003, N)
                    _id = _id_cl + _id_fl

                    tt = Table()
                    tt['EDR3Name'], tt['GLON'], tt['GLAT'], tt['Gmag'],\
                        tt['e_Gmag'], tt['BP-RP'], tt['e_BP-RP'] =\
                        _id, x, y, Gmag, e_Gmag, BP_RP, e_BP_RP
                    tt.write(
                        out_path + "{:.4f}_{:.4f}_{:.4f}_{}_{}".format(
                            kcp, ell, theta, N_outl, _) + ".dat", format='csv')


def CIfunc(n_in_cl_reg, field_dens, area):
    """
    Contamination index
    """
    # Star density in the cluster region.
    cl_dens = n_in_cl_reg / area
    CI = field_dens / cl_dens

    return CI


def estimateNfield(N_membs, CI, tot_area, xmax, ymax):
    """
    Estimate the total number of field stars that should be generated so
    that the CI is respected.
    """
    # This is an approximation of what ASteCA's cluster radius is compared
    # to the tidal radius
    clust_rad = rt_fix * .75

    cl_area = np.pi * clust_rad**2
    tot_area = 2 * xmax * 2 * ymax

    for Nf in np.arange(10, N_membs * 100, 50):
        x_fl = list(np.random.uniform(-xmax, xmax, Nf))
        y_fl = list(np.random.uniform(-ymax, ymax, Nf))

        dist = cdist([cl_cent], np.array([x_fl, y_fl]).T)[0]
        Nf_r = (dist < clust_rad).sum()
        n_in_cl_reg = Nf_r + N_membs

        fl_dens = Nf / tot_area
        CI_i = CIfunc(n_in_cl_reg, fl_dens, cl_area)

        if abs(CI_i - CI) < 0.01:
            break

    return Nf, x_fl, y_fl


def invTrnsfSmpl_ellip(cl_cent, rc, rt, ell, theta, Nsample):
    """
    Sample King's profile using the inverse CDF method.
    """

    c1 = 1 / rc**2
    c2 = (1 - ell)**2
    c3 = -1. / np.sqrt(1 + (rt / rc)**2)

    def KP_ellip(x, y, rc, rt):
        return (1. / np.sqrt(1 + c1 * (x**2 + (y**2 / c2))) + c3)**2

    width = rt
    height = width * (1 - ell)
    x = np.linspace(-width, width, 50)
    y = np.linspace(-height, height, 50)
    XX, YY = np.meshgrid(x, y)
    P = KP_ellip(XX, YY, rc, rt)
    p = pinky.Pinky(P=P, extent=[-width, width, -height, height])

    in_cent, in_theta = (0., 0.), 0.
    xy_in_ellipse = []
    while True:
        sampled_points = p.sample(Nsample)
        msk = inEllipse(sampled_points.T, in_cent, rt, ell, in_theta)
        xy_in_ellipse += list(sampled_points[msk])
        if len(xy_in_ellipse) >= Nsample:
            break
    samples = np.array(xy_in_ellipse)[:Nsample]

    # Rotate sample via rotation matrix
    R = np.array(
        ((np.cos(theta), -np.sin(theta)),
            (np.sin(theta), np.cos(theta))))
    xy_clust = R.dot(samples.T)

    # Shift to center
    xy_clust = (xy_clust.T + cl_cent).T
    x_cl, y_cl = xy_clust

    return list(x_cl), list(y_cl)


def addOutliers(x_cl, y_cl):
    """
    """
    N_outl = int(
        len(x_cl) * np.random.uniform(outl_perc[0], outl_perc[1]) / 100.)
    xo, yo = np.random.uniform(
        -rt_fix * r_max_outl, rt_fix * r_max_outl, (2, N_outl))
    x_cl = x_cl[:len(x_cl) - N_outl] + list(xo) + x_cl[-N_outl:]
    y_cl = y_cl[:len(y_cl) - N_outl] + list(yo) + y_cl[-N_outl:]

    return N_outl, x_cl, y_cl


def inEllipse(xy_in, cl_cent, rt, ell, theta):
    """
    Source: https://stackoverflow.com/a/59718947/1391441
    """
    # Transpose
    xy = xy_in.T

    # The tidal radius 'rt' is made to represent the width ('a')
    # Width (squared)
    a2 = rt**2
    # Height (squared)
    b2 = a2 * (1. - ell)**2

    # distance between the center and the foci
    foc_dist = np.sqrt(np.abs(b2 - a2))
    # vector from center to one of the foci
    foc_vect = np.array([foc_dist * np.cos(theta), foc_dist * np.sin(theta)])
    # the two foci
    el_foc1 = cl_cent + foc_vect
    el_foc2 = cl_cent - foc_vect

    # For each x,y: calculate z as the sum of the distances to the foci;
    # np.ravel is needed to change the array of arrays (of 1 element) into a
    # single array. Points are exactly on the ellipse when the sum of distances
    # is equal to the width
    z = np.ravel(np.linalg.norm(xy - el_foc1, axis=-1)
                 + np.linalg.norm(xy - el_foc2, axis=-1))

    # Mask that identifies the points inside the ellipse
    in_ellip_msk = z <= 2. * rt  # np.sqrt(max(a2, b2)) * 2.

    return in_ellip_msk


if __name__ == '__main__':
    # plt.style.use('science')
    main()
