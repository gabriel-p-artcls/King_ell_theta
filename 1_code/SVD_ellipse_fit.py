
import os
import numpy as np
from astropy.io import ascii
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
# import matplotlib.transforms as transforms
# from skimage.measure import EllipseModel
from KP_synth_generate import N_clust

# Assign a number to the run to differentiate from other runs
run = 3
path = f"../0_data/KP_synth_{run}/"
out_folder = f"SVD_results_{run}"
out_file = f"SVD_results_{run}"


def main():
    """
    Generate SVD fit to the synthetic clusters with varying King profiles.
    """
    all_files = os.listdir(path)

    res = []
    for file in all_files:
        print(file)
        # True values
        kpc, ell_t, theta_t, N_outl, Nf = file[:-4].split('_')
        N_outl = int(N_outl)

        data = ascii.read(path + f"{file}")
        x, y = data['GLON'].data, data['GLAT'].data

        # # Confidence ellipse without outliers.
        # x_no = np.array(list(x[:N_clust - N_outl])
        #                 + list(x[N_clust:N_clust + N_outl]))
        # y_no = np.array(list(y[:N_clust - N_outl])
        #                 + list(y[N_clust:N_clust + N_outl]))
        # points = np.array([x_no, y_no]).T
        # mean_pos, width, height, theta = SigmaEllipse(points)
        # theta = thetaInRange(theta)
        # ell = 1 - height / width

        # Confidence ellipse with outliers.
        x_o, y_o = x[:N_clust], y[:N_clust]
        points = np.array([x_o, y_o]).T
        mean_pos_o, width_o, height_o, theta_o = SigmaEllipse(points)
        theta_o = thetaInRange(theta_o)
        ell_o = 1 - height_o / width_o

        res.append([
            file, kpc, ell_t, theta_t, Nf, mean_pos_o[0], mean_pos_o[1],
            width_o, height_o, ell_o, theta_o])

        # print("({:.2f}, {:.2f}); ell={:.2f}, theta={:.2f}".format(
        #     *mean_pos, ell, theta))
        # print("({:.2f}, {:.2f}); ell={:.2f}, theta={:.2f}".format(
        #     *mean_pos_o, ell_o, theta_o))
        # xo, yo = x[N_clust - N_outl:N_clust], y[N_clust - N_outl:N_clust]
        # plot(file, data['EDR3Name'].data, x, y, xo, yo, mean_pos, width,
        #      height, theta, mean_pos_o, width_o, height_o, theta_o)

    res = np.array(res).T
    tt = Table()
    tt['file'], tt['kpc'], tt['ell_t'], tt['theta_t'], tt['Nf'], tt['x0'],\
        tt['y0'], tt['width'], tt['height'], tt['ell'],\
        tt['theta'] = res
    tt.write(
        "../2_pipeline/{}/{}.dat".format(out_folder, out_file),
        format='csv', overwrite=True)


# def plot(
#     file, ID, x, y, xo, yo, mean_pos, width, height, theta, mean_pos_o,
#     width_o, height_o, theta_o
# ):
#     """
#     """
#     figsize_x, figsize_y = 8, 8
#     fig = plt.figure(figsize=(figsize_x, figsize_y))

#     xrang = x.max() - x.min()
#     yrang = y.max() - y.min()
#     xy_rang = max(xrang, yrang) * .52
#     xmid, ymid = (x.max() + x.min()) * .5, (y.max() + y.min()) * .5
#     x_lims = xmid - xy_rang, xmid + xy_rang
#     y_lims = ymid - xy_rang, ymid + xy_rang

#     ax = plt.gca()
#     plt.grid()

#     field, clust = [], []
#     for i, st in enumerate(ID):
#         if str(st)[0] == '1':
#             field.append([x[i], y[i]])
#         elif str(st)[0] == '2':
#             clust.append([x[i], y[i]])
#     field, clust = np.array(field).T, np.array(clust).T

#     # Plot the raw points
#     plt.scatter(field[0], field[1], c='grey', alpha=.7)
#     plt.scatter(clust[0], clust[1], c='green', alpha=.7)
#     # Plot outlier
#     plt.scatter(xo, yo, c='r', s=25)

#     ellipse = Ellipse(
#         xy=mean_pos, width=width, height=height, angle=theta,
#         edgecolor='k', fc='None', lw=2, zorder=4, label='SVD')
#     ax.add_patch(ellipse)
#     # plt.scatter(*mean_pos, c='g', marker='x', s=10)

#     ellipse = Ellipse(
#         xy=mean_pos_o, width=width_o, height=height_o, angle=theta_o,
#         edgecolor='r', fc='None', lw=2, zorder=4, label='SVD')
#     ax.add_patch(ellipse)
#     # plt.scatter(*mean_pos_o, c='r', marker='x', s=10)

#     #
#     # xc, yc, a, b, theta = scikitImage(points)
#     # theta = np.rad2deg(theta)
#     # theta2 = theta + 90 if b > a else theta
#     # theta2 = thetaInRange(theta2)
#     # # print(a, b)
#     # height, width = b, a
#     # if b > a:
#     #     height, width = a, b
#     # ecc = np.sqrt(1 - height**2/width**2)
#     # print("({:.2f}, {:.2f}); ecc={:.2f}, theta={:.2f}".format(
#     #     xc, yc, ecc, theta2))
#     # ell_patch = Ellipse((xc, yc), 2 * a, 2 * b, theta,
#     #                     edgecolor='b', lw=3, facecolor='none', zorder=5,
#     #                     label='sciKit')
#     # ax.add_patch(ell_patch)

#     # # Equal to SVD
#     # ell_radius_x, ell_radius_y, transf = confidence_ellipse(x, y)
#     # ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
#     #                   edgecolor='g', lw=3, facecolor='none')
#     # ellipse.set_transform(transf + ax.transData)
#     # ax.add_patch(ellipse)

#     plt.xlim(x_lims[0], x_lims[1])
#     plt.ylim(y_lims[0], y_lims[1])
#     # plt.legend()
#     # plt.show()
#     fig.tight_layout()
#     plt.savefig("../2_pipeline/{}/plots/{}.png".format(out_folder, file[:-4]))


# def get_correlated_dataset(n, dependency, mu, scale):
#     latent = np.random.randn(n, 2)
#     dependent = latent.dot(dependency)
#     scaled = dependent * scale
#     scaled_with_offset = scaled + mu
#     # return x and y of the new, correlated dataset
#     return scaled_with_offset[:, 0], scaled_with_offset[:, 1]


def thetaInRange(theta):
    """
    """
    if theta < -90:
        theta += 180
    elif theta > 90:
        theta -= 180
    return theta


def SigmaEllipse(points, Nsigma=2):
    """
    Generate a 'Nsigma' ellipse based on the mean and covariance of a point
    "cloud".

    Source: https://stackoverflow.com/a/12321306/1391441

    Parameters
    ----------
        points : An Nx2 array of the data points.
        Nsigma : probability value for the CI region.
    """
    def eigsorted(cov):
        """
        Eigenvalues and eigenvectors of the covariance matrix.
        """
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    # Location of the center of the ellipse.
    mean_pos = points.mean(axis=0)

    # The 2x2 covariance matrix to base the ellipse on.
    cov = np.cov(points, rowvar=False)

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * np.sqrt(vals) * Nsigma

    return mean_pos, width, height, theta


# def scikitImage(points):
#     """
#     Source : https://stackoverflow.com/a/58520339/1391441
#     """
#     ell = EllipseModel()
#     ell.estimate(points)
#     xc, yc, a, b, theta = ell.params

#     return xc, yc, a, b, theta


# def confidence_ellipse(x, y, n_std=1):
#     """
#     Parameters
#     ----------
#     x, y : array-like, shape (n, )
#         Input data.
#     n_std : float
#         The number of standard deviations to determine the ellipse's radiuses.

#     Source: https://matplotlib.org/3.5.0/gallery/statistics/confidence_ellipse.html#sphx-glr-gallery-statistics-confidence-ellipse-py
#     """
#     cov = np.cov(x, y)
#     pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
#     # Using a special case to obtain the eigenvalues of this
#     # two-dimensionl dataset.
#     ell_radius_x = np.sqrt(1 + pearson)
#     ell_radius_y = np.sqrt(1 - pearson)

#     # Calculating the stdandard deviation of x from
#     # the squareroot of the variance and multiplying
#     # with the given number of standard deviations.
#     scale_x = np.sqrt(cov[0, 0]) * n_std
#     mean_x = np.mean(x)

#     # calculating the stdandard deviation of y ...
#     scale_y = np.sqrt(cov[1, 1]) * n_std
#     mean_y = np.mean(y)

#     transf = transforms.Affine2D() \
#         .rotate_deg(45) \
#         .scale(scale_x, scale_y) \
#         .translate(mean_x, mean_y)

#     return ell_radius_x, ell_radius_y, transf


if __name__ == '__main__':
    main()
