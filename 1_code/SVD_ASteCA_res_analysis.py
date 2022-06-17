
from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt


path = "../2_pipeline/"
run = 3
asteca_data = path + "asteca_output_synth_{}.dat".format(run)
SVD_data = path + "SVD_results_{}/SVD_results_{}.dat".format(run, run)


def main():
    """
    Process the output data files from the SVD ans ASteCA analysis of the
    synthetic clusters with varying King profiles.
    """
    # plt.style.use('science')
    for flag in (True, False):
        figsize_x, figsize_y = 15, 8
        fig = plt.figure(figsize=(figsize_x, figsize_y))
        SVDvsTrue(flag)
        # ASteCAvsTrue()

        fig.tight_layout()
        if flag:
            plt.savefig(path + "w_outliers.png", dpi=300)
        else:
            plt.savefig(path + "no_outliers.png", dpi=300)
        breakpoint()


def ASteCAvsTrue():
    """
    """
    data = ascii.read(asteca_data)

    kpc_lst, ell_t_lst, theta_t_lst = [], [], []
    for cl in data['NAME']:
        kpc, ell_t, theta_t, Nf, _ = cl.split('/')[1:][0].split('_')
        kpc_lst.append(float(kpc))
        ell_t_lst.append(float(ell_t))
        theta_t_lst.append(float(theta_t))
    kpc_lst, ell_t_lst, theta_t_lst = [
        np.array(_) for _ in (kpc_lst, ell_t_lst, theta_t_lst)]
    theta_t_deg = np.rad2deg(theta_t_lst)

    plt.subplot(223)
    plt.title("ASteCA")
    # adiff = angleDiff(theta_t_deg, np.rad2deg(data['theta_mean']))
    adiff = angleDiff(theta_t_deg, np.rad2deg(data['theta_mode']))

    # for i, th in enumerate(theta_t_deg):
    #     print("{}: {:.2f} , {:.2f} = {:.2f}".format(
    #         data['NAME'][i], th, data['theta_mean'][i], adiff[i]))
    # breakpoint()

    plt.scatter(kpc_lst, adiff, alpha=.6, c=ell_t_lst)
    plt.axhline(np.mean(adiff), c='r')
    plt.colorbar()
    plt.ylim(0, 90)
    plt.xlabel("kpc")
    plt.ylabel("delta_theta")

    plt.subplot(224)
    plt.title("ASteCA")
    plt.scatter(
        kpc_lst, abs(ell_t_lst - data['ell_median']), alpha=.6, c=ell_t_lst)
    plt.axhline(np.mean(abs(ell_t_lst - data['ell_median'])), c='r')
    plt.colorbar()
    plt.ylim(-0.01, 0.95)
    plt.xlabel("kpc")
    plt.ylabel("ell_theta")


def SVDvsTrue(plotOutliers):
    """
    """
    data = ascii.read(SVD_data)

    theta_t_deg = np.rad2deg(data['theta_t'])

    # a_diff = angleDiff(theta_t_deg, data['theta'])
    # for i, th in enumerate(theta_t_deg):
    #     print("{}: {:.2f} , {:.2f} = {:.2f}".format(
    #         data['file'][i], th, data['theta'][i], a_diff[i]))

    plt.subplot(221)
    if plotOutliers is False:
        plt.title("SVD (no outliers)")
        adiff_1 = angleDiff(theta_t_deg, data['theta'])
        plt.scatter(data['kpc'], adiff_1, alpha=.6, c=data['ell_t'])
        plt.axhline(np.mean(adiff_1), c='r')
        plt.ylim(0, 90)
    else:
        plt.title("SVD (with outliers)")
        # with outliers
        adiff_2 = angleDiff(theta_t_deg, data['theta_o'])
        plt.scatter(data['kpc'], adiff_2, alpha=.6, c=data['ell_t'])
        plt.axhline(np.mean(adiff_2), c='r')
        plt.ylim(0, 90)
    plt.colorbar()
    plt.xlabel("kpc")
    plt.ylabel("delta_theta")

    plt.subplot(222)
    if plotOutliers is False:
        plt.title("SVD (no outliers)")
        # plt.scatter(theta_t_deg, data['theta'], alpha=.6, c='g')
        plt.scatter(data['kpc'], abs(data['ell_t'] - data['ell']), alpha=.6,
                    c=data['ell_t'])
        plt.axhline(np.median(abs(data['ell_t'] - data['ell'])), c='r')
    else:
        plt.title("SVD (with outliers)")
        # with outliers
        # plt.scatter(theta_t_deg, data['theta_o'], alpha=.6, c='r')
        plt.scatter(data['kpc'], abs(data['ell_t'] - data['ell_o']),
                    alpha=.6, c=data['ell_t'])
        plt.axhline(np.median(abs(data['ell_t'] - data['ell_o'])), c='r')
    plt.colorbar()
    plt.ylim(-0.01, 0.95)
    plt.xlabel("kpc")
    plt.ylabel("delta_ell")


def angleDiff(x, y):
    """
    https://stackoverflow.com/a/2007279/1391441

    Equivalent to:

    delta = []
    for i, xi in enumerate(x):
        if xi >= 0 and y[i] >= 0:
            delta.append(abs(xi - y[i]))
        elif xi >= 0 and y[i] < 0:
            t3, t4 = 90 - xi, 90 + y[i]
            delta.append(min(xi - y[i], t3 + t4))
        elif xi < 0 and y[i] >= 0:
            t3, t4 = 90 - y[i], 90 + xi
            delta.append(min(y[i] - xi, t3 + t4))
        elif xi < 0 and y[i] < 0:
            t3, t4 = 90 - y[i], 90 + xi
            delta.append(abs(xi - y[i]))
    return np.array(delta)

    """
    xr, yr = np.deg2rad(x), np.deg2rad(y)
    xy = xr - yr
    return abs(90 - abs(np.rad2deg(np.arctan2(np.cos(xy), np.sin(xy)))))


if __name__ == '__main__':
    main()
