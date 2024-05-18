import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

G = 4.302e-6


class ic:
    def __init__(self, N, file_name):
        self.N = int(N)
        self.file_name = file_name

        self.X = np.empty((0, 7))
        self.components = True

    def NFW(
        self, rho_0, r_c, M_disk, z_sigma, vz_sigma, N=None, r_0=0, z_0=0, rot=0
    ):  # Stable disk under an NFW profile.
        if N is None:
            N = self.N

        N = int(N)
        r = np.random.exponential(scale=r_c, size=N)
        phi = np.linspace(0, 2 * np.pi, N)
        z_p = np.random.normal(loc=0, scale=z_sigma, size=N)

        m = np.ones_like(r) * M_disk / len(r)

        M = np.empty_like(r)

        if self.components:
            for i in range(len(r)):
                M[i] = np.sum(r < r[i]) * m[i]
        else:
            for i in range(len(r)):
                r_rest = np.sqrt(
                    self.X[:, 1] ** 2 + self.X[:, 2] ** 2 + self.X[:, 3] ** 2
                )
                M[i] = np.sum(m[r < r[i]]) + np.sum(self.X[r_rest + r_0 < r[i], 0])

        x = r * np.cos(phi) + r_0 / np.sqrt(2)
        y_p = r * np.sin(phi)

        y = (
            y_p * np.cos(rot * np.pi / 180)
            + z_p * np.sin(rot * np.pi / 180)
            + r_0 / np.sqrt(2)
        )
        z = -y_p * np.sin(rot * np.pi / 180) + z_p * np.cos(rot * np.pi / 180) + z_0

        r0 = r_c + r
        dpotential = (
            4 * np.pi * G * r_c**3 * rho_0 * (-r + (r0) * np.log((r0) / r_c))
        ) / (r**2 * (r0)) + G * M / r**2

        v_c = np.sqrt(r * dpotential)
        v_x, v_y_p = -v_c * np.sin(phi), v_c * np.cos(phi) * np.cos(rot * np.pi / 180)
        v_z_p = np.random.normal(loc=0, scale=vz_sigma, size=N)

        v_y = v_y_p * np.cos(rot * np.pi / 180) + v_z_p * np.sin(rot * np.pi / 180)
        v_z = -v_y_p * np.sin(rot * np.pi / 180) + v_z_p * np.cos(rot * np.pi / 180)

        X = np.column_stack([m, x, y, z, v_x, v_y, v_z])

        self.X = np.vstack([self.X, X])

        self.components = False

    def uni_sphere(
        self, r_e, rho_0, r_c, M_sphere, N=None, r_0=0, z_0=0
    ):  # Stable sphere under an NFW profile.

        if N is None:
            N = self.N

        N = int(N)
        N_sphere = int(6 / np.pi * N)
        square = np.random.uniform(-r_e, r_e, (N_sphere, 3))
        mask = square[:, 0] ** 2 + square[:, 1] ** 2 + square[:, 2] ** 2 < r_e**2
        x, y, z = (
            square[mask, 0] + r_0 / np.sqrt(2),
            square[mask, 1] + r_0 / np.sqrt(2),
            square[mask, 2] + z_0,
        )

        r = np.sqrt(x * x + y * y + z * z)
        phi = np.arctan2(y, x)
        theta = np.arccos(z / r)

        m = np.ones_like(r) * M_sphere / len(r)

        M = np.empty_like(r)
        if self.components:
            for i in range(len(r)):
                M[i] = np.sum(r < r[i]) * m[i]
        else:
            for i in range(len(r)):
                r_rest = np.sqrt(
                    self.X[:, 1] ** 2 + self.X[:, 2] ** 2 + self.X[:, 3] ** 2
                )
                M[i] = np.sum(m[r < r[i]]) + np.sum(self.X[r_rest + r_0 < r[i], 0])

        r0 = r_c + r
        dpotential = (
            4 * np.pi * G * r_c**3 * rho_0 * (-r + (r0) * np.log((r0) / r_c))
        ) / (r**2 * (r0)) + G * M / r**2

        phi = np.random.uniform(0, np.pi, size=len(r))
        theta = np.random.uniform(0, 2 * np.pi, size=len(r))

        v_c = np.sqrt(r * dpotential)
        v_c = np.random.normal(loc=v_c, scale=v_c / 3, size=len(r))
        v_x, v_y = v_c * np.sin(theta) * np.cos(phi), v_c * np.sin(theta) * np.sin(phi)
        v_z = v_c * np.cos(theta)

        X = np.column_stack([m, x, y, z, v_x, v_y, v_z])

        self.X = np.vstack([self.X, X])

        self.components = False

    def save_file(self):
        np.savetxt(self.file_name, self.X)

    def v_curve(self):
        fig, ax = plt.subplots()
        ax.plot(
            np.sqrt(self.X[:, 1] ** 2 + self.X[:, 2] ** 2 + self.X[:, 3] ** 2),
            np.sqrt(self.X[:, 4] ** 2 + self.X[:, 5] ** 2 + self.X[:, 6] ** 2),
            marker='.',
            c='black',
            ls='',
        )
        ax.set_xlabel(r'$r$ [kpc]', fontsize=18)
        ax.set_ylabel(r'$v_c$ [km/s]', fontsize=18)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        plt.tight_layout()
        plt.show()
        return fig, ax

    def xyz(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(
            self.X[:, 1], self.X[:, 2], self.X[:, 3], marker='.', c='black', ls='-'
        )
        ax.set_xlabel(r'$X$ [kpc]')
        ax.set_ylabel(r'$Y$ [kpc]')
        ax.set_zlabel(r'$Z$ [kpc]')
        ax.axis('equal')
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        plt.tight_layout()
        plt.show()

        return fig, ax
