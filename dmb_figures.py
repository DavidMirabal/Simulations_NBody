import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np


def generar_colores(N):
    valores = np.linspace(0, 1, N)
    colores = plt.cm.rainbow(valores)
    colores_hex = [to_hex(color) for color in colores]
    return colores_hex


class Figura:
    def __init__(
        self,
        width=6.4,
        ratio=1.33,
        dpi=500,
        ticks='yes',
        lw_spine=3,
        c_face='white',
        c_lines='black',
    ):
        # plt.rcParams['text.usetex'] = True
        # plt.rcParams['text.latex.preamble'] = r'\usepackage[varg]{txfonts}'
        # plt.rcParams['text.antialiased'] = True
        plt.rcParams['xtick.labelsize'] = 13
        plt.rcParams['ytick.labelsize'] = 13
        plt.rcParams['image.cmap'] = 'rainbow'

        fig = plt.figure(figsize=(width, width / ratio), dpi=dpi)

        self.fig = fig
        self.dpi = dpi
        self.ax = []
        self.ticks = ticks
        self.lw_spine = lw_spine
        self.c_face = c_face
        self.c_lines = c_lines
        self.fig.set_facecolor(self.c_face)

    def axs(self, ncols=1, nrows=1, projection3d=False):
        if projection3d:
            self.ax.append(self.fig.add_subplot(1, 1, 1, projection='3d'))
            self.ax[-1].tick_params(axis='both')
            if self.ticks == "no":
                self.ax[-1].set_xticks([])
                self.ax[-1].set_yticks([])
                self.ax[-1].set_zticks([])
            else:
                self.ax[-1].tick_params(axis='both', colors=self.c_lines)

            self.ax[-1].set_facecolor(self.c_face)
            for spine in self.ax[-1].spines.values():
                spine.set_linewidth(self.lw_spine)
                spine.set_color(self.c_lines)
        else:
            k = 0
            for i in range(nrows):
                for j in range(ncols):
                    self.ax.append(self.fig.add_subplot(nrows, ncols, k + 1))
                    self.ax[-1].tick_params(axis='both')
                    if self.ticks == "no":
                        self.ax[-1].set_xticks([])
                        self.ax[-1].set_yticks([])
                    else:
                        self.ax[-1].tick_params(axis='both', colors=self.c_lines)

                    self.ax[-1].set_facecolor(self.c_face)
                    for spine in self.ax[-1].spines.values():
                        spine.set_linewidth(self.lw_spine)
                        spine.set_color(self.c_lines)

                    k = k + 1

        return self.fig, self.ax
