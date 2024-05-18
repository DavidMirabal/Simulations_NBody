import matplotlib.pyplot as plt

import dmb_figures as dmb

colors = {'black': 'white', 'white': 'black'}


class single_plot2D:
    def __init__(self, ticks='yes', width=8, ratio=1, c_face='black'):
        FIGURE = dmb.Figura(
            width=width,
            ticks=ticks,
            lw_spine=3,
            ratio=ratio,
            c_face=c_face,
            c_lines=colors[c_face],
        )
        self.fig, self.ax = FIGURE.axs()

    def return_figure(self):
        return self.fig, self.ax

    def update(self, plot, ax, x, xlim, ylim):
        ax[0].set_xlim(-xlim, xlim)
        ax[0].set_ylim(-ylim, ylim)

        plot.set_xdata(x[:, 0].cpu())
        plot.set_ydata(x[:, 1].cpu())


class single_plot3D:
    def __init__(self, ticks='yes', width=8, ratio=1, c_face='black'):
        FIGURE = dmb.Figura(
            width=width,
            ticks=ticks,
            lw_spine=3,
            ratio=ratio,
            c_face=c_face,
            c_lines=colors[c_face],
        )
        self.fig, self.ax = FIGURE.axs(projection3d=True)

    def return_figure(self):
        return self.fig, self.ax

    def update(self, plot, ax, x, xlim, ylim):
        ax[0].set_xlim(-xlim, xlim)
        ax[0].set_ylim(-ylim, ylim)
        ax[0].set_zlim(-ylim, ylim)

        plot._verts3d = (x[:, 0].cpu(), x[:, 1].cpu(), x[:, 2].cpu())


class triple_plot2D:
    def __init__(self, ticks='yes', width=8, ratio=2.7, c_face='black'):
        FIGURE = dmb.Figura(
            width=width,
            ticks=ticks,
            lw_spine=1,
            ratio=ratio,
            c_face=c_face,
            c_lines=colors[c_face],
        )
        self.fig, self.ax = FIGURE.axs(ncols=3)

    def return_figure(self):
        return self.fig, self.ax

    def update(self, plot, ax, x, xlim, ylim):
        for axis in ax:
            axis.set_xlim(-xlim, xlim)
            axis.set_ylim(-ylim, ylim)

        plot[0].set_xdata(x[:, 0].cpu())
        plot[0].set_ydata(x[:, 1].cpu())

        plot[1].set_xdata(x[:, 0].cpu())
        plot[1].set_ydata(x[:, 2].cpu())

        plot[2].set_xdata(x[:, 1].cpu())
        plot[2].set_ydata(x[:, 2].cpu())
