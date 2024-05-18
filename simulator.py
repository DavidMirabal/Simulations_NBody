import os
import sys
import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import methods
import plotting as plott
from equations import vel, der_vel
from image_to_video import ImageToVideoConverter


class Nbody_simulation:
    def __init__(self, dir_data, dir_images):
        self.dir_data = dir_data
        self.dir_images = dir_images

    def set_config(
        self,
        rho_0,
        r_c,
        epsilon,
        interactions,
        NFW,
        t_final,
        timesteps,
        frames,
        fps,
        title,
        resize,
        device,
    ):
        self.rho_0, self.r_c, self.epsilon = rho_0, r_c, epsilon
        self.interactions, self.NFW = interactions, NFW
        self.fps, self.frames = fps, frames
        self.video_title = title
        self.resize = resize
        self.device = device

        data = np.loadtxt(self.dir_data)
        if len(np.shape(data)) == 1:
            data = data[None, :]
        self.m = torch.from_numpy(data[:, 0]).to(device)

        self.dt = t_final / timesteps
        self.dt_show = t_final / frames
        self.t_show = np.float32(0.0)
        self.next_show = 0.0

        self.x0 = torch.from_numpy(data[:, 1:]).to(device)
        self.x = torch.clone(self.x0)  # Positions[:, :3] and velocities[:, 3:] array
        self.x_line = torch.empty(frames, len(self.m), 3)

        if not (os.path.exists(self.dir_images)):
            os.makedirs(self.dir_images)

        files = os.listdir(self.dir_images)
        for file in files:
            file_path = os.path.join(self.dir_images, file)
            os.remove(file_path)

    def set_visual(
        self, plot_type, ticks, color, markersize, xlim, ylim, line, marker='.', alpha=1
    ):
        self.plot_type = plot_type
        self.xlim = xlim
        self.ylim = ylim
        self.line = line
        self.color = color
        if self.plot_type == 'single_2D':
            self.fig_class = plott.single_plot2D(ticks=ticks)
            self.fig, self.ax = self.fig_class.return_figure()

            self.text = self.fig.text(
                0.5,
                0.95,
                f't= 0 Myr',
                ha='center',
                va='center',
                color=self.color,
            )
            self.ax[0].set_xlim(-self.xlim, self.xlim)
            self.ax[0].set_ylim(-self.ylim, self.ylim)

            self.ax[0].set_aspect('equal')

            self.ax[0].set_xlabel(r'$X$ [kpc]')
            self.ax[0].set_ylabel(r'$Y$ [kpc]')

            (self.plot,) = self.ax[0].plot(
                self.x[:, 0].cpu(),
                self.x[:, 1].cpu(),
                color=color,
                ls='',
                ms=markersize,
                marker=marker,
                alpha=alpha,
            )
            if self.line:
                self.plot_line = [None] * len(self.m)
                for i in range(len(self.m)):
                    (self.plot_line[i],) = self.ax[0].plot([], color=color, alpha=0.2)

        elif self.plot_type == 'triple_2D':
            self.fig_class = plott.triple_plot2D(ticks=ticks, c_face='white')
            self.fig, self.ax = self.fig_class.return_figure()
            self.plot = [None] * len(self.ax)
            self.plot_line = [None] * len(self.ax)
            self.text = self.fig.text(
                0.37,
                0.12,
                f't= 0 Myr',
                ha='center',
                va='center',
                c=color,
            )
            (self.plot[0],) = self.ax[0].plot(
                self.x[:, 0].cpu(),
                self.x[:, 1].cpu(),
                color=color,
                ls='',
                ms=markersize,
                marker=marker,
                alpha=alpha,
            )
            (self.plot[1],) = self.ax[1].plot(
                self.x[:, 0].cpu(),
                self.x[:, 2].cpu(),
                color=color,
                ls='',
                ms=markersize,
                marker=marker,
                alpha=alpha,
            )
            (self.plot[2],) = self.ax[2].plot(
                self.x[:, 1].cpu(),
                self.x[:, 2].cpu(),
                color=color,
                ls='',
                ms=markersize,
                marker=marker,
                alpha=alpha,
            )
            if self.line:
                self.plot_line = np.empty((len(self.ax), len(self.m)), dtype=object)
            labels = [
                (r'$X$ [kpc]', r'$Y$ [kpc]'),
                (r'$X$ [kpc]', r'$Z$ [kpc]'),
                (r'$Y$ [kpc]', r'$Z$ [kpc]'),
            ]
            for i, axis in enumerate(self.ax):
                axis.set_xlim(-self.xlim, self.xlim)
                axis.set_ylim(-self.ylim, self.ylim)

                axis.set_aspect('equal')

                axis.set_xlabel(labels[i][0])
                axis.set_ylabel(labels[i][1])
                if self.line:
                    for j in range(len(self.m)):
                        (self.plot_line[i][j],) = axis.plot([], color=color, alpha=0.2)

        elif self.plot_type == 'single_3D':
            self.fig_class = plott.single_plot3D(ticks=ticks, c_face='black')
            self.fig, self.ax = self.fig_class.return_figure()

            self.text = self.fig.text(
                0.5,
                0.95,
                f't= 0 Myr',
                ha='center',
                va='center',
                color=self.color,
            )

            self.ax[0].set_xlim(-self.xlim, self.xlim)
            self.ax[0].set_ylim(-self.ylim, self.ylim)
            self.ax[0].set_zlim(-self.ylim, self.ylim)
            self.ax[0].plot(
                (-xlim, xlim, xlim, xlim, xlim, -xlim, -xlim, -xlim),
                (-ylim, -ylim, -ylim, ylim, ylim, ylim, ylim, -ylim),
                (ylim, ylim, ylim, ylim, ylim, ylim, ylim, ylim),
                marker='',
                ls='-',
                c='white',
                alpha=0.5,
            )

            self.ax[0].plot(
                (-xlim, -xlim),
                (-ylim, -ylim),
                (-ylim, ylim),
                marker='',
                ls='-',
                c='white',
                alpha=0.5,
            )
            self.ax[0].plot(
                (xlim, xlim),
                (-ylim, -ylim),
                (-ylim, ylim),
                marker='',
                ls='-',
                c='white',
                alpha=0.5,
            )

            self.ax[0].plot(
                (-xlim, -xlim),
                (ylim, ylim),
                (-ylim, ylim),
                marker='',
                ls='-',
                c='white',
                alpha=0.5,
            )
            self.ax[0].plot(
                (xlim, xlim),
                (ylim, ylim),
                (-ylim, ylim),
                marker='',
                ls='-',
                c='white',
                alpha=0.5,
            )

            self.ax[0].plot(
                (-xlim, xlim, xlim, xlim, xlim, -xlim, -xlim, -xlim),
                (-ylim, -ylim, -ylim, ylim, ylim, ylim, ylim, -ylim),
                (-ylim, -ylim, -ylim, -ylim, -ylim, -ylim, -ylim, -ylim),
                marker='',
                ls='-',
                c='white',
                alpha=0.5,
            )

            self.ax[0].set_aspect('equal')
            self.ax[0].view_init(elev=30, azim=0)
            self.ax[0].axis('off')

            (self.plot,) = self.ax[0].plot(
                self.x[:, 0].cpu(),
                self.x[:, 1].cpu(),
                self.x[:, 2].cpu(),
                color=color,
                ls='',
                ms=markersize,
                marker=marker,
                alpha=alpha,
            )
            if self.line:
                self.plot_line = [None] * len(self.m)
                for i in range(len(self.m)):
                    (self.plot_line[i],) = self.ax[0].plot([], color=color, alpha=0.2)

        self.fig.tight_layout()
        self.fig.savefig(f'{self.dir_images}/im_0.jpg')

    def iteration(
        self, i, integration_method
    ):  # to calculate the iteration i and plot and save it
        t_ex_actual = time.time()
        while self.t_show <= self.next_show:
            integrator = methods.integrate(
                (self.x[:, :3], self.x[:, 3:]),
                (vel, der_vel),
                (
                    self.m,
                    self.rho_0,
                    self.r_c,
                    self.epsilon,
                    self.interactions,
                    self.NFW,
                    self.device,
                ),
            )
            if integration_method == 'Euler':
                self.x[:, :3], self.x[:, 3:] = integrator.euler(self.dt)

            elif integration_method == 'RK2':
                self.x[:, :3], self.x[:, 3:] = integrator.runge_kutta2(self.dt)

            elif integration_method == 'RK4':
                self.x[:, :3], self.x[:, 3:] = integrator.runge_kutta4(self.dt)
            else:
                print('The integration method is not valid. Use Euler, RK2 or RK4')
                sys.exit()
            self.t_show = self.t_show + self.dt

        self.execution_time = time.time() - t_ex_actual + self.execution_time
        self.next_show = self.t_show + self.dt_show

        if self.plot_type == 'single_2D':
            self.fig_class.update(self.plot, self.ax, self.x, self.xlim, self.ylim)
            if self.line:
                self.x_line[i, :] = self.x[:, :3]
                for j in range(len(self.m)):
                    self.fig_class.update(
                        self.plot_line[j],
                        self.ax,
                        self.x_line[: i + 1, j, :],
                        self.xlim,
                        self.ylim,
                    )

        elif self.plot_type == 'triple_2D':
            self.fig_class.update(self.plot, self.ax, self.x, self.xlim, self.ylim)

            if self.line:
                self.x_line[i, :] = self.x[:, :3]
                for j in range(len(self.m)):
                    self.fig_class.update(
                        self.plot_line[:, j],
                        self.ax,
                        self.x_line[: i + 1, j, :],
                        self.xlim,
                        self.ylim,
                    )
            self.text.remove()
            self.text = self.fig.text(
                0.37,
                0.12,
                f't={self.t_show * 9.8e2:.0f} Myr',
                ha='center',
                va='center',
                color=self.color,
            )

        elif self.plot_type == 'single_3D':
            self.fig_class.update(self.plot, self.ax, self.x, self.xlim, self.ylim)
            self.ax[0].view_init(elev=30, azim=4 * np.pi * self.fps / self.frames * i)

            self.text.remove()
            self.text = self.fig.text(
                0.5,
                0.95,
                f't={self.t_show * 9.8e2:.0f} Myr',
                ha='center',
                va='center',
                color=self.color,
            )

            if self.line:
                self.x_line[i, :] = self.x[:, :3]
                for j in range(len(self.m)):
                    self.fig_class.update(
                        self.plot_line[j],
                        self.ax,
                        self.x_line[: i + 1, j, :],
                        self.xlim,
                        self.ylim,
                    )

        self.fig.savefig(f'{self.dir_images}/im_{i+1}.jpg')

    def simulate(self, integration_method):  # to run the simulation
        self.execution_time = 0.0
        for i in tqdm(range(self.frames), desc="Progress"):
            self.iteration(i, integration_method)

    def restart(self):  # to restart the simulation to the initial conditions
        self.x_line = torch.empty(self.frames, len(self.m), 3)
        self.x = self.x0
        self.t_show = 0
        self.next_show = 0.0

    def videoMaker(self, video_title=None):  # to create the video from the frames
        if video_title is None:
            video_title = self.video_title

        ImageToVideoConverter.png_to_mp4(
            self.dir_images,
            extension=".jpg",
            digit_format="01d",
            fps=self.fps,
            title=video_title,
            resize_factor=self.resize,
        )
