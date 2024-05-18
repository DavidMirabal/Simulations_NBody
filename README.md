# N-body Galaxy Simulator

Done by David Mirabal Betancort


## Requirements

The code has been tested in Python 3.10, and the necessary external packages are included in `requirements.txt`

> [!WARNING]
>  This module is optimized to use the GPU for calculations. For this, an NVIDIA GPU with CUDA drivers installed is necessary. If that's not the case, the CPU can be used for calculations (but the code will be much slower, see `Deliverable.ipynb`).

> [!WARNING]
>  To create videos of the simulations, the [ffmpeg](https://ffmpeg.org/) software is necessary. Otherwise, only frames can be generated.

## Usage

`Deliverable.ipynb` contains examples of how to use the code. The idea is to have a file with initial conditions (or create it) and then set the config of the simulation. Next, run the simulation that will generate frames. Finally you can generate a video from these frames.

## Code structure

* `simulator.py`: Creates the simulation and sets the visual and simulation settings. You can modify parameters such as $\rho_0$, $r_c$, the softening length, enable or disable interactions, and enable or disable an NFW potential, ... After applying the configuration, run the simulation.

* `methods.py`: Different methods for solving differential equations, including Euler, RK2, and RK4.

* `equations.py`: Differential equations to solve (velocity and acceleration). The parameter `batch_size` indicates how many batches the main array (NxNx3) is divided into. Each batch will have a size of (N x batch_size x 3). This parameter is set to 1024 (optimized for a GPU memory of 12 GB). Increase (decrease) this parameter if you have a GPU with more (less) memory to optimize the code. If this value is too large, the arrays will not fit in the graphics memory, and the program will crash.

* `ic_generator.py`: Generates initial conditions. You can add a rotating galaxy disk and a rotating bulge.

* `plotting.py`: Contains utility functions to create and update plots.

* `dmb_figures.py`: Used to create figures easily with the same style.

* `image_to_video.py`: Script to transform frames to a video.

* `images`: Directory with all the frames generated from the simulations carried out in `Deliverable.ipynb`.

* `videos`: Directory with all the videos generated from the simulations carried out in `Deliverable.ipynb`. Also here: [N-body playlist](https://youtube.com/playlist?list=PLmVmqY3WbRtl9BYNBvtgxdRB4fSKR4dF0&si=lfuA13CoeZ6GQ4CQ)

* `time_test`: Initial condition files used for the analysis of execution time.

* `ics`: Initial conditions of the simulations carried out in `Deliverable.ipynb`.
