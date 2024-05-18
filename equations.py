import torch


G = 4.302e-6


def vel(v):
    return v


def der_vel(x, args):
    m, rho_0, r_c, epsilon, interactions, NFW, device = args

    if interactions:
        # Interactions between particles:
        # The idea is to have arrays of size (NxNx3). These arrays can exceed the available memory, so they are divided into
        # several arrays (N x batch_size x 3), called batches. The acceleration is then calculated for each batch. Then, I sum
        # across the columns to obtain the acceleration of particle N_i affected by batch_size particles. A loop iterates through
        # all the batches to obtain the total acceleration.

        batch_size = 1024
        N_part = len(x[:, 0])

        n_batches = N_part // batch_size
        if N_part < batch_size:
            n_batches = 1

        a_int = torch.zeros(N_part, len(x[0, :])).to(device)

        for j in range(n_batches):
            start_idx = j * batch_size
            end_idx = min((j + 1) * batch_size, N_part)
            xj_batch = x[start_idx:end_idx]
            m_batch = m[start_idx:end_idx]
            xi_batch = x[:, None, :].expand(-1, len(m_batch), -1)

            mass = m_batch[None, :, None]

            delta_pos = torch.sqrt(
                (xi_batch[:, :, 0] - xj_batch[None, :, 0]) ** 2
                + (xi_batch[:, :, 1] - xj_batch[None, :, 1]) ** 2
                + (xi_batch[:, :, 2] - xj_batch[None, :, 2]) ** 2
            )

            a_int = (
                torch.sum(
                    -G
                    * mass
                    * (xi_batch - xj_batch[None, :, :])
                    / ((delta_pos[:, :, None] + epsilon) ** 3),
                    dim=1,
                )
                + a_int
            )

    else:
        a_int = torch.zeros_like(x)
    if NFW:
        # Aceleration due to NFW:
        r = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2)
        r0 = r + r_c
        M = 4 * torch.pi * rho_0 * r_c**3 * (torch.log(r0 / r_c) - r / r0)
        a_NFW = -G * M[:, None] * x / (r[:, None] ** 3)
    else:
        a_NFW = torch.zeros_like(x)

    return a_NFW + a_int
