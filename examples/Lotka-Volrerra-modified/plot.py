import numpy as np
from matplotlib import pyplot as plt


def plot(cs, u_errors, gs, tls, param_indices=None):
    """
    cs : (num_iters, num_params)
    gs : (num_params,)
    tls : time array
    param_indices : list of indices of parameters to plot (optional)
    """
    cs = np.nan_to_num(np.asarray(cs), nan=0.0, posinf=1e6, neginf=-1e6)
    gs = np.asarray(gs)

    num_iters, num_params = cs.shape
    ils = np.arange(num_iters)

    if param_indices is None:
        param_indices = list(range(num_params))

    ncols = 3
    nrows = int(np.ceil((len(param_indices) + 1) / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axs = axs.flatten()

    for plot_idx, i in enumerate(param_indices):
        ax = axs[plot_idx]
        ax.hlines(gs[i], ils[0], ils[-1], color="black", label=f"g{i + 1}")
        ax.plot(ils, cs[:, i], label=f"c{i + 1}")

        # Center axis around true value
        spread = np.max(np.abs(cs[:, i] - gs[i]))
        spread = max(spread, 1e-6)
        ax.set_ylim(gs[i] - 1.5 * spread, gs[i] + 1.5 * spread)

        ax.set_title(f"c{i + 1} vs g{i + 1}")
        ax.set_xlabel("Iteration")
        ax.legend()

    # Plot relative error
    ax = axs[len(param_indices)]
    ax.plot(tls[1:], u_errors)
    ax.set_yscale("log")
    ax.set_title("Relative error in u")
    ax.set_xlabel("Time")

    # Remove any unused axes
    for j in range(len(param_indices) + 1, len(axs)):
        fig.delaxes(axs[j])

    fig.tight_layout()
    return fig, axs


def plot_with_param_errors(cs, gs, tls, param_indices=None):
    """
    Plots each selected parameter alongside its log-scale error.

    cs : (num_iters, num_params) assimilated parameter values over iterations
    gs : (num_params,) true parameter values
    tls : time array corresponding to u_errors
    param_indices : list of indices of parameters to plot (optional)
    """
    cs = np.nan_to_num(np.asarray(cs), nan=0.0, posinf=1e6, neginf=-1e6)
    gs = np.asarray(gs)

    num_iters, num_params = cs.shape
    ils = np.arange(num_iters)

    if param_indices is None:
        param_indices = list(range(num_params))

    nrows = len(param_indices)
    ncols = 2  # one for the parameter, one for its error
    fig, axs = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows))

    # Ensure axs is 2D even if nrows=1
    if nrows == 1:
        axs = np.expand_dims(axs, axis=0)

    for row_idx, i in enumerate(param_indices):
        # Plot parameter convergence
        ax_param = axs[row_idx, 0]
        ax_param.hlines(
            gs[i], ils[0], ils[-1], color="black", label=f"g{i + 1}"
        )
        ax_param.plot(ils, cs[:, i], label=f"c{i + 1}")
        spread = max(np.max(np.abs(cs[:, i] - gs[i])), 1e-6)
        ax_param.set_ylim(gs[i] - 1.5 * spread, gs[i] + 1.5 * spread)
        ax_param.set_title(f"Parameter c{i + 1} vs g{i + 1}")
        ax_param.set_xlabel("Iteration")
        ax_param.set_ylabel("Value")
        ax_param.legend()

        # Plot parameter error on log scale
        ax_err = axs[row_idx, 1]
        param_error = np.abs(cs[:, i] - gs[i])
        ax_err.plot(ils, param_error)
        ax_err.set_yscale("log")
        ax_err.set_title(f"Error for c{i + 1}")
        ax_err.set_xlabel("Iteration")
        ax_err.set_ylabel("|c{i+1} - g{i+1}|")

    fig.tight_layout()
    return fig, axs


def plot_species_parameters(
    cs, gs, tls, species_index, N, u_errors=None, include_K=True
):
    """Plot error and per-species parameter convergence for one population.

    cs : (num_iters, num_params)
    gs : (num_params,)
    tls : time array
    species_index : index of species to plot (0..N-1)
    N : number of species in model
    u_errors : optional state error array over time (length len(tls)-1)
    include_K : bool, if True, also plot K row.
    """
    cs = np.nan_to_num(np.asarray(cs), nan=0.0, posinf=1e6, neginf=-1e6)
    gs = np.asarray(gs)

    num_iters, num_params = cs.shape
    assert 0 <= species_index < N, "species_index must be in [0, N-1]"

    b0_idx = species_index
    B_start = N + species_index * N
    B_end = B_start + N
    K_start = N + N * N + species_index * N
    K_end = K_start + N
    A_start = N + 2 * N * N + species_index * N
    A_end = A_start + N

    b0_true = gs[b0_idx]
    b0_series = cs[:, b0_idx]

    B_true = gs[B_start:B_end]
    B_series = cs[:, B_start:B_end]

    K_true = gs[K_start:K_end]
    K_series = cs[:, K_start:K_end]

    A_true = gs[A_start:A_end]
    A_series = cs[:, A_start:A_end]

    n_plots = 3 + (1 if include_K else 0) + (1 if u_errors is not None else 0)
    fig, axs = plt.subplots(
        n_plots, 1, figsize=(12, 4 * n_plots), squeeze=False
    )

    row = 0
    if u_errors is not None:
        if len(u_errors) == len(tls) - 1:
            t_err = tls[1:]
        elif len(u_errors) == len(tls):
            t_err = tls
        else:
            t_err = np.arange(len(u_errors))
        ax = axs[row, 0]
        ax.plot(t_err, u_errors, color="tab:blue")
        ax.set_yscale("log")
        ax.set_title(f"State error for species {species_index}")
        ax.set_xlabel("time")
        ax.set_ylabel("u error")
        ax.grid(True)
        row += 1

    ax = axs[row, 0]
    ax.plot(np.arange(num_iters), b0_series, label=f"c_b0[{species_index}]")
    ax.hlines(
        b0_true,
        0,
        num_iters - 1,
        color="k",
        linestyle="--",
        label=f"g_b0[{species_index}]",
    )
    ax.set_title(f"b0 (base growth) for species {species_index}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("value")
    ax.legend()
    ax.grid(True)
    row += 1

    ax = axs[row, 0]
    for j in range(N):
        ax.plot(
            np.arange(num_iters),
            B_series[:, j],
            label=f"c_b[{species_index},{j}]",
        )
        ax.hlines(
            B_true[j], 0, num_iters - 1, color="k", linestyle="--", alpha=0.4
        )
    ax.set_title(f"Monod b_ij row for species {species_index}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("value")
    ax.legend(fontsize="small", ncol=min(4, N))
    ax.grid(True)
    row += 1

    if include_K:
        ax = axs[row, 0]
        for j in range(N):
            ax.plot(
                np.arange(num_iters),
                K_series[:, j],
                label=f"c_K[{species_index},{j}]",
            )
            ax.hlines(
                K_true[j],
                0,
                num_iters - 1,
                color="k",
                linestyle="--",
                alpha=0.4,
            )
        ax.set_title(f"K_ij row for species {species_index}")
        ax.set_xlabel("iteration")
        ax.set_ylabel("value")
        ax.legend(fontsize="small", ncol=min(4, N))
        ax.grid(True)
        row += 1

    ax = axs[row, 0]
    for j in range(N):
        ax.plot(
            np.arange(num_iters),
            A_series[:, j],
            label=f"c_A[{species_index},{j}]",
        )
        ax.hlines(
            A_true[j], 0, num_iters - 1, color="k", linestyle="--", alpha=0.4
        )
    ax.set_title(f"A_ij interaction row for species {species_index}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("value")
    ax.legend(fontsize="small", ncol=min(4, N))
    ax.grid(True)

    fig.tight_layout()
    return fig, axs
