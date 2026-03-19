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
        ax.set_ylim(gs[i] - 1.5*spread, gs[i] + 1.5*spread)

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
    fig, axs = plt.subplots(nrows, ncols, figsize=(10, 4*nrows))

    # Ensure axs is 2D even if nrows=1
    if nrows == 1:
        axs = np.expand_dims(axs, axis=0)

    for row_idx, i in enumerate(param_indices):
        # Plot parameter convergence
        ax_param = axs[row_idx, 0]
        ax_param.hlines(gs[i], ils[0], ils[-1], color="black", label=f"g{i + 1}")
        ax_param.plot(ils, cs[:, i], label=f"c{i + 1}")
        spread = max(np.max(np.abs(cs[:, i] - gs[i])), 1e-6)
        ax_param.set_ylim(gs[i] - 1.5*spread, gs[i] + 1.5*spread)
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