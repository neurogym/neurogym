import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf


def probit(x, beta, alpha):
    """Compute the probit (cumulative Gaussian) function.

    Parameters
    ----------
    x : float or np.ndarray
        Independent variable (e.g., signed stimulus evidence).
    beta : float
        Sensitivity (slope) of the psychometric function.
    alpha : float
        Bias term (horizontal shift of the curve).

    Returns:
    -------
    np.ndarray
        Probability of choosing the "correct" option for each x.
    """
    return 0.5 * (1 + erf((beta * x + alpha) / np.sqrt(2)))


def plot_psychometric(sig_ev, ch, ax, label=None):
    """Fit and plot a psychometric curve using the probit function.

    Parameters
    ----------
    sig_ev : np.ndarray
        Signed stimulus evidence (positive = rightward, negative = leftward).
    ch : np.ndarray
        Binary choices (1 = right, 0 = left).
    ax : matplotlib.axes.Axes
        Axes object to plot into.
    label : str, optional
        Title or label for the subplot.
    """
    popt, _ = curve_fit(probit, sig_ev, ch, maxfev=10000)
    x_fit = np.linspace(np.min(sig_ev), np.max(sig_ev), 100)
    y_fit = probit(x_fit, *popt)

    ax.plot(x_fit, y_fit, "-", label="Probit fit")

    means = []
    sems = []
    for e in np.unique(sig_ev):
        means.append(np.mean(ch[sig_ev == e]))
        sems.append(np.std(ch[sig_ev == e]) / np.sqrt(np.sum(sig_ev == e)))
    x = np.unique(sig_ev)
    plt_opts = {}
    plt_opts["linestyle"] = ""
    ax.errorbar(x, means, sems, **plt_opts)
    ax.plot([0, 0], [0, 1], "--", lw=0.2, color=(0.5, 0.5, 0.5))

    ax.plot([0, 0], [0, 1], "--", lw=0.5, color=(0.5, 0.5, 0.5))
    ax.plot([-50, 50], [0.5, 0.5], "--", lw=0.5, color=(0.5, 0.5, 0.5))

    ax.set_xlim((-52, 52))
    ax.set_ylim((0, 1))
    ax.set_xticks([-52, 0, 52])
    ax.set_xticklabels(["-1", "0", "1"])
    ax.set_xlabel("Stimulus evidence")
    ax.set_ylabel("P(Right)")
    ax.set_title(label or "")
    ax.legend().set_visible(False)
