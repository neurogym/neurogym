import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf


def probit(x, beta, alpha):
    """Cumulative Gaussian (probit) function.

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


def plot_psychometric(sig_ev, ch, ax, title=None, legend=None):
    """Fit and plot a psychometric curve using a probit function.

    Parameters
    ----------
    sig_ev : np.ndarray
        Signed stimulus evidence (positive = rightward, negative = leftward).
    ch : np.ndarray
        Binary choices (1 = right, 0 = left).
    ax : matplotlib.axes.Axes
        Axis to plot on.
    title : str, optional
        Title for the plot.
    legend : str, optional
        Label for the fit curve.
    """
    # Fit the probit model to data
    popt, _ = curve_fit(probit, sig_ev, ch, maxfev=10000)

    # Plot smooth fit line
    x_fit = np.linspace(np.min(sig_ev), np.max(sig_ev), 300)
    y_fit = probit(x_fit, *popt)
    (line,) = ax.plot(x_fit, y_fit, label=legend)

    # Match error bar color to fit line
    fit_color = line.get_color()

    # Compute bin means and SEMs
    unique_evidence = np.unique(sig_ev)
    mean_p_right = [np.mean(ch[sig_ev == e]) for e in unique_evidence]
    sem_p_right = [np.std(ch[sig_ev == e]) / np.sqrt(np.sum(sig_ev == e)) for e in unique_evidence]

    # Plot error bars
    ax.errorbar(
        unique_evidence,
        mean_p_right,
        yerr=sem_p_right,
        fmt="o",
        color=fit_color,
        ecolor=fit_color,
        capsize=3,
        label=None,
    )

    # Reference lines
    ax.axvline(0, linestyle="--", linewidth=0.5, color="gray")
    ax.axhline(0.5, linestyle="--", linewidth=0.5, color="gray")

    # Axis formatting
    ax.set_xlim((-52, 52))
    ax.set_ylim((-0.05, 1.05))
    ax.set_xticks([-52, 0, 52])
    ax.set_xticklabels(["-1", "0", "1"])
    ax.set_xlabel("Stimulus evidence")
    ax.set_ylabel("P(Right)")
    ax.set_title(title or "")

    if legend:
        ax.legend()
