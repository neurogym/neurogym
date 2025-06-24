import matplotlib as mpl
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from scipy.special import erf

from neurogym.utils.decorators import suppress_during_pytest


def probit(
    x: NDArray[np.float64],
    beta: float,
    alpha: float,
) -> NDArray[np.float64]:
    """Cumulative Gaussian (probit) function.

    Args:
        x: Independent variable (e.g., signed stimulus evidence).
        beta: Sensitivity (slope) of the psychometric function.
        alpha: Bias term (horizontal shift of the curve).

    Returns:
        Probability of choosing the "correct" option for each x.
    """
    return np.asarray(0.5 * (1 + erf((beta * x + alpha) / np.sqrt(2))), dtype=np.float64)


@suppress_during_pytest(
    ValueError,
    message="This may be due to a small sample size; please increase to get reasonable results.",
)
def plot_psychometric(
    sig_ev: NDArray,
    ch: NDArray,
    ax: mpl.axes.Axes,
    title: str | None = None,
    legend: str | None = None,
):
    """Fit and plot a psychometric curve using a probit function.

    Args:
        sig_ev: Signed stimulus evidence (positive = rightward, negative = leftward).
        ch: Binary choices (1 = right, 0 = left).
        ax: Axis to plot on.
        title: Title for the plot. Optional.
        legend: Label for the fit curve. Optional.
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
