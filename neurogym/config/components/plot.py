from matplotlib.font_manager import FontProperties

from neurogym.config.base import ConfigBase


def _default_font_properties(size: float = 7.5):
    return FontProperties(size=size, family=["DejaVu Sans", "Noto Sans", "sans-serif"], weight="book")


class PlotFontConfig(ConfigBase):
    """Font configuration for all plots."""

    xtick: FontProperties = _default_font_properties()
    ytick: FontProperties = _default_font_properties()
    title: FontProperties = _default_font_properties(size=12)
    subtitle: FontProperties = _default_font_properties(size=10)
    label: FontProperties = _default_font_properties(size=9)
    legend: FontProperties = _default_font_properties()


class PlotLegendConfig(ConfigBase):
    """Font configuration for all plots.

    Attributes:
        location: Legend location relative to the axis.
        bbox_to_anchor: Bounding box parameters (x, y, width, height).
        mode: The legend expansion mode.

    """

    location: str = "upper right"
    bbox_to_anchor: tuple[float, ...] = (1.01, 1.0, 0.15, 0)
    mode: str | None = "expand"


class PlotConfig(ConfigBase):
    """Plotting configuration during monitoring.

    Attributes:
        font: Font configuration for plots, legends, etc.
        legend: Legend configuration.
    """

    font: PlotFontConfig = PlotFontConfig()
    legend: PlotLegendConfig = PlotLegendConfig()
