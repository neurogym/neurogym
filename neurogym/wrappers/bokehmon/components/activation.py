# --------------------------------------
import numpy as np

# --------------------------------------
import torch
from torch import nn

# --------------------------------------
from functools import partial

# --------------------------------------
from bokeh.models import Button
from bokeh.models import Legend
from bokeh.models import Select
from bokeh.models import Slider
from bokeh.models import Line
from bokeh.models import TabPanel
from bokeh.models import LegendItem
from bokeh.models import ColorPicker
from bokeh.models import Row
from bokeh.models import Column
from bokeh.palettes import Turbo256
from bokeh.palettes import linear_palette
from bokeh.plotting import ColumnDataSource
from bokeh.plotting import figure

# --------------------------------------
from neurogym import conf
from neurogym.wrappers.bokehmon.components.base import ComponentBase


class ActivationComponent(ComponentBase):

    def __init__(
        self,
        module: nn.Module,
    ):
        '''
        Activation monitoring component.

        Args:
            module (nn.Module):
                The module being monitored.
        '''

        super().__init__(module)

        # The neuron count
        self.neurons = self._get_neuron_count()

        # Data
        # ==================================================
        self.activations: list[list] = []

        # UI elements
        # ==================================================
        self.colours = linear_palette(Turbo256, self.neurons)

        # State variables
        # ==================================================
        self.traces_muted = False

    def _get_neuron_count(self) -> int:
        '''
        Method

        Returns:
            int: _description_
        '''
        return self.module.out_features

    def _start_trial(self):
        """
        Start monitoring parameters for a new trial.
        """

        # Add a new entry in the activation tracker.
        self.activations.append([])

    def _make_figure(self) -> figure:
        """
        Make and return a Bokeh figure for plotting neuron activations.

        Returns:
            figure:
                The figure.
        """

        # Figure
        # ==================================================
        fig = figure(
            output_backend="webgl",
            tools=[
                "hover",
                "pan",
                "box_select",
                "box_zoom",
                "wheel_zoom",
                "reset",
                "save",
            ],
        )

        fig.width = 1000
        fig.height = 500
        fig.toolbar.autohide = True
        fig.border_fill_color = "white"
        fig.background_fill_color = "white"
        fig.outline_line_color = "black"
        fig.grid.grid_line_color = None
        fig.hover.tooltips = [
            ("Neuron", "$name"),
            ("(x,y)", "($x, $y)"),
        ]

        fig.yaxis.axis_label = "Activation"
        fig.xaxis.axis_label = "Time [ms]"

        return fig

    def _make_traces(
        self,
        fig: figure,
        source: ColumnDataSource,
        labels: list[str],
    ) -> list[Line]:
        """
        Plot traces for neuron activations.

        Args:
            fig (figure):
                The figure to plot to.

            source (ColumnDataSource):
                The data source.

            labels (list[str]):
                Neuron labels.

        Returns:
            list[Line]:
                The traces.
        """

        # Activation traces as lines
        traces = [
            fig.line(
                x="x",
                y=label,
                source=source,
                color=self.colours[n],
                name=label,
                line_alpha=1.0,
                line_width=1.0,
            )
            for n, label in enumerate(labels)
        ]

        for trace in traces:
            trace.muted_glyph = trace.glyph.clone(line_alpha=0.25)

        return traces

    def _make_legend(
        self,
        fig: figure,
        traces: list[Line],
        labels: list[str],
    ) -> Legend:
        """
        An interactive legend for the traces.

        Args:
            fig (figure):
                The figure to attach the legend to.

            traces (list[Line]):
                Traces to add to the legend.

            labels (list[str]):
                Neuron labels.

        Returns:
            Legend:
                The legend.
        """

        legend = Legend(
            items=[
                LegendItem(label=label, renderers=[traces[n]])
                for n, label in enumerate(labels)
            ],
            nrows=20,
            label_height=8,
            padding=0,
            orientation="horizontal",
            title="Neurons",
            label_text_font_size="8pt",
        )

        fig.add_layout(legend, "right")
        fig.legend.click_policy = "mute"

        return legend

    def _toggle_traces(
        self,
        btn: Button,
        legend: Legend,
    ):
        self.traces_muted = not self.traces_muted

        for item in legend.items:
            for renderer in item.renderers:
                renderer.muted = self.traces_muted

        btn.label = "Highlight all traces" if self.traces_muted else "Mute all traces"

    def _set_highlighted_trace_alpha(
        self,
        traces: list[Line],
    ):
        def _inner(attr: str, old: float, new: float):
            for ln in traces:
                ln.glyph.line_alpha = new

        return _inner

    def _set_muted_trace_alpha(
        self,
        traces: list[Line],
    ):
        def _inner(attr: str, old: float, new: float):
            for ln in traces:
                ln.muted_glyph.line_alpha = new

        return _inner

    def _sync_cp_line_colour(
        self,
        cp_line: ColorPicker,
        traces: list[Line],
        selector: Select,
    ):
        def _inner(attr: str, old: int, new: int):
            cp_line.color = traces[selector.value].glyph.line_color

        return _inner

    def _set_line_colour(
        self,
        traces: list,
        selector: Select,
    ):
        def _inner(attr: str, old: int, new: int):
            traces[selector.value].glyph.line_color = new
            traces[selector.value].muted_glyph.line_color = new

        return _inner

    def process(
        self,
        module: nn.Module,
        input_: torch.Tensor,
        output: torch.Tensor,
        trial: int,
        step: int,
    ):
        """
        This method processes new activations.

        Args:
            module (nn.Module):
                The module that this hook is registered with.

            input_ (torch.Tensor):
                The current input provided to the module.

            output (torch.Tensor):
                The module's output.

            trial (int):
                The current trial.

            step (int):
                The current step in the current trial.
        """

        self.activations[-1].append(output)

    def _plot(self) -> TabPanel:
        """
        Render this component as a tab.

        Returns:
            TabPanel:
                The current component as a tab.
        """

        # Prepare the activations for plotting.
        # ==================================================
        # Eliminate empty activation sets
        activations = [a for a in self.activations if len(a) > 0]
        max_len = max([len(arr) for arr in activations])
        trials = [np.array(a) for a in activations]
        mean = np.array([trial.mean(axis=1) for trial in trials])
        # sd = np.sqrt(np.array([((trial - mean) ** 2).mean(axis=0) for trial in trials]).mean())
        # trials = [(trial - mean) / sd for trial in trials]

        # mean, sd = trials.mean(axis=2), trials.std(axis=2)
        data = {"x": np.arange(max_len) * conf.monitor.dt}
        labels = [f"{l}" for l in range(self.neurons)]
        data.update({l: mean[n] for n, l in enumerate(labels)})

        # Columnar data source
        source = ColumnDataSource(data)

        # Figure, traces and legend
        # ==================================================
        fig = self._make_figure()
        traces = self._make_traces(fig, source, labels)
        legend = self._make_legend(fig, traces, labels)

        # Control elements
        # ==================================================
        # Make a button to mute/highlight all traces
        btn_toggle_traces = Button(label="Mute all traces", width=180)
        btn_toggle_traces.on_click(
            partial(self._toggle_traces, btn_toggle_traces, legend)
        )

        # Colour picker for the background colour
        cp_bg_colour = ColorPicker(
            title="Background colour",
            color="#ffffff",
            width=180,
        )
        cp_bg_colour.js_link("color", fig, "background_fill_color")

        # Slider for adjusting the alpha of highlighted traces
        sld_highlighed_trace_alpha = Slider(
            title="Alpha | Highlighted traces",
            start=0.0,
            end=1.0,
            step=0.01,
            value=1.0,
            width=180,
        )
        sld_highlighed_trace_alpha.on_change(
            "value_throttled", self._set_highlighted_trace_alpha(traces)
        )

        # Slider for adjusting the alpha of muted traces
        sld_muted_trace_alpha = Slider(
            title="Alpha | Muted traces",
            start=0.0,
            end=1.0,
            step=0.01,
            value=0.25,
            width=180,
        )
        sld_muted_trace_alpha.on_change(
            "value_throttled", self._set_muted_trace_alpha(traces)
        )

        # Dropdown list of neurons
        sel_neuron = Select(
            title="Neuron",
            options=[(n, f"Neuron {n}") for n in range(self.neurons)],
            value=0,
            width=180,
        )
        # Colour picker for the currently selected neuron trace
        cp_line = ColorPicker(
            title="Line colour",
            color=traces[sel_neuron.value].glyph.line_color,
            width=180,
        )

        sel_neuron.on_change(
            "value", self._sync_cp_line_colour(cp_line, traces, sel_neuron)
        )
        cp_line.on_change("color", self._set_line_colour(traces, sel_neuron))

        return TabPanel(
            title="Activations",
            child=Row(
                Column(
                    btn_toggle_traces,
                    sld_highlighed_trace_alpha,
                    sld_muted_trace_alpha,
                    cp_bg_colour,
                    sel_neuron,
                    cp_line,
                    width=200,
                ),
                Column(
                    fig,
                ),
            ),
        )
