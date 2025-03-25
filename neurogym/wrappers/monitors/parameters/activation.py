from functools import partial
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import panel as pn
import torch
from bokeh.models import (  # type: ignore[attr-defined]
    Legend,
    LegendItem,
    Line,
)
from bokeh.palettes import Turbo256, linear_palette
from bokeh.plotting import ColumnDataSource, figure

import param
from torch import nn

import neurogym as ngym
from neurogym.wrappers.monitors.parameters.base import ParamMonitorBase
from neurogym.wrappers.monitors.parameters.base import PlotParams


class Trace(param.Parameterized):

    colour = param.Color(
        doc="Colour for the neuron activation trace.",
    )

    def __init__(
        self,
        trace: Line,
    ):
        super().__init__()
        self.trace = trace
        self.trace.muted_glyph = trace.glyph.clone(line_alpha=0.25)


class ActivationPlot(PlotParams):

    # btn_show_hide = pn.widgets.Button(
    #     name="",
    #     icon="caret-left",
    #     icon_size="16px",
    # )

    active_opacity = param.Magnitude(
        default=1.0,
        doc="Opacity of active traces.",
        label="Opacity of active traces",
        step=0.01,
    )

    muted_opacity = param.Magnitude(
        default=0.25,
        doc="Opacity of muted traces.",
        label="Opacity of muted traces",
        step=0.01,
    )

    traces = param.Selector(
        doc="Neuron trace selector.",
        label="Neuron",
    )

    def __init__(
        self,
        neurons: int,
        data: np.ndarray,
        labels: list[str],
    ):

        super().__init__()

        # Data
        # ==================================================
        self.colours = linear_palette(Turbo256, neurons)

        # Data source
        self.source = ColumnDataSource(data)

        # UI elements
        # ==================================================
        # Neuron labels
        self.labels = labels

        # self.btn_mute = pn.widgets.Button(
        #     name="Mute traces",
        #     width=180,
        # )

        # Bokeh figure
        self.figure = self._make_figure()

        # Activation traces
        self.traces = self._make_traces()

        # Figure legend
        self.legend = self._make_legend()

    def _make_figure(self) -> figure:
        # Figure
        # ==================================================
        p = figure(
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
            sizing_mode="stretch_width"
        )

        p.toolbar.autohide = True  # type: ignore[attr-defined]
        p.border_fill_color = "white"  # type: ignore[assignment]
        p.background_fill_color = "white"  # type: ignore[assignment]
        p.outline_line_color = "black"  # type: ignore[assignment]
        p.grid.grid_line_color = None
        p.hover.tooltips = [
            ("Neuron", "$name"),
            ("(x,y)", "($x, $y)"),
        ]

        p.yaxis.axis_label = "Activation"
        p.xaxis.axis_label = "Time [ms]"

        return p

    @param.depends("bg_colour", watch=True)
    def _change_bg_colour(self):
        self.figure.background_fill_color = self.bg_colour

    def _make_traces(self) -> list[Line]:
        """Traces for neuron activations.

        Returns:
            list[Line]:
                The activation traces as a list of Line instances.
        """

        # Activation traces as lines
        traces = [
            self.figure.line(
                x="x",
                y=label,
                source=self.source,
                color=self.colours[n],
                name=label,
                line_alpha=1.0,
                line_width=1.0,
            )
            for n, label in enumerate(self.labels)
        ]

        return traces

    def _make_legend(self) -> Legend:
        """An interactive legend for the traces.

        Returns:
            Legend:
                The figure legend.
        """
        legend = Legend(
            items=[
                LegendItem(label=label, renderers=[self.traces[n]])
                for n, label in enumerate(self.labels)
            ],
            nrows=20,
            label_height=8,
            padding=0,
            orientation="horizontal",
            title="Neurons",
            label_text_font_size="8pt",
        )

        self.figure.add_layout(legend, "right")
        self.figure.legend.click_policy = "mute"

        return legend

    @param.depends("muted", watch=True)
    def _toggle_traces(self):

        for item in self.legend.items:
            for renderer in item.renderers:
                renderer.muted = self.muted

        self.btn_mute.label = (
            "Highlight all traces" if self.muted else "Mute all traces"
        )

    def _set_highlighted_trace_alpha(
        self,
        traces: list[Line],
    ):
        def _inner(
            attr: str,  # noqa: ARG001
            old: float,  # noqa: ARG001
            new: float,
        ):
            for ln in traces:
                ln.glyph.line_alpha = new

        return _inner

    def _set_muted_trace_alpha(
        self,
        traces: list[Line],
    ):
        def _inner(
            attr: str,  # noqa: ARG001
            old: float,  # noqa: ARG001
            new: float,
        ):
            for ln in traces:
                ln.muted_glyph.line_alpha = new

        return _inner

    # def _sync_cp_line_colour(self):
    #     def _inner(
    #         attr: str,  # noqa: ARG001
    #         old: int,  # noqa: ARG001
    #         new: int,  # noqa: ARG001
    #     ):
    #         cp_line.color = traces[selector.value].glyph.line_color

    #     return _inner

    def _set_line_colour(
        self,
        traces: list,
        selector: pn.widgets.Select,
    ):
        def _inner(
            attr: str,  # noqa: ARG001
            old: int,  # noqa: ARG001
            new: int,
        ):
            traces[selector.value].glyph.line_color = new
            traces[selector.value].muted_glyph.line_color = new

        return _inner

    def _make_sidebar(
        self,
        fig: figure,
        traces: list[Line],
        legend: Legend,
    ) -> pn.Column:

        # Control elements
        # ==================================================
        # Make a button to mute/highlight all traces
        btn_toggle_traces = pn.widgets.Button(name="Mute all traces", width=180)
        pn.bind(
            btn_toggle_traces, partial(self._toggle_traces, btn_toggle_traces, legend)
        )

        # Colour picker for the background colour
        cp_bg_colour = pn.widgets.ColorPicker(
            name="Background colour",
            value="#ffffff",
            width=180,
        )
        # cp_bg_colour.jslink(fig, value=fig.background_fill_color)

        # Slider for adjusting the alpha of highlighted traces
        sld_highlighed_trace_alpha = pn.widgets.FloatSlider(
            name="Alpha | Highlighted traces",
            start=0.0,
            end=1.0,
            step=0.01,
            value=1.0,
            width=180,
        )
        # pn.bind(sld_highlighed_trace_alpha, self._set_highlighted_trace_alpha(traces))

        # Slider for adjusting the alpha of muted traces
        sld_muted_trace_alpha = pn.widgets.FloatSlider(
            name="Alpha | Muted traces",
            start=0.0,
            end=1.0,
            step=0.01,
            value=0.25,
            width=180,
        )
        # pn.bind(sld_muted_trace_alpha, self._set_muted_trace_alpha(traces))

        # Dropdown list of neurons
        sel_neuron = pn.widgets.Select(
            name="Neuron",
            options={f"Neuron {n}": n for n in range(self.neurons)},
            value=0,
            width=180,
        )
        # Colour picker for the currently selected neuron trace
        cp_line = pn.widgets.ColorPicker(
            name="Line colour",
            value=traces[sel_neuron.value].glyph.line_color,
            width=180,
        )

        # pn.bind(sel_neuron, self._sync_cp_line_colour(cp_line, traces, sel_neuron))
        # pn.bind(cp_line, self._set_line_colour(traces, sel_neuron))

        sidebar = pn.Column(
            btn_toggle_traces,
            sld_highlighed_trace_alpha,
            sld_muted_trace_alpha,
            cp_bg_colour,
            sel_neuron,
            cp_line,
            width=180,
            margin=20,
        )

        # def _hide_sidebar(*args):
        #     sidebar.visible = not sidebar.visible
        #     btn_show_hide.icon = "caret-left" if sidebar.visible else "caret-right"

        # btn_show_hide.on_click(_hide_sidebar)
        # _hide_sidebar()

    def view(self):
        p = figure(
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

        # p.width = 1000
        # p.height = 500
        # p.sizing_mode = "stretch_both"
        p.toolbar.autohide = True  # type: ignore[attr-defined]
        p.border_fill_color = "white"  # type: ignore[assignment]
        p.background_fill_color = "white"  # type: ignore[assignment]
        p.outline_line_color = "black"  # type: ignore[assignment]
        p.grid.grid_line_color = None
        p.hover.tooltips = [
            ("Neuron", "$name"),
            ("(x,y)", "($x, $y)"),
        ]

        p.yaxis.axis_label = "Activation"
        p.xaxis.axis_label = "Time [ms]"
        return pn.Row(
            pn.Column(self.param),
            self.figure,
            # p,
        )


class ActivationMonitor(ParamMonitorBase):
    param_type: ngym.NetParam = ngym.NetParam.Activation

    def __init__(
        self,
        monitor: Any,
        layer: nn.Module,
        phases: set[ngym.MonitorPhase],
    ):
        """Activation monitoring component.

        Args:
            monitor (Any):
                The parent monitor that this component is attached to.

            layer (nn.Module):
                The layer being monitored.

            phases (set[ngym.MonitorPhase]):
                Phases during which the parameters should be monitored.
        """
        super().__init__(monitor, layer, phases)

        # The neuron count
        self.neurons = self._get_neuron_count()

    def process(
        self,
        module: nn.Module,  # noqa: ARG002
        input_: torch.Tensor,  # noqa: ARG002
        output: torch.Tensor,
    ):
        """This method processes new activations.

        Args:
            module (nn.Module):
                The module that this hook is registered with.

            input_ (torch.Tensor):
                The current input provided to the module.

            output (torch.Tensor):
                The module's output.
        """
        self.history[self.phase][-1].append(output)

    def _prepare_data(
        self,
        activations: list[list[np.ndarray]],
    ):

        # Prepare the activations for plotting.
        # ==================================================
        # Turn the activations into an Awkward array.
        # This is necessary to handle ragged arrays.
        activations = ak.Array([act for act in activations if len(act) > 0])
        max_len = max([len(a) for a in activations])
        activations = ak.fill_none(
            ak.pad_none(activations, max_len),
            np.array([0, 0]),
        ).to_numpy()

        mean = activations.mean(axis=0)

        data = {"x": np.arange(len(mean)) * ngym.conf.monitor.dt}
        labels = [f"{n}" for n in range(self.neurons)]
        data.update({lbl: mean[:, n] for n, lbl in enumerate(labels)})

        return data, labels

    def _make_plot(
        self,
        activations: list[list[np.ndarray]],
    ) -> figure:
        """Make and return a Bokeh figure for plotting neuron activations.

        Args:
            activations (list[list[np.ndarray]]):
                A (ragged) list of neuron activations for a single trial.

        Returns:
            figure:
                The plot and all knobs wrapped in a Panel pane.
        """

        # Prepare the activations for plotting
        data, labels = self._prepare_data(activations)

        # Plot with associated parameters
        plot = ActivationPlot(self.neurons, data, labels)

        return plot.view()

    def plot(self) -> pn.Tabs:
        """A tab panel with results for different phases.

        Returns:
            Tab:
                A tab panel.
        """

        # Separate tabs for each phase.
        tabs = []

        for phase, data in self.history.items():
            tabs.append((phase.capitalize(), self._make_plot(data)))

        return pn.Tabs(
            *tabs,
            tabs_location="left",
        )
