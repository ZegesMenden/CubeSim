import numpy as np
import plotly.figure_factory
import plotly.graph_objects
import plotly.subplots

from astropy import units as u
from poliastro.earth.plotting import GroundtrackPlotter
from poliastro.plotting import OrbitPlotter3D, OrbitPlotter2D, StaticOrbitPlotter
import matplotlib.pyplot as plt
from poliastro.util import time_range
from poliastro.earth import EarthSatellite, Spacecraft
from poliastro import constants
import astropy.time as astime

from loguru import logger

from poliastro.twobody import Orbit
from poliastro.bodies import Earth
import plotly

import plotly.graph_objects as go

def GenerateGroundTrackPlot(orbit: Orbit, orbits: float = 1, time_offset: u.Quantity = 0 * u.s, color: str = "red", label: str = "Ground track"):

    t0 = orbit.epoch - time_offset
    t_span = time_range(t0, periods=150, end=t0 + (orbits * orbit.period) )
    gp = GroundtrackPlotter()
    gp.plot(EarthSatellite(orbit, Spacecraft(1.0 * (u.m**2), 1.0, 1.0 * u.kg)), t_span=t_span, color=color, label=label)

    dtick = 30
    # add "label" trace
    x = list(range(-180, 180 + dtick, dtick))
    y = list(range(-90, 90 + dtick, dtick))
    xpos = -175
    ypos = -85
    gp.update_layout()
    gp.add_trace(
        go.Scattergeo(
            {
                "lon": x[1:-1] + [xpos] * (len(y) - 2),
                "lat": [ypos] * (len(x) - 2) + y[1:-1],
                "showlegend": False,
                "text": x[1:-1] + y[1:-1],
                "mode": "text",
            }
        )
    )


    return gp.fig

def GenerateScatterTrace(t_range: astime.TimeDelta|astime.Time|float, data: list[any]|list[list[any]], label: str|list[str] = None, color: str|list[str] = None) -> go.Figure:

    # Add axis labels

    if not isinstance(data, list):
        raise TypeError("Data must be a list of lists")
    
    if not isinstance(data[0], list):
        data = [data]

    # Create time span for x-axis

    dataRange = len(data[0])
    if not all(dataRange == len(d) for d in data):
        raise ValueError("Data lists must be of the same length")
    
    t_span = np.linspace(0, t_range.value if isinstance(t_range, (astime.TimeDelta, astime.Time)) else t_range, dataRange)

    # Create labels and colors for each trace

    labels = None
    if isinstance(label, list):
        if len(label) != len(data):
            raise ValueError("Number of labels must match the number of data lists")    
        labels = label
    elif isinstance(label, str):
        labels = [label for _ in range(len(data))]
    else:
        labels = [f"Data {i}" for i in range(len(data))]

    colors = None
    if isinstance(color, list):
        if len(color) != len(data):
            raise ValueError("Number of colors must match the number of data lists")    
        colors = color
    elif isinstance(color, str):
        colors = [color for _ in range(len(data))]
    else:
        colors = [None for _ in range(len(data))]

    figList = [go.Scatter(x=t_span, y=d, mode="lines", name=l, line=dict(color=c)) for d, l, c in zip(data, labels, colors)]
    return figList

def RenderPlotsToHTML(figures: list[go.Scatter | go.Histogram] | go.Figure, title: str = None, x_label: str = None, y_label: str = None, filename: str = "plot.html"):
    
    if isinstance(figures, go.Figure):
        combined_figure = figures
    else:
        # Ensure figures is a list of plotly figure objects
        if not isinstance(figures, list):
            figures = [figures]
        
        # Flatten the list of figures if it contains nested lists
        flat_figures = []
        for fig in figures:
            if isinstance(fig, list):
                flat_figures.extend(fig)
            else:
                flat_figures.append(fig)
        
        # Create a single figure with all traces
        combined_figure = go.Figure()
        for fig in flat_figures:
            combined_figure.add_trace(fig)

    combined_figure.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
    
    plotly.offline.plot(combined_figure, filename=f"{filename.strip('.html')}.html", auto_open=True)

def Generate3DOrbitPlot(orbit: Orbit, label: str = "Orbit", color: str = "red"):
    fig = go.Figure()
    op = OrbitPlotter3D(fig, num_points=300)
    
    op.plot(orbit, label=label, color=color)
    fig.update_layout(title="3D Orbit Plot", scene=dict(
        xaxis_title="X (km)",
        yaxis_title="Y (km)",
        zaxis_title="Z (km)"
    ))
    op.show()

    return fig
