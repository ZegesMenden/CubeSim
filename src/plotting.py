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
    op = GroundtrackPlotter()

    return op.plot(EarthSatellite(orbit, Spacecraft(1.0 * (u.m**2), 1.0, 1.0 * u.kg)), t_span=t_span, color=color, label=label)

def GenerateScatterTrace(t_range: astime.TimeDelta|astime.Time, data: list[any]|list[list[any]], label: str = None, color: str = "red"):

    if not isinstance(data, list):
        raise TypeError("Data must be a list of lists")
    
    if not isinstance(data[0], list):
        data = [data]

    dataRange = len(data[0])
    for d in data:
        if len(d) != dataRange:
            raise ValueError("Data lists must be of the same length")
    
    t_span = np.linspace(0, t_range.value, dataRange)
    
    figList = [go.Scatter(x=t_span, y=d, mode="lines") for d in data]
    return figList

def RenderPlotsToHTML(plots: list[go.Scatter | go.Histogram], filename: str = "plot.html"):

    fig = {
        "data": [plots[i] for i in range(len(plots))],
        "layout": go.Layout(title="My Scatter Plot"),
    }

    plotly.offline.plot(fig, filename=f"{filename.strip('.html')}.html", auto_open=True)
