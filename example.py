import src.common as common
import src.orbits as orbits
import src.plotting as plotting

import astropy.units as u
from poliastro.examples import iss

import astropy.time as astime

orbit = orbits.OrbitWrapper(orbit=iss, cd=2.2, areaMassRatio=1.0*u.km**2/u.kg)
orbit.hasSun()
global sun_array, iters
sun_array = [orbits.getSunStrength(orbit.orbit).value * orbit.hasSun()]
iters = 0
t_prop: astime.TimeDelta = (24*1) * u.h

@orbit.propCallback
def logSunStrength(orbit: orbits.OrbitWrapper) -> None:
    global sun_array, iters
    sun_array[0] = (orbits.getSunStrength(orbit.orbit).value * orbit.hasSun())
    iters += 1

orbit.propagateOrbit(t_prop)
print(f"ran {iters} iterations")
fig = plotting.GenerateScatterTrace(t_prop, [sun_array], label="Sun strength", color="red")
plotting.RenderPlotsToHTML([fig], "sunStrength.html")