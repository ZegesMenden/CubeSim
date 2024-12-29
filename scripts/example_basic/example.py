import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import src.common as common
import src.orbits as orbits
import src.plotting as plotting
import src.power as power

from poliastro.twobody import Orbit
from poliastro.bodies import Earth

import astropy.units as u
import astropy.time as astime

import numpy as np

# Orbit setup

orbit = orbits.OrbitWrapper(orbit=Orbit.circular(Earth, 
                                                 550*u.km,
                                                 inc=25*u.deg,
                                                 raan=0*u.deg,
                                                 arglat=0*u.deg),
                            cd=2.2, 
                            areaMassRatio=1.0*u.km**2/u.kg)

# Power system setup

battery = power.Battery(chargeEfficiency=0.96,  # Charge efficiency of 96%
                        capacity=9.66,          # 9.66 Wh per battery
                        peakVoltage=4.2,        # Peak voltage of 4.2 V when charged
                        initialCharge=0.8,      # Start at 80% charge
                        voltageCurve=power.LiIonVoltageCurve())

batteryPack: power.BatteryPack = power.BatteryPack(battery, 4, 6)

# solarPanel = power.SolarPannel(1.67, 5.71, 0.06, 0.31)
# solarPanel = power.SolarPannel(1.67, 5.71, 0.06*2, 0.31)
solarPanel = power.SolarPannel(1.67, 5.71, 0.06*8, 0.31)
solarArray = power.SolarArray(solarPanel, 1, 1)

global packVoltage
global panelVoltage
global packPower
global last_t

last_t = orbit.orbit.epoch
packPower = []
panelVoltage = []
packVoltage = []

@orbit.propCallback
def powerCallback(orb: Orbit) -> float:
    global packVoltage
    global panelVoltage
    global packPower
    global last_t
    
    r = orb.rv()[0].to(u.km).value
    sunStrength = (orbits.getSunStrength(r, orb.epoch).value * orbits.isInSun(r, orb.epoch))

    solarVoltage = solarArray.getPower(sunStrength, 0)
    solarPower = solarArray.getPower(sunStrength, 0)

    dPower = solarPower - 20

    dt = (orb.epoch - last_t).to(u.s).value
    print(dt)    
    if dPower * (dt/3600) + batteryPack.getEnergy() >= batteryPack.capacity:
        dPower = 0

    elif dPower * (dt/3600) + batteryPack.getEnergy() <= 0:
        dPower = 0
    
    batteryPack.transferPower(dPower * (dt/3600))
    
    panelVoltage.append(solarVoltage)
    packVoltage.append(batteryPack.getVoltage())
    packPower.append(batteryPack.getEnergy())

    last_t = orb.epoch

propogationDur = 1.5 * u.h
orbit.propagateOrbit(propogationDur)

figBatteryPower = plotting.GenerateScatterTrace(propogationDur, packPower, "Battery Power")
figPanelVoltage = plotting.GenerateScatterTrace(propogationDur, panelVoltage, "Panel Power")
figPackVoltage = plotting.GenerateScatterTrace(propogationDur, packVoltage, "Pack Voltage")

plotting.RenderPlotsToHTML([figBatteryPower, figPanelVoltage, figPackVoltage], filename="batteryPower.html", title="Battery Power and Voltage")
