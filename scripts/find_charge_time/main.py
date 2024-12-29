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

for orbitAlt in [450, 500, 550, 600, 650, 700, 750]:

    # Orbit altitude in km
    # orbitAlt = 450

    solarAreaPercent = 0.6 # Back of the napkin math says 60% of the surface area is solar panels on most commercial models

    uWidth = 3 # 3 U wide
    uHeight = 2 # 2 U tall

    pannelArea = uWidth * uHeight * 0.1 * 0.1 * solarAreaPercent

    # Battery charging efficiency
    chargeEfficiency = 0.96

    # Battery capacity

    battVoltage = 3.7
    battCap = 3.5

    capacity = battVoltage * battCap
    # print(f"Per-cell capacity: {capacity}")

    initialCharge = 0

    cellSeries = 4
    cellParallel = 4

    callbackDt = 30*u.s

    # Orbit setup
    orbit = orbits.OrbitWrapper(orbit=Orbit.circular(Earth, 
                                                    orbitAlt*u.km,
                                                    inc=55*u.deg,
                                                    raan=0*u.deg,
                                                    arglat=0*u.deg),
                                cd=2.2, 
                                areaMassRatio=1.0*u.km**2/u.kg)

    # Period of the orbit, in seconds
    period = orbit.orbit.period

    # Power system setup
    battery = power.Battery(chargeEfficiency=chargeEfficiency,      # Charge efficiency of 96%
                            capacity=capacity,                      # 9.66 Wh per battery
                            peakVoltage=4.2,                        # Peak voltage of 4.2 V when charged
                            initialCharge=initialCharge,            # Start at 80% charge
                            voltageCurve=power.LiIonVoltageCurve())

    batteryPack: power.BatteryPack = power.BatteryPack(battery, cellSeries, cellParallel)

    # solarPanel = power.SolarPannel(1.67, 5.71, 0.06, 0.31)
    # solarPanel = power.SolarPannel(1.67, 5.71, 0.06*2, 0.31)
    solarPanel = power.SolarPannel(1.67, 5.71, pannelArea, 0.3)
    solarArray = power.SolarArray(solarPanel, 1, 1)

    global packVoltage
    global panelVoltage
    global packPower
    global last_t

    global limbAngle
    global distDiff

    limbAngle = []
    distDiff = []

    last_t = orbit.orbit.epoch
    packPower = []
    panelVoltage = []
    packVoltage = []

    global solarWatts
    solarWatts = 0

    @orbit.propCallback
    def powerCallback(orb: Orbit) -> float:
        global packVoltage
        global panelVoltage
        global packPower
        global last_t
        global solarWatts
        
        global limbAngle
        global distDiff
        
        limbVal, distVal = orbits.sunData(orb.rv()[0].to(u.km).value, orb.epoch)

        limbAngle.append(limbVal.value)
        distDiff.append(distVal.value)

        dt = callbackDt.to(u.s).value

        r = orb.rv()[0].to(u.km).value
        sunStrength = (orbits.getSunStrength(r, orb.epoch).value * orbits.isInSun(r, orb.epoch))

        solarVoltage = solarArray.getPower(sunStrength, 0)
        solarPower = solarArray.getPower(sunStrength, 0)

        solarWatts += solarPower*(dt/3600)

        dPower = solarPower

        if dPower * (dt/3600) + batteryPack.getEnergy() >= batteryPack.capacity:
            dPower = 0

        elif dPower * (dt/3600) + batteryPack.getEnergy() <= 0:
            dPower = 0
        
        batteryPack.transferPower(dPower * (dt/3600))
        
        panelVoltage.append(solarVoltage)
        packVoltage.append(batteryPack.getVoltage())
        packPower.append(batteryPack.getEnergy())

        last_t = orb.epoch

    propogationDur = period
    orbit.propagateOrbit(propogationDur, callbackDt=callbackDt)

    print(f"Total solar watts for {orbitAlt} km: {solarWatts}")

    # figBatteryPower = plotting.GenerateScatterTrace(propogationDur, packPower, "Battery Power")
    # figPanelVoltage = plotting.GenerateScatterTrace(propogationDur, panelVoltage, "Panel Power")
    # figPackVoltage = plotting.GenerateScatterTrace(propogationDur, packVoltage, "Pack Voltage")
    # figLimbAngle = plotting.GenerateScatterTrace(propogationDur, limbAngle, "Limb Angle")
    # figDistDiff = plotting.GenerateScatterTrace(propogationDur, distDiff, "Distance Difference")

    # plotting.RenderPlotsToHTML([figBatteryPower, figPanelVoltage, figPackVoltage, figLimbAngle, figDistDiff], filename="batteryPower.html", title="Battery Power and Voltage")

powers = [15.489844770472981, 
          15.748006537143775, 
          16.13524087892946, 
          16.39340234021592, 
          16.780636017612746, 
          17.16786921155252, 
          17.42603019279724]

15.489844770472981, 
15.748006537143775, 
16.13524087892946, 
16.39340234021592, 
16.780636017612746, 
17.16786921155252, 
17.42603019279724