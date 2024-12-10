import numpy as np
import sys

import loguru
from loguru import logger

from .common import timedWrapper as timed
from .common import initLogging as initLogging
from .util import castCheck

import os

class SolarPannel:

    def __init__(self, iMpp: float, vMpp: float, area: float, efficiency: float, temperatureCoeff: float = 0, referenceTemp: float = 0) -> None:

        self._iMpp: float = castCheck(iMpp, float, "iMpp")
        self._vMpp: float = castCheck(vMpp, float, "vMpp")
        self._area: float = castCheck(area, float, "area")
        self._efficiency: float = castCheck(efficiency, float, "efficiency")
        self._temperatureCoeff: float = castCheck(temperatureCoeff, float, "temperatureCoeff")
        self._referenceTemp: float = castCheck(referenceTemp, float, "referenceTemp")

    @property
    def area(self) -> float:
        return self._area
    
    @property
    def efficiency(self) -> float:
        return self._efficiency
    
    @property
    def temperatureCoeff(self) -> float:
        return self._temperatureCoeff
    
    @property
    def referenceTemp(self) -> float:
        return self._referenceTemp
    
    def getPower(self, irradiance: float, temperature: float) -> float:
        """_summary_

        Args:
            irradiance (float): _description_
            temperature (float): _description_

        Returns:
            float: _description_
        """
        return irradiance * self._area * self._efficiency * (1 + self._temperatureCoeff * (temperature - self._referenceTemp))

    def maxPower(self) -> float:
        return self._iMpp * self._vMpp
    
    def idealCurrent(self) -> float:
        return self._iMpp
    
    def idealVoltage(self) -> float:
        return self._vMpp

class SolarArray:

    def __init__(self, panel: SolarPannel, parallelPanels: int, seriesPanels: int) -> None:

        self._parallelPanels: int = castCheck(parallelPanels, int, "parallelPanels")
        self._seriesPanels: int = castCheck(seriesPanels, int, "seriesPanels")

        if self._parallelPanels < 1:
            raise ValueError("Parallel panels must be greater than 0")
        
        if self._seriesPanels < 1:
            raise ValueError("Series panels must be greater than 0")

        if not isinstance(panel, SolarPannel):
            raise TypeError("Panel must be of type SolarPannel")
        
        self._panel: SolarPannel = panel

        logger.trace(f"Created solar array with {self.parallelPanels} parallel panels and {self.seriesPanels} series panels")

    @property
    def panel(self) -> SolarPannel:
        return self._panel
    
    @property
    def parallelPanels(self) -> int:
        return self._parallelPanels
    
    @property
    def seriesPanels(self) -> int:
        return self._seriesPanels

    def getPower(self, irradiance: float, temperature: float) -> float:
        return self._panel.getPower(irradiance, temperature) * self._parallelPanels * self._seriesPanels
    
    def getVoltage(self, irradiance: float, temperature: float) -> float:
        return (self.getPower(irradiance, temperature) / self._panel.idealCurrent()) / self._parallelPanels
    
    def getCurrent(self, irradiance: float, temperature: float) -> float:
        return (self.getPower(irradiance, temperature) / self._panel.idealVoltage()) / self._seriesPanels

class VoltageCurve:
    """This class functions as a replacable voltage curve for the battery class. It is a base class and should not be used directly."""
    voltagePoints = []
    timePoints = []       

class LiIonVoltageCurve(VoltageCurve):
    """Voltage curve for a Li-Ion battery"""
    voltagePoints = [1.0, 0.988262268872214, 0.977878132787282, 0.9666972871917748, 0.9582095871242174, 0.9529782077227634, 0.9497908923517056, 0.9476864054247567, 0.946071534672072, 0.9446168447630209, 0.9431602946123937, 0.9416369051784408, 0.940032783085034, 0.9383592200549068, 0.9366382560913789, 0.9348937815858267, 0.933149454786451, 0.9314244471571747, 0.9297352668457711, 0.9280931442112262, 0.9265047684233411, 0.9249713555075938, 0.9234910016905832, 0.9220640790524723, 0.9206939053630714, 0.9193823759836329, 0.9181306124777959, 0.9169381341272365, 0.9158034890119482, 0.9147241174035174, 0.9136972238301226, 0.912719341574877, 0.9117865842477226, 0.9108945063312363, 0.9100386274820211, 0.9092132841971313, 0.9084120545736365, 0.907626904408743, 0.9068463641654294, 0.9060518443233568, 0.9052076201356232, 0.9042430513099757, 0.9030906422869459, 0.901834130338613, 0.9006536265703963, 0.8996558799444346, 0.8988421702257494, 0.8981663152523021, 0.8975842133922245, 0.8970685465945725, 0.8961828790548491, 0.8954336095218384, 0.8946771708273747, 0.8938368092405232, 0.8929083764282371, 0.8924062419576199, 0.8918709288980551, 0.8912910997618837, 0.8906477066718665, 0.8899117911081869, 0.8890416530637971, 0.887977568650438, 0.8866279169255017, 0.8848520299934461, 0.8824625233223399, 0.8791735515131593, 0.8743737307377532, 0.8666363697854997, 0.853159688965404, 0.8296235351634422, 0.7894977183088556, 0.7234463886529435, 0.6217192681973563, 0.5477925926426326]
    timePoints = [0.0, 0.016666666666666666, 0.03333333333333333, 0.05, 0.06666666666666667, 0.08333333333333333, 0.1, 0.11666666666666667, 0.13333333333333333, 0.15, 0.16666666666666666, 0.18333333333333332, 0.2, 0.21666666666666667, 0.23333333333333334, 0.25, 0.26666666666666666, 0.2833333333333333, 0.3, 0.31666666666666665, 0.3333333333333333, 0.35, 0.36666666666666664, 0.38333333333333336, 0.4, 0.4166666666666667, 0.43333333333333335, 0.45, 0.4666666666666667, 0.48333333333333334, 0.5, 0.5166666666666667, 0.5333333333333333, 0.55, 0.5666666666666667, 0.5833333333333334, 0.6, 0.6166666666666667, 0.6333333333333333, 0.65, 0.6666666666666666, 0.6833333333333333, 0.7, 0.7166666666666667, 0.7333333333333333, 0.75, 0.7666666666666667, 0.7833333333333333, 0.8, 0.8166666666666667, 0.85, 0.8833333333333333, 0.9166666666666666, 0.95, 0.9833333333333333, 1.0, 1.0166666666666666, 1.0333333333333334, 1.05, 1.0666666666666667, 1.0833333333333333, 1.1, 1.1166666666666667, 1.1333333333333333, 1.15, 1.1666666666666667, 1.1833333333333333, 1.2, 1.2166666666666666, 1.2333333333333334, 1.25, 1.2666666666666666, 1.2833333333333334, 1.2926292365689414]

class Battery:

    def __init__(self, chargeEfficiency: float = 1, capacity: float = 1, peakVoltage: float = 4.2, initialCharge: float = 1.0, voltageCurve: VoltageCurve = LiIonVoltageCurve()) -> None:
        """Battery class

        Args:
            chargeEfficiency (float, optional): percentage of power supplied to the battery that is retained. Defaults to 1.
            capacity (float, optional): capacity of the battery in Wh. Defaults to 1.
            peakVoltage (float, optional): peak voltage of the battery in V. Defaults to 4.2.
        """

        if not isinstance(voltageCurve, VoltageCurve):
            raise TypeError("Voltage curve must be of type VoltageCurve")

        self._chargeEfficiency: float = castCheck(chargeEfficiency, float, "chargeEfficiency")
        self._capacity = castCheck(capacity, float, "capacity")
        self._peakVoltage = castCheck(peakVoltage, float, "peakVoltage")
        self._charge: float = castCheck(initialCharge, float, "initialCharge")
        if self._charge < 0 or self._charge > 1:
            raise ValueError("Initial charge must be between 0 and 1")
        
        self._voltageCurve: VoltageCurve = voltageCurve

    @property
    def chargeEfficiency(self) -> float:
        return self._chargeEfficiency
    
    @property
    def capacity(self) -> float:
        return self._capacity
    
    @property
    def peakVoltage(self) -> float:
        return self._peakVoltage

    @property
    def voltageCurve(self) -> VoltageCurve:
        return self._voltageCurve
    
    @property
    def charge(self) -> float:
        return self._charge
    
    @property
    def energy(self) -> float:
        return self._charge * self._capacity
    
    def powerTransfer(self, power: float) -> float:
        """Transfer power to/from the battery, positive charges the battery, negative discharges it

        Args:
            power (float): power to transfer in W

        Raises:
            ValueError: if battery is overcharged or undercharged
        """

        p = castCheck(power, float, "power")
        self._charge += p * self._chargeEfficiency / self._capacity

        if self._charge > 1:
            self._charge = 1
            logger.warning(f"Battery overcharged to {round(100*self._charge, 2)}% capacity")
            raise ValueError("Battery overcharged")

        if self._charge < 0:
            self._charge = 0
            logger.warning(f"Battery undercharged to {round(100*self._charge, 2)}% capacity")
            raise ValueError("Battery undercharged")

        return self.energy

    def getVoltage(self) -> float:

        if self.charge == 0:
            return self._voltageCurve.voltagePoints[-1]
        elif self.charge == 1:
            return self._voltageCurve.voltagePoints[0]

        chargeScaled = (1 - self.charge) * (self._voltageCurve.timePoints[-1] - self._voltageCurve.timePoints[0])

        # BST for the correct voltage
        lo = 0
        mid = len(self._voltageCurve.voltagePoints) // 2
        hi = len(self._voltageCurve.voltagePoints) - 1

        while hi - lo > 1:
            if chargeScaled < self._voltageCurve.timePoints[mid]:
                hi = mid
            else:
                lo = mid

            mid = (hi + lo) // 2

        # Perform linear interpolation between the two points
        
        mid -= (mid == len(self._voltageCurve.voltagePoints) - 1)
        
        p0 = self._voltageCurve.voltagePoints[mid]
        p1 = self._voltageCurve.voltagePoints[mid+1]
        t0 = self._voltageCurve.timePoints[mid]
        t1 = self._voltageCurve.timePoints[mid+1]

        return self.peakVoltage * (p0 + (p1 - p0) * (chargeScaled - t0) / (t1 - t0))
        
class BatteryPack:

    def __init__(self, battery: Battery, nS: int = 1, np: int = 1) -> None:

        if not isinstance(battery, Battery):
            raise TypeError("Battery must be of type Battery")

        self._battery: Battery = battery
        self._nS: int = castCheck(nS, int, "nS")
        self._nP: int = castCheck(np, int, "nP")

        logger.trace(f"Created {nS}s {np}p battery with {self.capacity} Wh capacity")

    @property
    def capacity(self) -> float:
        return self._battery.capacity * self._nS * self._nP

    def getVoltage(self) -> float:
        """Get the voltage of the battery pack

        Returns:
            float: voltage of the battery pack
        """
        return self._battery.getVoltage() * self._nS
    
    def getEnergy(self) -> float:
        """Get the total energy stored in the battery

        Returns:
            float: total energy stored in the battery in Wh
        """
        return self._battery.energy * self._nS * self._nP
    
    def transferPower(self, power: float) -> float:
        """transfer power to/from the battery pack

        Args:
            power (float): power to transfer in W, positive charges the batteries, negative discharges it

        Returns:
            float: new energy stored in the battery pack
        """
        return self._battery.powerTransfer(power/(self._nS * self._nP)) * self._nS * self._nP
