import numpy as np
import sys

import loguru
from loguru import logger

from .common import timedWrapper as timed
from .common import initLogging as initLogging

class SolarPannel:

    def __init__(self, iMpp: float, vMpp: float, area: float, efficiency: float, temperatureCoeff: float, referenceTemp: float) -> None:

        def castCheck(val: any, cast: type, name: str) -> any:
            try: return cast(val)
            except (ValueError, TypeError): raise TypeError(f"{name} must be a {cast}-like object")
            except Exception as e: raise e

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

        def castCheck(val: any, cast: type, name: str) -> any:
            try: return cast(val)
            except (ValueError, TypeError): raise TypeError(f"{name} must be a {cast}-like object")
            except Exception as e: raise e

        self._parallelPanels: int = castCheck(parallelPanels, int, "parallelPanels")
        self._seriesPanels: int = castCheck(seriesPanels, int, "seriesPanels")

        if self._parallelPanels < 1:
            raise ValueError("Parallel panels must be greater than 0")
        
        if self._seriesPanels < 1:
            raise ValueError("Series panels must be greater than 0")

        if not isinstance(panel, SolarPannel):
            raise TypeError("Panel must be of type SolarPannel")
        
        self._panel: SolarPannel = panel

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
        return self.getPower(irradiance, temperature) / self._panel.idealCurrent()
    
    def getCurrent(self, irradiance: float, temperature: float) -> float:
        return self.getPower(irradiance, temperature) / self._panel.idealVoltage()
