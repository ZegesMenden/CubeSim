import numpy as np
import sys

import loguru
from loguru import logger

import time

import liionpack as lp
import pandas as pd
import pybamm

import common
from .common import timedWrapper as timed

class BatteryParams:
    Ri: float = 0.001
    Rc: float = 0.001
    Rb: float = 0.001
    Rt: float = 0.001

class SatPower:

    def __init__(self, parallelCells: int = 1, seriesCells: int = 1, cellVoltage: float = 3.7, battery: BatteryParams = BatteryParams()) -> None:
        """_summary_

        Args:
            parallelCells (int, optional): _description_. Defaults to 1.
            seriesCells (int, optional): _description_. Defaults to 1.
            cellVoltage (float, optional): _description_. Defaults to 3.7.
            Ri (float, optional): _description_. Defaults to 0.001.
            Rc (float, optional): _description_. Defaults to 0.001.
            Rb (float, optional): _description_. Defaults to 0.001.
            Rt (float, optional): _description_. Defaults to 0.001.
        """

        def typeCheck(val: any, types: tuple[type], name: str) -> any:
            if not isinstance(val, types): raise TypeError(f"{name} must be of type {types}")
            return val

        def castCheck(val: any, t: type, name: str) -> any:
            try: val = t(val)
            except (ValueError, TypeError): raise TypeError(f"{name} must be a {t}-like object")
            return t(val)
        
        self._parallelCells = castCheck(parallelCells, int, "parallelCells")
        self._seriesCells = castCheck(seriesCells, int, "seriesCells")

        self._batteryParams = typeCheck(battery, BatteryParams, "battery")
        self._batteryParams.Ri = castCheck(self._batteryParams.Ri, float, "battery.Ri")
        self._batteryParams.Rc = castCheck(self._batteryParams.Rc, float, "battery.Rc")
        self._batteryParams.Rb = castCheck(self._batteryParams.Rb, float, "battery.Rb")
        self._batteryParams.Rt = castCheck(self._batteryParams.Rt, float, "battery.Rt")
        
        if self._seriesCells < 1:
            raise ValueError("Series cells must be greater than 0")
        
        if self._parallelCells < 1:
            raise ValueError("Parallel cells must be greater than 0")
        
        initialVoltage = castCheck(cellVoltage, float, "cellVoltage")
        
        self._circuit: pd.DataFrame = lp.setup_circuit(Np=self._parallelCells, 
                                                       Ns=self._seriesCells, 
                                                       V=initialVoltage, 
                                                       Ri=self._batteryParams._Ri, 
                                                       Rc=self._batteryParams._Rc, 
                                                       Rb=self._batteryParams._Rb, 
                                                       Rt=self._batteryParams._Rt)

    @property
    def parallelCells(self) -> int:
        return self._parallelCells

    @property
    def seriesCells(self) -> int: return self._seriesCells
    
    @property
    def batteryParams(self) -> BatteryParams:
        return self._batteryParams
