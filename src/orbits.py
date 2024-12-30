import numpy as np

import astropy.time as astime
from astropy import units as u
import astropy.coordinates as coordinates

from poliastro.bodies import Earth, Sun
from poliastro.twobody import Orbit
from poliastro.twobody.sampling import EpochsArray

import poliastro.constants as constants 
import poliastro.core.perturbations as perturbations
import poliastro.core.propagation as propagation
import poliastro.twobody.propagation as twobodyPropagation

import time

import pyIGRF

import loguru
from loguru import logger

from .common import timedWrapper as timed
from .common import initLogging as initLogging

def getMagneticField(orb: Orbit) -> tuple[u.Quantity, u.Quantity, u.Quantity]:
    """Get the magnetic field at the given position of the satellite

    Args:
        orb (Orbit): orbit of the satellite

    Returns:
        tuple[u.Quantity, u.Quantity, u.Quantity]: x, y, z components of the magnetic field at the given position in nT, x is north, y is east, z is down 
    """

    # Get lat/lon position of the satellite
    pos = coordinates.CartesianRepresentation(orb.r).represent_as(coordinates.WGS84GeodeticRepresentation)

    # Convert lat/lon to degrees, and height to km
    lat = pos.lat.to(u.deg).value
    lon = pos.lon.to(u.deg).value
    height = pos.height.to(u.km).value

    # Get the magnetic field at the given position
    d, i, h, x, y, z, f = pyIGRF.igrf_value(lat, lon, height, orb.epoch.decimalyear)
    return ((x/1e9)*u.T, (y/1e9)*u.T, (z/1e9)*u.T)

def getSunStrength(orb: Orbit) -> u.Quantity:
    """Calculate the strength of the sun at the given position of the satellite

    Args:
        orb (Orbit): orbit of the satellite

    Returns:
        u.Quantity: strength of the sun at the given position in W/m^2
    """

    posEarth = coordinates.get_body_barycentric("earth", orb.epoch)
    posSatEarth = coordinates.CartesianRepresentation(orb.r)

    # Calculate the satellite position relative to the sun
    posSatSun = posSatEarth + posEarth

    dist = posSatSun.norm()

    return ((3.9e26 * u.W) / (4 * np.pi * dist ** 2)).to(u.W / u.m ** 2)

def isInSun(orb: Orbit) -> bool:
    """Check if the satellite is in the sun

    Args:
        orb (Orbit): orbit of the satellite

    Returns:
        bool: True if the satellite is exposed to the sun, False otherwise
    """

    posEarth = coordinates.get_body_barycentric("earth", orb.epoch)
    posSatEarth = coordinates.CartesianRepresentation(orb.r)

    # Calculate the satellite position relative to the sun
    posSatSun = posSatEarth + posEarth

    # ===========================================================================

    # Calculate distances
    distSunEarth = posEarth.norm()
    distSunSat = posSatSun.norm()

    # Calculate the normal vectors
    sunEarthNorm = posEarth / distSunEarth
    sunSatNorm = posSatSun / distSunSat

    angle = np.arccos((sunEarthNorm.xyz * sunSatNorm.xyz).sum(axis=0))

    # find distance from sun to earth in meters
    distSunEarthKm = distSunEarth.to(u.km)

    limbAngle = np.arctan2(Earth.R.to(u.km), distSunEarthKm)

    return (angle > limbAngle) or (distSunSat < distSunEarth)

class OrbitWrapper:
    
    def __init__(self, orbit: Orbit, cd: u.Quantity = None, areaMassRatio: u.Quantity = None) -> None:
        """Initialize a OrbitWrapper object

        Args:
            orbit (Orbit): Class representing the orbit of the body
            cd (Quantity, optional): Drag coefficient of the body, only required when simulating drag. Defaults to None.
            areaMassRatio (Quantity, optional): area to mass ratio (km^2/kg) body, only required when simulating drag. Defaults to None.
        """

        initLogging(level=0, backtrace=True, diagnose=True, logfile=False)

        self._orbit: Orbit = orbit
        self._propagator: callable = twobodyPropagation.CowellPropagator

        self._cd: u.Quantity = cd
        self._amRatio: u.Quantity = areaMassRatio

        # list of callbacks to be called during orbit propagation
        self._propCallbacks: list[callable] = []

        if self._cd is not None:
            try:
                self._cd = u.Quantity(cd)
            except u.UnitConversionError:
                raise TypeError("Drag coefficient must be a Quantity or Quantity-like object")
        
        if self._amRatio is not None:
            try:
                if not isinstance(areaMassRatio, u.Quantity):
                    self._amRatio = u.Quantity(areaMassRatio) * (u.km ** 2 / u.kg)
                else:
                    self._amRatio = areaMassRatio.to(u.km ** 2 / u.kg)
            except u.UnitConversionError:
                raise TypeError("Area to mass ratio must be a Quantity or Quantity-like object")

        logger.info("OrbitWrapper initialized")

    @timed
    def propagateOrbit(self, duration: u.Quantity, useAero: bool = True, callbackDt: u.Quantity = 10 * u.s) -> None:
        """propagate the orbit of the OrbitWrapper by a set time

        Args:
            duration (u.Quantity): duration of time to propagate the orbit
            useAero (bool, optional): incorporate aerodynamic drag into the simulation. Defaults to True.
            callbackDt (u.Quantity, optional): time interval between calling the propagation callbacks. Defaults to 10*u.s.

        Raises:
            ValueError: when drag is to be simulated, but the drag coefficient or area to mass ratio are not set

        """

        # Check if the duration is a Quantity
        if not isinstance(duration, u.Quantity):
            raise TypeError("Duration must be of type Quantity")

        # Convert the duration to seconds
        try:
            duration = duration.to(u.s)
        except u.UnitConversionError:
            raise ValueError("Duration must be a time Quantity")

        # Check if the duration is > 0s
        if duration.to(u.s).value < 0:
            raise ValueError("Duration must be a positive time Quantity")

        # Check if the callback time interval is a Quantity
        if not isinstance(callbackDt, u.Quantity):
            raise TypeError("Callback time interval must be a Quantity")
        
        # Convert the callback time interval to seconds
        try:
            callbackDt = callbackDt.to(u.s)
        except u.UnitConversionError:
            raise ValueError("Callback time interval must be a time Quantity")
        
        # Check if the callback time interval is > 0s
        if callbackDt.to(u.s).value <= 0:
            raise ValueError("Callback time interval must be a non-zero positive time Quantity")

        # Check if the all aero params are present
        if useAero and (self._cd is None or self._amRatio is None):
            raise ValueError("Drag coefficient and area to mass ratio must be set to simulate drag")

        _t0 = time.time_ns()

        try:

            logger.info("Propagating orbit" + (" with aerodynamic drag" if useAero else ""))

            def propFn(t0, state, k):
                
                twoBodyForce = propagation.func_twobody(t0, state, k)
                
                if not useAero:
                    return twoBodyForce
                
                ax, ay, az = perturbations.atmospheric_drag_exponential(
                    t0,
                    state,
                    k,
                    R=Earth.R.to(u.km),
                    C_D=self._cd * u.one,
                    A_over_m=self._amRatio.to_value(u.km ** 2 / u.kg),
                    H0=constants.H0_earth.to(u.km).value,
                    rho0=constants.rho0_earth.to(u.kg / u.km ** 3).value
                )
                aeroForce = np.array([0, 0, 0, ax, ay, az])

                return twoBodyForce + aeroForce
            
            r, v = self._orbit.to_ephem(EpochsArray(self._orbit.epoch + (np.arange(0, duration.value, callbackDt.value) << u.s), method=self._propagator(f=propFn))).rv()
            for i in range(len(r)):
                self._orbit = Orbit.from_vectors(Earth, r[i], v[i], self._orbit.epoch + callbackDt)

                for callback in self._propCallbacks: 
                    callback(self.orbit)

        except Exception as e:
            logger.error(f"Error while propagating orbit: {e}")
            raise e

        _t1 = time.time_ns()
        tCompute = (_t1 - _t0) / 1e9
        tRatio = duration.to(u.s).value / tCompute

        logger.info(f"Propagated orbit in {round(tCompute, 2)} seconds ({round(tRatio, 1)}x real time)")

    def hasSun(self) -> bool:
        """Check if the body is exposed to the sun

        Returns:
            bool: True if the body is exposed to the sun, False otherwise
        """

        posEarth = coordinates.get_body_barycentric("earth", self._orbit.epoch) - coordinates.get_body_barycentric("sun", self._orbit.epoch)
        
        posSatEarth = coordinates.CartesianRepresentation(self._orbit.rv()[0])

        # Calculate the satellite position relative to the sun
        posSatSun = posSatEarth + posEarth

        # ===========================================================================

        # Calculate distances
        distSunEarth = posEarth.norm()
        distSunSat = posSatSun.norm()

        # Calculate the normal vectors
        sunEarthNorm = posEarth / distSunEarth
        sunSatNorm = posSatSun / distSunSat

        angle = np.arccos((sunEarthNorm.xyz * sunSatNorm.xyz).sum(axis=0))

        # find distance from sun to earth in meters
        distSunEarthKm = distSunEarth.to(u.km)

        limbAngle = np.arctan2(6378.137 * u.km, distSunEarthKm)

        return (angle > limbAngle) or (distSunSat < distSunEarth)

    def sunDist(self) -> u.Quantity:
        """Calculate the distance from the body to the sun

        Returns:
            u.Quantity: distance from the body to the sun
        """

        posSun = coordinates.CartesianRepresentation(coordinates.get_body_barycentric("sun", self._orbit.epoch).xyz)
        posEarth = coordinates.CartesianRepresentation(coordinates.get_body_barycentric("earth", self._orbit.epoch).xyz) - posSun
        posSatEarth = coordinates.CartesianRepresentation(self._orbit.rv()[0])

        # Calculate the satellite position relative to the sun
        posSatSun = posSatEarth + posEarth

        return posSatSun.norm()

    def getGroundTrack(self) -> coordinates.WGS84GeodeticRepresentation:
        """_summary_

        Returns:
            coordinates.WGS84GeodeticRepresentation: lat/lon of the body's position on the ground
        """

        # Not 100% sure that this works, but tbh this is black magic to me 
        return coordinates.CartesianRepresentation(self._orbit.rv()[0]).represent_as(coordinates.WGS84GeodeticRepresentation)

    def propCallback(self, callback: callable) -> callable:
        """Add a callback to be called during orbit propagation

        Args:
            callback (callable): function to be called during orbit propagation
        """

        if not callable(callback):
            raise TypeError("Callback must be a callable object")

        self._propCallbacks.append(callback)
        logger.info(f"Callback |{callback.__name__}| added to propagation callback list")

        return callback

    @property
    def orbit(self) -> Orbit:
        return self._orbit
    
    @property
    def cd(self) -> u.Quantity:
        return self._cd
    
    @property
    def amRatio(self) -> u.Quantity:
        return self._amRatio
    
    @orbit.setter
    def orbit(self, value: Orbit) -> None:
        if not isinstance(value, Orbit):
            raise TypeError("Orbit must be of type Orbit")
        logger.info("Orbit updated")
        self._orbit: Orbit = value

    @amRatio.setter
    def amRatio(self, value) -> None:
        try:
            if not isinstance(value, u.Quantity):
                self._amRatio = u.Quantity(value) * (u.km ** 2 / u.kg)
            else:
                self._amRatio = value.to(u.km ** 2 / u.kg)
        except u.UnitConversionError:
            raise TypeError("Area to mass ratio must be a Quantity or Quantity-like object")

    @cd.setter
    def cd(self, value) -> None:
        try:
            value = float(value)
        except (ValueError, TypeError):
            raise TypeError("Drag coefficient must be a float or float-like object")
        self._cd: float = value
