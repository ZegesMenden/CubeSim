from .common import timedWrapper as timed
from .common import initLogging as initLogging

from astropy import units as u

from .orbits import Orbit, OrbitWrapper
from .rigidBody import Vector3, Quaternion, rigidBody3DOF

import numpy as np

class ControlActor:

    def __init__(self):
        pass

    def update(self, dt: u.Quantity, state: rigidBody3DOF, orb: Orbit) -> tuple[u.Quantity, u.Quantity]:
        pass

class ReactionWheel(ControlActor):

    def __init__(self, maxTorque: float|u.Quantity, maxSpeed: float|u.Quantity, inertia: float|u.Quantity, axis: Vector3):
        """Initializes a reaction wheel control actor

        Args:
            maxTorque (float | u.Quantity): _description_
            maxSpeed (float | u.Quantity): _description_
            inertia (float | u.Quantity): _description_
        """

        self._maxTorque = maxTorque.to_value(u.N*u.m) if isinstance(maxTorque, u.Quantity) else maxTorque
        self._maxSpeed = maxSpeed.to_value(u.rad/u.s) if isinstance(maxSpeed, u.Quantity) else maxSpeed
        self._inertia = inertia.to_value(u.kg*u.m**2) if isinstance(inertia, u.Quantity) else inertia
        self._targetTorque = 0
        self._omega = 0
        self._axis = axis
        self._generatedTorque = 0

    def setTargetTorque(self, torque: float|u.Quantity) -> None:
        """Sets the target torque for the reaction wheel

        Args:
            torque (float | u.Quantity): _description_
        """

        self._targetTorque = torque.to_value(u.N*u.m) if isinstance(torque, u.Quantity) else torque

    def getGeneratedTorque(self) -> float:
        return self._generatedTorque
    
    def applyPrecessionTorque(self, precessionTorque: Vector3, dt) -> None:
        self._omega -= abs(precessionTorque * self._axis / self._inertia) * dt

    def update(self, dt: u.Quantity, state: rigidBody3DOF, orb: Orbit) -> tuple[Vector3, u.Quantity]:

        # Torque acting on the wheel (1-DOF)

        torque = self._targetTorque

        if abs(torque) > abs(self._maxTorque):
            torque = abs(self._maxTorque) * np.sign(torque)

        # Calculate the angular acceleration
        alpha = torque / self._inertia

        if abs(self._omega + alpha * dt) > self._maxSpeed:
            alpha = np.sign(alpha) * (self._maxSpeed - self._omega) / dt
            torque = alpha * self._inertia

        # Calculate the new angular velocity
        self._omega = self._omega + alpha * dt

        # Torque acting on the body (3-DOF)

        # Calculate the inertia tensor
        inertiaTensor = self._axis * self._inertia

        # Precession torque is the cross product of rotationaly velocity and wheel rotational momentum
        precessionTorque = state.velocity.cross(inertiaTensor * self._omega)

        # Net torque is the precession torque minus the torque applied by the wheel
        netTorque = precessionTorque - self._axis * torque

        self._generatedTorque = torque

        return self._axis * -torque, precessionTorque, 0 * u.W

class SatelliteControl:

    def __init__(self, sat: rigidBody3DOF, actors: list[ControlActor]):
        
        if not isinstance(sat, rigidBody3DOF):
            raise ValueError("sat must be a rigidBody3DOF object")
        
        if not isinstance(actors, list):
            if isinstance(actors, ControlActor):
                actors = [actors]
            else:
                raise ValueError("actors must be a list of ControlActor objects")
        
        if len(actors) != 0:
            for actor in actors:
                if not isinstance(actor, ControlActor):
                    raise ValueError("actors must be a list of ControlActor objects")
        
        self._actors = actors
        self._sat = sat

    def update(self, dt: u.Quantity, orb: Orbit) -> u.Quantity:

        powerDraw = 0 * u.W

        netPrecessionTorques = Vector3(0, 0, 0)

        for actor in self._actors:
            wheelTorque, precessionTorque, power = actor.update(dt, self._sat, orb)
            netPrecessionTorques += precessionTorque
            self._sat.applyLocalTorque(wheelTorque)
            powerDraw += power

        for actor in self._actors:
            actor.applyPrecessionTorque(netPrecessionTorques, dt)
        
        self._sat.update(dt)

        return powerDraw
