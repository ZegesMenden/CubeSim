from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class Vector3:

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        """Initialize a Vector3 object

        Args:
            x (float, optional): x component of the vector. Defaults to 0.0.
            y (float, optional): y component of the vector. Defaults to 0.0.
            z (float, optional): z component of the vector. Defaults to 0.0.
        """
        self.x: float = x
        self.y: float = y
        self.z: float = z

    def __add__(self, other: Vector3) -> Vector3:
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector3) -> Vector3:
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, other: Vector3|float) -> Vector3:
        if isinstance(other, Vector3): return Vector3(self.x * other.x, self.y * other.y, self.z * other.z)
        return Vector3(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other: float) -> Vector3:
        if isinstance(other, Vector3): 
            if other.x == 0 or other.y == 0 or other.z == 0: raise ZeroDivisionError("Division by zero")
            return Vector3(self.x / other.x, self.y / other.y, self.z / other.z)
        if other == 0: raise ZeroDivisionError("Division by zero")
        return Vector3(self.x / other, self.y / other, self.z / other)

    def __eq__(self, other: Vector3) -> bool:
        return self.x == other.x and self.y == other.y and self.z == other.z
    
    def __ne__(self, other: Vector3) -> bool:
        return not self.__eq__(other)
    
    def __iter__(self) -> iter[float]:
        return iter([self.x, self.y, self.z])
    
    def __getitem__(self, index: int) -> float:
        return [self.x, self.y, self.z][index]
    
    def __abs__(self) -> float:
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def len(self) -> float:
        """Calculate the magnitude of a Vector3 object

        Returns:
            float: Magnitude of the Vector3 object
        """
        return abs(self)
    
    def norm(self) -> Vector3:
        """Normalize a Vector3 object

        Returns:
            Vector3: Normalized Vector3 object
        """
        len = abs(self)
        if len == 0: raise ZeroDivisionError("Vector length is zero")
        return self / len
    
    def cross(self, other: Vector3) -> Vector3:
        """Calculate the cross product of two Vector3 objects

        Args:
            other (Vector3): Vector3 object to calculate the cross product with

        Returns:
            Vector3: Cross product of the two Vector3 objects
        """
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def dot(self, other: Vector3) -> float:
        """Calculate the dot product of two Vector3 objects

        Args:
            other (Vector3): Vector3 object to calculate the dot product with

        Returns:
            float: Dot product of the two Vector3 objects
        """
        return self.x * other.x + self.y * other.y + self.z * other.z

    def __repr__(self) -> str:
        return f"Vector3({self.x}, {self.y}, {self.z})"

    def __str__(self) -> str:
        return self.__repr__()

@dataclass 
class Quaternion:

    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        """Initialize a Quaternion object

        Args:
            w (float, optional): Real component of the quaternion. Defaults to 1.0.
            x (float, optional): i component of the quaternion. Defaults to 0.0.
            y (float, optional): j component of the quaternion. Defaults to 0.0.
            z (float, optional): k component of the quaternion. Defaults to 0.0.
        """
        self.w: float = w
        self.x: float = x
        self.y: float = y
        self.z: float = z
    
    def __add__(self, other: Quaternion) -> Quaternion:
        return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: Quaternion) -> Quaternion:
        return Quaternion(self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, other: Quaternion) -> Quaternion:
        return Quaternion(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        )
    
    def __truediv__(self, other: float) -> Quaternion:
        return Quaternion(self.w / other, self.x / other, self.y / other, self.z / other)
    
    def __eq__(self, other: Quaternion) -> bool:
        return self.w == other.w and self.x == other.x and self.y == other.y and self.z == other.z
    
    def __ne__(self, other: Quaternion) -> bool:
        return not self.__eq__(other)
    
    def __iter__(self) -> iter[float]:
        return iter([self.w, self.x, self.y, self.z])
    
    def __getitem__(self, index: int) -> float:
        return [self.w, self.x, self.y, self.z][index]
    
    def conjugate(self) -> Quaternion:
        """Calculate the conjugate of a Quaternion object

        Returns:
            Quaternion: Conjugate of the Quaternion object
        """
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def __abs__(self) -> float:
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def len(self) -> float:
        """Calculate the magnitude of a Quaternion object
        
        Returns:
            float: Magnitude of the Quaternion object
        """
        return abs(self)
    
    def norm(self) -> Quaternion:
        """Normalize a Quaternion object

        Returns:
            Quaternion: Normalized Quaternion object
        """
        len = abs(self)
        return Quaternion(self.w / len, self.x / len, self.y / len, self.z / len)
    
    def rotate(self, v: Vector3) -> Vector3:
        """Rotate a Vector3 object by a Quaternion object

        Args:
            v (Vector3): Vector3 object to rotate

        Returns:
            Vector3: Rotated Vector3 object
        """
        qv = Quaternion(0, v.x, v.y, v.z)
        return (self * qv * self.conjugate()).xyz

    @property
    def xyz(self) -> Vector3:
        """Get the xyz components of a Quaternion object

        Returns:
            Vector3: xyz components of the Quaternion object
        """
        return Vector3(self.x, self.y, self.z)
        
    def dot(self, other: Quaternion) -> float:
        """Calculate the dot product of two Quaternion objects

        Args:
            other (Quaternion): Quaternion object to calculate the dot product with

        Returns:
            float: Dot product of the two Quaternion objects
        """
        return self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z
    
    def fromAxisAngle(axis: Vector3, angle: float) -> Quaternion:
        """Create a Quaternion object from an axis and an angle

        Args:
            axis (Vector3): Axis of rotation
            angle (float): Angle of rotation

        Returns:
            Quaternion: Quaternion object representing the rotation
        """
        halfAngle = angle / 2
        return Quaternion(np.cos(halfAngle), axis.x * np.sin(halfAngle), axis.y * np.sin(halfAngle), axis.z * np.sin(halfAngle))
    
    def toAxisAngle(self) -> tuple[Vector3, float]:
        """Convert a Quaternion object to an axis and an angle

        Returns:
            tuple[Vector3, float]: Axis and angle of rotation
        """
        angle = 2 * np.arccos(self.w)
        axis = self.xyz / np.sin(angle / 2)
        return axis, angle

    def fromEulerAngles(rpy: Vector3|iter) -> Quaternion:
        """Create a Quaternion object from Euler angles

        Args:
            roll (float): Roll angle
            pitch (float): Pitch angle
            yaw (float): Yaw angle

        Returns:
            Quaternion: Quaternion object representing the rotation
        """
        roll = rpy[0]
        pitch = rpy[1]
        yaw = rpy[2]

        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        return Quaternion(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        )
    
    def toEulerAngles(self) -> Vector3:
        """Convert a Quaternion object to Euler angles

        Returns:
            Vector3: Vector3 object containing roll, pitch, and yaw angles
        """
        sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x**2 + self.y**2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (self.w * self.y - self.z * self.x)
        pitch = np.arcsin(sinp)

        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y**2 + self.z**2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return Vector3(roll, pitch, yaw)

    def __repr__(self) -> str:
        return f"Quaternion({self.w}, {self.x}, {self.y}, {self.z})"
    
    def __str__(self) -> str:
        return self.__repr__()
    
class rigidBody3DOF:

    def __init__(self, inertia: Vector3 = Vector3(1.0, 1.0, 1.0), velocity: Vector3 = Vector3(0.0, 0.0, 0.0), rotation: Quaternion = Quaternion(1.0, 0.0, 0.0, 0.0)) -> None:
        """Initialize a rigidBody3DOF object

        Args:
            inertia (Vector3, optional): Inertia tensor of the rigid body. Defaults to Vector3(1.0, 1.0, 1.0).
            velocity (Vector3, optional): Velocity of the rigid body. Defaults to Vector3(0.0, 0.0, 0.0).        
            rotation (Quaternion, optional): rotation of the rigid body. Defaults to Quaternion(1.0, 0.0, 0.0, 0.0).
        """

        if not isinstance(inertia, Vector3):
            raise TypeError("Inertia must be a Vector3 object")
        
        if not isinstance(velocity, Vector3):
            raise TypeError("Velocity must be a Vector3 object")
        
        if not isinstance(rotation, Quaternion):
            raise TypeError("Rotation must be a Quaternion object")

        self.inertia: Vector3 = inertia
        self.velocity: Vector3 = velocity
        self.rotation: Quaternion = rotation

        self._torques = Vector3(0.0, 0.0, 0.0)

    def applyTorque(self, torque: Vector3) -> None:
        """Apply a torque to the rigid body

        Args:
            torque (Vector3): Torque to apply
        """
        self._torques += torque

    def applyLocalTorque(self, torque: Vector3) -> None:
        """Apply a torque to the rigid body

        Args:
            torque (Vector3): Torque to apply
        """
        self.applyTorque(self.rotation.rotate(torque))

    def getTorques(self) -> Vector3:
        """Get the torques applied to the rigid body

        Returns:
            Vector3: Torques applied to the rigid body
        """
        return self._torques

    def update(self, dt: float) -> None:
        """Update the rigid body state based on the applied torques

        Args:
            dt (float): Time step
        """

        # Calculate the angular acceleration
        angularAcceleration = self._torques / self.inertia

        self.velocity += angularAcceleration * dt

        velLen = abs(self.velocity)

        if abs(velLen) > 1e-6:

            # Update the rotation
            self.rotation *= Quaternion.fromAxisAngle(self.velocity / velLen, velLen * dt)
            self.rotation = self.rotation.norm()    

        # Update the torques
        self._torques = Vector3(0.0, 0.0, 0.0)
