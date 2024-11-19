import numpy as np

class Vector3:

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        """Initialize a Vector3 object

        Args:
            x (float, optional): x component of the vector. Defaults to 0.0.
            y (float, optional): y component of the vector. Defaults to 0.0.
            z (float, optional): z component of the vector. Defaults to 0.0.
        """
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other: float) -> "Vector3":
        return Vector3(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other: float) -> "Vector3":
        return Vector3(self.x / other, self.y / other, self.z / other)

    def __eq__(self, other: "Vector3") -> bool:
        return self.x == other.x and self.y == other.y and self.z == other.z
    
    def __ne__(self, other: "Vector3") -> bool:
        return not self.__eq__(other)
    
    def __abs__(self) -> float:
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def norm(self) -> "Vector3":
        """Normalize a Vector3 object

        Returns:
            Vector3: Normalized Vector3 object
        """
        return self / abs(self)
    
    def cross(self, other: "Vector3") -> "Vector3":
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
    
    def dot(self, other: "Vector3") -> float:
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
    
class Quaternion:

    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        """Initialize a Quaternion object

        Args:
            w (float, optional): Real component of the quaternion. Defaults to 1.0.
            x (float, optional): i component of the quaternion. Defaults to 0.0.
            y (float, optional): j component of the quaternion. Defaults to 0.0.
            z (float, optional): k component of the quaternion. Defaults to 0.0.
        """
        self.w = w
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        )
    
    def __mul__(self, other: float) -> "Quaternion":
        return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)
    
    def __truediv__(self, other: float) -> "Quaternion":
        return Quaternion(self.w / other, self.x / other, self.y / other, self.z / other)
    
    def conjugate(self) -> "Quaternion":
        """Calculate the conjugate of a Quaternion object

        Returns:
            Quaternion: Conjugate of the Quaternion object
        """
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def __abs__(self) -> float:
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
    
    def norm(self) -> "Quaternion":
        """Normalize a Quaternion object

        Returns:
            Quaternion: Normalized Quaternion object
        """
        return self / abs(self)
    
    def __mul__(self, other: float) -> "Quaternion":
        return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other: float) -> "Quaternion":
        return Quaternion(self.w / other, self.x / other, self.y / other, self.z / other)
    
    def __eq__(self, other: "Quaternion") -> bool:
        return self.w == other.w and self.x == other.x and self.y == other.y and self.z == other.z
    
    def __ne__(self, other: "Quaternion") -> bool:
        return not self.__eq__(other)
    
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
        
    def dot(self, other: "Quaternion") -> float:
        """Calculate the dot product of two Quaternion objects

        Args:
            other (Quaternion): Quaternion object to calculate the dot product with

        Returns:
            float: Dot product of the two Quaternion objects
        """
        return self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z
    
    def fromAxisAngle(axis: Vector3, angle: float) -> "Quaternion":
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

    def fromEulerAngles(roll: float, pitch: float, yaw: float) -> "Quaternion":
        """Create a Quaternion object from Euler angles

        Args:
            roll (float): Roll angle
            pitch (float): Pitch angle
            yaw (float): Yaw angle

        Returns:
            Quaternion: Quaternion object representing the rotation
        """
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
    
    def fromEulerAngles(v: Vector3) -> "Quaternion":
        """Create a Quaternion object from Euler angles

        Args:
            v (Vector3): Vector3 object containing roll, pitch, and yaw angles

        Returns:
            Quaternion: Quaternion object representing the rotation
        """
        return Quaternion.fromEulerAngles(v.x, v.y, v.z)
    
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

    def __init__(self, inertia: Vector3 = Vector3(1.0, 1.0, 1.0), velocity: Vector3 = Vector3(0.0, 0.0, 0.0), orientation: Quaternion = Quaternion()) -> None:
        """Initialize a rigidBody3DOF object

        Args:
            mass (float, optional): Mass of the rigid body. Defaults to 1.0.
            velocity (Vector3, optional): Velocity of the rigid body. Defaults to Vector3(0.0, 0.0, 0.0).
            inertia (Vector3, optional): Inertia tensor of the rigid body. Defaults to Vector3(1.0, 1.0, 1.0).
            orientation (Quaternion, optional): Orientation of the rigid body. Defaults to Quaternion().
        """
        self.inertia = inertia
        self.velocity = velocity
        self.orientation = orientation

        self._torques = Vector3(0.0, 0.0, 0.0)

    def applyTorque(self, torque: Vector3) -> None:
        """Apply a torque to the rigid body

        Args:
            torque (Vector3): Torque to apply
        """
        self._torques += torque

    def applyForce(self, force: Vector3, position: Vector3) -> None:
        """Apply a force to the rigid body

        Args:
            force (Vector3): Force to apply
        """
        self.applyTorque(position.cross(force))

    def update(self, dt: float) -> None:
        """Update the rigid body state based on the applied torques

        Args:
            dt (float): Time step
        """

        # Calculate the angular acceleration
        angularAcceleration = self._torques / self.inertia

        # Update the orientation
        self.orientation += Quaternion.fromAxisAngle(angularAcceleration, angularAcceleration.abs() * dt / 2)

        # Update the torques
        self._torques = Vector3(0.0, 0.0, 0.0)
