import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

from poliastro.twobody import Orbit
from poliastro.bodies import Earth

import astropy.units as u

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import src.common as common
import src.controls as controls
from src.rigidBody import Vector3, Quaternion, rigidBody3DOF
import pyvista as pv
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
from matplotlib.animation import FuncAnimation
import time

def cubeAnim(quaternions, interval=30):
    # Define smaller cube vertices (scaled down)
    scale = 0.5
    cube_vertices = scale * np.array([
        [1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1],
        [1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]
    ])
    
    # Define cube edges (pairs of vertices)
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Front face edges
        [4, 5], [5, 6], [6, 7], [7, 4],  # Back face edges
        [0, 4], [1, 5], [2, 6], [3, 7]   # Connect front and back
    ]
    
    # Define axis lines (x, y, z axes)
    axes_lines = np.array([
        [0, 0, 0], [1.5, 0, 0],  # x-axis
        [0, 0, 0], [0, 1.5, 0],  # y-axis
        [0, 0, 0], [0, 0, 1.5]   # z-axis
    ])

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    
    # Initialize plot objects for cube and axes
    cube_lines = [ax.plot([], [], [], color="b")[0] for _ in edges]
    x_axis, = ax.plot([], [], [], color='r', label='X-axis')
    y_axis, = ax.plot([], [], [], color='g', label='Y-axis')
    z_axis, = ax.plot([], [], [], color='b', label='Z-axis')
    
    # Animation update function
    def update(frame):
        r = R.from_quat(quaternions[frame])
        rotated_vertices = r.apply(cube_vertices)
        rotated_axes = r.apply(axes_lines)
        
        # Update cube lines
        for i, edge in enumerate(edges):
            line = cube_lines[i]
            line.set_data(rotated_vertices[edge, 0], rotated_vertices[edge, 1])
            line.set_3d_properties(rotated_vertices[edge, 2])
        
        # Update axes lines
        x_axis.set_data(rotated_axes[0:2, 0], rotated_axes[0:2, 1])
        x_axis.set_3d_properties(rotated_axes[0:2, 2])
        
        y_axis.set_data(rotated_axes[2:4, 0], rotated_axes[2:4, 1])
        y_axis.set_3d_properties(rotated_axes[2:4, 2])
        
        z_axis.set_data(rotated_axes[4:6, 0], rotated_axes[4:6, 1])
        z_axis.set_3d_properties(rotated_axes[4:6, 2])
        
        return cube_lines + [x_axis, y_axis, z_axis]

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(quaternions), interval=interval, blit=False)
    plt.show()

# Create a control actor

wheelX = controls.ReactionWheel(maxTorque=0.1, maxSpeed=733.04, inertia=0.000432, axis=Vector3(1, 0, 0))
wheelY = controls.ReactionWheel(maxTorque=0.1, maxSpeed=733.04, inertia=0.000432, axis=Vector3(0, 1, 0))
wheelZ = controls.ReactionWheel(maxTorque=0.1, maxSpeed=733.04, inertia=0.000432, axis=Vector3(0, 0, 1))

# Create a state

state = rigidBody3DOF(
    Vector3( # Inertia tensor
        0.05, 
        0.10, 
        0.13), 
    Vector3( # Angular velocity
        45, 
        5, 
        0
        ) * common.DEG2RAD,
    Quaternion.fromEulerAngles(Vector3(
        0, 
        0, 
        0
        ) * common.DEG2RAD)
    )

# Create an orbit

system = controls.SatelliteControl(state, [wheelX, wheelY, wheelZ])

orb = Orbit.circular( Earth, 
                500*u.km,
                inc=0*u.deg,
                raan=0*u.deg,
                arglat=0*u.deg)

# Simulate the system

# Control gain
K = np.array([[1.        , 0.        , 0.        , 3.17804972, 0.        ,
        0.        ],
       [0.        , 1.        , 0.        , 0.        , 3.19374388,
        0.        ],
       [0.        , 0.        , 1.        , 0.        , 0.        ,
        3.20312348]])

dt = 0.001
t = 0
tEnd = 15

quatArr = []
eulArr = []
wheelArr = []
wheelSpeedArr = []
angularMomentumArr = []
print(state.inertia * state.velocity)

targets = [
    Quaternion.fromEulerAngles(Vector3(0, 0, 0) * common.DEG2RAD),
    Quaternion.fromEulerAngles(Vector3(0, 0, 75) * common.DEG2RAD),
    Quaternion.fromEulerAngles(Vector3(0, 15, 90) * common.DEG2RAD)
]

targetIndex = 0

while t < tEnd:
    t += dt

    # Compute the control input

    # get error in degrees
    targetRot = targets[targetIndex]
    err = state.rotation.toEulerAngles() - targetRot.toEulerAngles()
    err = abs(err)

    # if err < 1 * common.DEG2RAD:
    #     if targetIndex == len(targets) - 1:
    #         break
    #     targetIndex += 1

    # targetRot = Quaternion.fromEulerAngles(Vector3(0, 0, 0) * common.DEG2RAD)
    # if t > 15:    
    #     targetRot = Quaternion.fromEulerAngles(Vector3(0, 90, 90) * common.DEG2RAD)
    
    state = system._sat

    eulers = state.rotation.toEulerAngles() - targetRot.toEulerAngles()
    
    omega = state.velocity
    RWOmega = np.array([wheelX._omega, wheelY._omega, wheelZ._omega])

    x = np.array([eulers.x, eulers.y, eulers.z, omega.x, omega.y, omega.z])
    control_input = K @ x

    WheelMomentum = Vector3(wheelX._inertia * wheelX._omega, wheelY._inertia * wheelY._omega, wheelZ._inertia * wheelZ._omega)
    precessionTorque = state.velocity.cross(WheelMomentum)

    wheelX.setTargetTorque(control_input[0] + precessionTorque.x)
    wheelY.setTargetTorque(control_input[1] + precessionTorque.y)
    wheelZ.setTargetTorque(control_input[2] + precessionTorque.z)

    system.update(dt, orb)

    # check if any numnbers are NAN or INF, and break if they are
    if np.isnan(state.rotation.x) or np.isnan(state.rotation.y) or np.isnan(state.rotation.z) or np.isnan(state.rotation.w):
        break

    if np.isnan(state.velocity.x) or np.isnan(state.velocity.y) or np.isnan(state.velocity.z):
        break

    if np.isnan(state.inertia.x) or np.isnan(state.inertia.y) or np.isnan(state.inertia.z):
        break

    # check wheels
    if np.isnan(wheelX._omega) or np.isnan(wheelY._omega) or np.isnan(wheelZ._omega):
        break

    if isinstance(state.rotation, Quaternion):
        quatArr.append(state.rotation)
    else:
        print("Not a quaternion")
    
    rotMomentumSat = state.inertia * state.velocity
    rotMomentumWheels = Vector3(wheelX._omega, wheelY._omega, wheelZ._omega) * wheelX._inertia
    angularMomentumArr.append(state.rotation.conjugate().rotate(rotMomentumSat) + (rotMomentumWheels))

    eulArr.append(eulers * 180 / np.pi)
    wheelArr.append(Vector3(wheelX.getGeneratedTorque(), wheelY.getGeneratedTorque(), wheelZ.getGeneratedTorque()))
    wheelSpeedArr.append(Vector3(wheelX._omega, wheelY._omega, wheelZ._omega) * 9.55)
print(Vector3(wheelX._omega, wheelY._omega, wheelZ._omega) * wheelX._inertia)
print(state.inertia * state.velocity)
# Plot the results
timeArr = np.linspace(0, tEnd, len(quatArr))

fig, ax = plt.subplots(4, 1)

ax[0].plot(timeArr, [a.x for a in wheelArr], label="Torque")
ax[0].plot(timeArr, [a.x for a in eulArr], label="Euler")
ax[0].plot(timeArr, [a.x for a in wheelSpeedArr], label="Speed")
ax[0].legend()
ax[0].set_title("Roll")

ax[1].plot(timeArr, [a.y for a in wheelArr], label="Torque")
ax[1].plot(timeArr, [a.y for a in eulArr], label="Euler")
ax[1].plot(timeArr, [a.y for a in wheelSpeedArr], label="Speed")
ax[1].legend()
ax[1].set_title("Pitch")

ax[2].plot(timeArr, [a.z for a in wheelArr], label="Torque")
ax[2].plot(timeArr, [a.z for a in eulArr], label="Euler")
ax[2].plot(timeArr, [a.z for a in wheelSpeedArr], label="Speed")
ax[2].legend()
ax[2].set_title("Yaw")

ax[3].plot(timeArr, [a.x for a in angularMomentumArr], label="Angular Momentum x")
ax[3].plot(timeArr, [a.y for a in angularMomentumArr], label="Angular Momentum y")
ax[3].plot(timeArr, [a.z for a in angularMomentumArr], label="Angular Momentum z")
ax[3].plot(timeArr, [a.x + a.y + a.z for a in angularMomentumArr], label="Angular Momentum")
ax[3].legend()

plt.show()

animArr = []
interval = int((1/30) / dt)
print(interval)
for i in range(len(quatArr)):
    if i % interval == 0:
        animArr.append([a for a in quatArr[i]])
        if not isinstance(quatArr[i], Quaternion):
            print("Not a quaternion")

cubeAnim(animArr, 30)