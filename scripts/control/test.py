import numpy as np
import control
import matplotlib.pyplot as plt

def design_lqr_controller(Ix, Iy, Iz):
    """
    Designs an LQR controller for a CubeSat attitude control system without tracking reaction wheel speeds.

    Parameters:
    - Ix (float): Moment of inertia about the x-axis (roll).
    - Iy (float): Moment of inertia about the y-axis (pitch).
    - Iz (float): Moment of inertia about the z-axis (yaw).

    Returns:
    - K (ndarray): LQR gain matrix.
    """
    # Define the state-space matrices
    A = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])

    B = np.array([
        [0,      0,      0],
        [0,      0,      0],
        [0,      0,      0],
        [1/Ix,   0,      0],
        [0,    1/Iy,     0],
        [0,      0,   1/Iz]
    ])

    # Define the weighting matrices for LQR
    # These can be tuned based on desired performance
    Q = np.diag([1e3, 1e3, 1e3, 1e4, 1e4, 1e4])  # State weighting matrix
    # R = np.diag([1e6, 1e6, 1e6])               # Control weighting matrix
    R = np.diag([1e3, 1e3, 1e3])               # Control weighting matrix

    # Compute the LQR controller gain matrix K
    K, S, E = control.lqr(A, B, Q, R)
    print(f"K = np.{repr(K)}")
    return K

def main():
    # Example moments of inertia (kg·m²)
    Ix = 0.05  # Moment of inertia about x-axis
    Iy = 0.10  # Moment of inertia about y-axis
    Iz = 0.13  # Moment of inertia about z-axis

    # Design the LQR controller
    K = design_lqr_controller(Ix, Iy, Iz)

    print("LQR Gain Matrix K:")
    print(K)

    # Define the state-space system
    A = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])

    B = np.array([
        [0,      0,      0],
        [0,      0,      0],
        [0,      0,      0],
        [1/Ix,   0,      0],
        [0,    1/Iy,     0],
        [0,      0,   1/Iz]
    ])

    # Closed-loop system matrix
    A_cl = A - B @ K

    # Create a state-space system for simulation
    system_cl = control.ss(A_cl, B, np.eye(6), np.zeros((6, 3)))

    # Initial conditions
    initial_state = np.array([0.1, 0.1, 0.1, 0, 0, 0])  # Small initial angles and zero angular velocities

    # Time vector for simulation
    T = np.linspace(0, 30, 3000)  # 20 seconds

    # Input: No external inputs since it's closed-loop with state feedback
    U = np.zeros((3, len(T)))

    # Simulate the closed-loop response
    T, x = control.forced_response(system_cl, T, U, X0=initial_state)

    # Compute control inputs over time: u = -K x
    # Note: x is the state over time, with shape (6, len(T))
    # K has shape (3,6), so u will have shape (3, len(T))
    u = -K @ x

    # Plotting the results
    state_labels = [
        'Roll (phi)', 'Pitch (theta)', 'Yaw (psi)',
        'Roll Rate (p)', 'Pitch Rate (q)', 'Yaw Rate (r)'
    ]

    plt.figure(figsize=(15, 10))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.plot(T, x[i, :], label=state_labels[i])
        plt.title(f'{state_labels[i]} Response')
        plt.xlabel('Time (s)')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot Control Inputs
    control_labels = ['Control Input τ_x (Torque)', 'Control Input τ_y (Torque)', 'Control Input τ_z (Torque)']
    plt.figure(figsize=(12, 6))
    for i in range(3):
        plt.plot(T, u[i, :], label=control_labels[i])
    plt.title('Control Inputs Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (N·m)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
