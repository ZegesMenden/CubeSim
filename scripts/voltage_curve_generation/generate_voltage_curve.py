import pybamm
import numpy as np
import matplotlib.pyplot as plt

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import src.power as power

outputFile = "vCurve.py"

def modelBattery():

    # Set up the model
    model = pybamm.lithium_ion.DFN()
    
    # Initialize and run the simulation
    sim = pybamm.Simulation(model)

    # Change this experiment to yield a different voltage curve
    experiment = pybamm.Experiment(["Discharge at 1C until 2.2 V"])

    # Change the model and initial state to test different battery types / chemistries
    model = pybamm.lithium_ion.DFN()

    # Initialize and run the simulation
    sim = pybamm.Simulation(model, experiment=experiment)
    sim.solve(initial_soc=1)
    
    # Extract results
    v = sim.solution["Terminal voltage [V]"].entries
    t = sim.solution["Time [h]"].entries
    
    return np.array(t), np.array(v)

def main():

    # Get the voltage curve
    t, v = modelBattery()

    # Output voltage data
    out_t = []
    out_v = []

    # Minimum voltage change to save output, lower values will result in more data points and higher accuracy
    dV = 0.002

    # Only save the voltage data if it changes by more than dV
    for t_h, v in zip(t, v):
        if len(out_v) == 0 or len(out_t) == 0:
            out_t.append(t_h)
            out_v.append(v)
            continue

        dv = v - out_v[-1]
        if abs(dv) > dV:
            out_t.append(t_h)
            out_v.append(v)

    out_v = np.array(out_v)
    out_t = np.array(out_t)

    v0 = out_v[0]
    for i in range(len(out_v)):
        out_v[i] /= v0

    tf = out_t[-1]
    for i in range(len(out_t)):
        out_t[i] /= tf

    try:
        with open(outputFile, "w") as f:
            f.write("out_t = [")
            chars = 0
            for t in out_t:
                f.write(f"{t},")
                chars += len(str(t))+1
                if chars > 100:
                    f.write("\n")
                    chars = 0
            f.write("]\n\n")
            f.write("out_v = [")
            chars = 0
            for v in out_v:
                f.write(f"{v},")
                chars += len(str(v))+1
                if chars > 100:
                    f.write("\n")
                    chars = 0
            f.write("]\n")
    except:
        print("Failed to write output file")

    # Plot the voltage curve

    batt = power.Battery()

    # Generate 1000 points of the voltage curve
    t = np.linspace(0, 1, 1000)
    v = np.zeros(1000)

    # Discharge the battery
    for i in range(1000):
        try: batt.powerTransfer(-0.001)
        except: pass
        v[i] = batt.getVoltage()

    # Plot the voltage curve
    plt.figure()
    plt.plot(t, v)
    plt.plot(out_t, out_v*batt.peakVoltage)
    plt.show()

if __name__ == "__main__":
    main()
