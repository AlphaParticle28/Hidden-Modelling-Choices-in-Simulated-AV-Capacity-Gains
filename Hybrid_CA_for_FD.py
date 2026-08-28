# Hybrid_CA_for_FD.py
import numpy as np
import matplotlib.pyplot as plt
from CA_for_FD import measure_steady_state_flow


if __name__ == "__main__":
    l = 200
    Vmax = 5
    T = 1000
    thresh = 0.25
    n = 50
    headway_HDV = 1
    headway_AV = 0
    n_runs = 4

    penetrations = np.linspace(0, 1, 11)   # 0%, 10%, ..., 100%
    flows = []

    for pen in penetrations:
        runs = [measure_steady_state_flow(l, n, Vmax, T, thresh, pen, headway_HDV, headway_AV) for _ in range(n_runs)]
        flows.append(np.mean(runs))
        print(f"penetration={pen:.1f}  flow={np.mean(runs):.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(penetrations, flows, 'o-')
    plt.xlabel('AV Penetration Rate')
    plt.ylabel('Flow (vehicles/timestep)')
    plt.title(f'Flow vs. AV Penetration (density={n/l:.2f}, headway HDV={headway_HDV}/AV={headway_AV})')
    plt.grid(alpha=0.3)
    plt.show()