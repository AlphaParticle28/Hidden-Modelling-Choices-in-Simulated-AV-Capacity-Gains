# CA_for_FD.py
import numpy as np
import matplotlib.pyplot as plt
from cellular_automata_naschca import NaschCA, NaschCA_twoClass

def build_vehicle_positions(l, n):
    s = [0] * l
    positions = np.random.choice(l, size=n, replace=False)
    for pos in positions:
        s[pos] = 1
    return s

def chunk_mean_flow(B_arr, Vel_arr, chunk_size, l):
    T_total = B_arr.shape[0]
    n_chunks = T_total // chunk_size
    chunk_flows = []
    for c in range(n_chunks):
        start = c * chunk_size
        end = start + chunk_size
        B_chunk = B_arr[start:end]
        V_chunk = Vel_arr[start:end]
        Vel_masked = np.where(B_chunk == 1, V_chunk, np.nan)
        per_step_flow = np.nansum(Vel_masked, axis=1) / l  # sum(v) / l = flow
        chunk_flows.append(np.mean(per_step_flow))
    return np.array(chunk_flows)

def diagnose_transient(l, n, Vmax, T, thresh, chunk_size=25):
    s = []
    V = []
    C = []
    # B_arr, Vel_arr = NaschCA(s, V, l, n, Vmax, T, thresh)
    B_arr, Vel_arr, C_arr = NaschCA_twoClass(s, V, [], l, n, Vmax, T, thresh)
    chunk_flows = chunk_mean_flow(B_arr, Vel_arr, chunk_size, l)

    plt.figure(figsize=(8, 5))
    plt.plot(chunk_flows, 'o-', markersize=3)
    plt.xlabel(f'Chunk index ({chunk_size} steps each)')
    plt.ylabel('Mean flow in chunk')
    plt.title(f'Chunk-mean flow over time (density={n/l:.2f})')
    plt.grid(alpha=0.3)
    plt.show()
    return chunk_flows

def measure_steady_state_flow(l, n, Vmax, T, thresh, pen = 0.0, headway_HDV = 1, headway_AV = 0,
                               thresh_AV=None, chunk_size=25, transient_chunks=5):
    s = []
    V = []
    C = []
    B_arr, Vel_arr, C_arr = NaschCA_twoClass(s, V, C, l, n, Vmax, T, thresh, pen, headway_HDV, headway_AV,
                                              thresh_AV=thresh_AV)
    chunk_flows = chunk_mean_flow(B_arr, Vel_arr, chunk_size, l)
    steady_chunks = chunk_flows[transient_chunks:]
    return np.mean(steady_chunks)


# if __name__ == "__main__":
#     l = 200
#     Vmax = 5
#     T = 1000
#     thresh = 0.25
#     chunk_size = 25
#     n_runs_per_density = 3
#     diagnose_transient(l, n=20, Vmax=Vmax, T=T, thresh=thresh, chunk_size=chunk_size)
#     diagnose_transient(l, n=80, Vmax=Vmax, T=T, thresh=thresh, chunk_size=chunk_size)
#     diagnose_transient(l, n=100, Vmax=Vmax, T=T, thresh=thresh, chunk_size=chunk_size)

#     transient_chunks = 8

#     densities = []
#     flows = []
#     for n in range(2, l, 4):
#         density = n / l
#         flow_runs = [
#             measure_steady_state_flow(l, n, Vmax, T, thresh, pen=0.0, headway_HDV=1, headway_AV=0, chunk_size=chunk_size, transient_chunks=transient_chunks)
#             for _ in range(n_runs_per_density)
#         ]
#         densities.append(density)
#         flows.append(np.mean(flow_runs))

#     plt.figure(figsize=(8, 6))
#     plt.plot(densities, flows, 'o-', markersize=3)
#     plt.xlabel('Density (vehicles/cell)')
#     plt.ylabel('Flow (vehicles/timestep)')
#     plt.title(f'NaSch Fundamental Diagram (Vmax={Vmax}, p={thresh})')
#     plt.grid(alpha=0.3)
#     plt.show()