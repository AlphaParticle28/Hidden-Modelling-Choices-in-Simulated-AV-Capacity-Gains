"""Frozen copy of the project's original pure-Python ring simulator (Aug-Sep 2026 version).

Used only by tests/test_simulator.py to check that the numba rewrite in nasch/simulator.py gives
the same results. Do not edit: the functions below are copied verbatim from the original
BasicFunctions/cellular_automata_naschca.py and BasicFunctions/CA_for_FD.py.
"""
import numpy as np


def NaschCA_twoClass(s, V, C, l, n, Vmax, T, thresh=0.5, penetration=0.1,
                     headway_HDV=1, headway_AV=0, thresh_AV=None,
                     thresh0=None, thresh0_AV=None):
    """Two-class (HDV/AV) NaSch on a closed ring.

    This is the only ring implementation in the project. Setting
    penetration=0 with headway_HDV=0 recovers plain single-class NaSch
    exactly, so a separate single-class function would be duplicated
    physics that has to be kept in sync by hand.

    thresh0 / thresh0_AV enable the slow-to-start (VDR) variant: a vehicle
    that is currently stopped dawdles with probability thresh0 instead of
    thresh. Leaving them as None reproduces plain NaSch exactly, so every
    result produced before this option existed still reproduces.
    """
    if thresh_AV is None:
        thresh_AV = thresh
    def randomAllotVehicles(s):
        ctr = 0
        s = [0] * l
        while(ctr < n):
            pos = np.random.randint(0, l)
            if(s[pos] == 0):
                s[pos] = 1
                ctr += 1
        return s

    def randomAllotVelocities(s):
        v = []
        for i in range(len(s)):
            v.append(np.random.randint(0, Vmax + 1) if s[i] == 1 else 0)
        return v

    def randomAllotClasses(s):
        c = []
        for i in range(len(s)):
            if s[i] == 1:
                c.append(1 if np.random.random() < penetration else 0)
            else:
                c.append(-1)
        return c

    if s == []:
        s = randomAllotVehicles(s)
    if V == []:
        V = randomAllotVelocities(s)
    if C == []:
        C = randomAllotClasses(s)

    B, B_vel, B_class = [], [], []
    B.append(s.copy()); B_vel.append(V.copy()); B_class.append(C.copy())

    for t in range(T):
        s_new = [0] * l
        V_new = [0] * l
        C_new = [-1] * l

        cars = [i for i in range(l) if s[i] == 1]
        m = len(cars)

        for k in range(m):
            i = cars[k]
            prev = cars[(k + 1) % m]
            hw = headway_AV if C[i] == 1 else headway_HDV

            if m == 1:
                raw_gap = l - 1
            else:
                raw_gap = (prev - i - 1) % l

            eff_gap = max(raw_gap - hw, 0)

            V_new[i] = min(V[i] + 1, Vmax) #acceleration
            V_new[i] = min(V_new[i], eff_gap) #deceleration
            #class-specific braking prob, with slow-to-start (VDR) applied
            #to standing vehicles when thresh0 is supplied
            if C[i] == 1:
                p_brake = thresh0_AV if (thresh0_AV is not None and V[i] == 0) else thresh_AV
            else:
                p_brake = thresh0 if (thresh0 is not None and V[i] == 0) else thresh
            p = np.random.random() #random braking
            if(p < p_brake):
                V_new[i] = max(V_new[i] - 1, 0)

            target = (i + V_new[i]) % l
            s_new[target] = 1
            V_new[target] = V_new[i]
            C_new[target] = C[i]

        s, V, C = s_new, V_new, C_new
        B.append(s.copy()); B_vel.append(V.copy()); B_class.append(C.copy())
    B = np.array(B, dtype=float)
    B_vel = np.array(B_vel, dtype=float)
    B_class = np.array(B_class, dtype=float)

    return B, B_vel, B_class


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

def measure_steady_state_flow(l, n, Vmax, T, thresh, pen = 0.0, headway_HDV = 1, headway_AV = 0,
                               thresh_AV=None, chunk_size=25, transient_chunks=5,
                               thresh0=None, thresh0_AV=None):
    s = []
    V = []
    C = []
    B_arr, Vel_arr, C_arr = NaschCA_twoClass(s, V, C, l, n, Vmax, T, thresh, pen, headway_HDV, headway_AV,
                                              thresh_AV=thresh_AV,
                                              thresh0=thresh0, thresh0_AV=thresh0_AV)
    chunk_flows = chunk_mean_flow(B_arr, Vel_arr, chunk_size, l)
    steady_chunks = chunk_flows[transient_chunks:]
    return np.mean(steady_chunks)
