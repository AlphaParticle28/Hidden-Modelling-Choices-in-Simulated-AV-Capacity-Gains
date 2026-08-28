# cellular_automata_naschca.py
import numpy as np
import matplotlib.pyplot as plt

def NaschCA(s, V, l, n, Vmax, T, thresh=0.5):
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
            if s[i] == 1:
                v.append(np.random.randint(0, Vmax + 1))
            else:
                v.append(0)
        return v
    if s == []:
        s = randomAllotVehicles(s)
    if V == []:
        V = randomAllotVelocities(s)

    B = []
    B_vel = []
    B.append(s.copy())
    B_vel.append(V.copy())

    for t in range(T):
        s_new = [0] * l
        V_new = [0] * l

        cars = [i for i in range(l) if s[i] == 1]
        m = len(cars)

        for k in range(m):
            i = cars[k]
            prev = cars[(k + 1) % m]

            if m == 1:
                gap = l - 1
            else:
                gap = (prev - i - 1) % l

            #acceleration
            v = min(V[i] + 1, Vmax)
            #deceleration
            v = min(v, gap)
            # random braking
            p = np.random.random()
            if(p < thresh):
                v = max(v - 1, 0)

            new_pos = (i + v) % l
            s_new[new_pos] = 1
            V_new[new_pos] = v

        s, V = s_new, V_new
        B.append(s.copy())
        B_vel.append(V.copy())

    B_arr = np.array(B, dtype=float)
    Vel_arr = np.array(B_vel, dtype=float)
    return B_arr, Vel_arr

def NaschCA_twoClass(s, V, C, l, n, Vmax, T, thresh=0.5, penetration=0.1,
                     headway_HDV=1, headway_AV=0):
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
            p = np.random.random() #random braking
            if(p < thresh):
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

    # plot_traffic(B, B_vel, Vmax)
    return B, B_vel, B_class


def plot_traffic(B_arr, Vel_arr, Vmax):
    Vel_masked = np.where(B_arr == 1, Vel_arr, np.nan)
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color='black')
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(Vel_masked, aspect='auto', cmap=cmap, origin='upper',
                    interpolation='nearest', vmin=0, vmax=Vmax)
    ax.set_xlabel('Cell position (space)')
    ax.set_ylabel('Time step')
    ax.set_title('Space-Time Velocity Diagram')
    plt.colorbar(im, label='Velocity (vehicles only)')
    plt.tight_layout()
    plt.show()

# l = 200
# T = 1000
# Vmax = 5
# n = 50
# thresh = 0.5
# s = []
# V = []
# B_arr, Vel_arr = NashCA(s, V, l, n, Vmax, T, thresh)
# plot_traffic(B_arr, Vel_arr, Vmax)
# B_arr, Vel_arr, C_arr = NashCA_twoClass(s, V, C, l, n, Vmax, T, thresh, penetration=0.10)
# plot_traffic(B_arr, Vel_arr, Vmax)