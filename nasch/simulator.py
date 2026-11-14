"""Two-class Nagel-Schreckenberg cellular automaton with slow-to-start, compiled with numba.

Cells are 7.5 m long and hold at most one vehicle. Every step, all vehicles update in parallel
from the state at the start of the step:

    1. accelerate:   v = min(v + 1, vmax)
    2. keep a gap:   v = min(v, max(gap - h, 0))      h = extra cells kept free (class-specific)
    3. dawdle:       v = max(v - 1, 0) with probability p, or p0 if the vehicle was stopped
    4. move:         x = x + v

Class 0 is a human driver and class 1 an automated vehicle (AV). Each class has its own (p, p0, h);
vmax is shared. p0 = None means no slow-to-start (stopped vehicles dawdle with p like everyone else).

Positions are stored as ever-increasing integers rather than being wrapped around the ring. Vehicles
never overtake, so vehicle i+1 is always the leader of vehicle i, and on a ring the leader of the
last vehicle is vehicle 0 one lap ahead (position + L). This avoids re-sorting after every step.

Random numbers: every simulation seeds its own generator from (seed, number of vehicles), and every
vehicle draws exactly one random number per step whether or not it can dawdle. Two runs that differ
only in driver parameters therefore consume the same random stream ("common random numbers"), so
differences between parameter sets are not drowned out by luck.
"""
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

OPEN_ROAD_GAP = 1 << 40   # gap seen by the first vehicle on an open road


@dataclass(frozen=True)
class Driver:
    p: float              # dawdling probability while moving
    p0: float = None      # dawdling probability when stopped (slow-to-start); None = same as p
    h: int = 0            # extra empty cells kept in front, on top of the no-collision rule


AV = Driver(p=0.0, p0=None, h=0)   # assumed automated vehicle: no dawdling, no hesitation


def _class_arrays(human, av):
    p = np.array([human.p, av.p], dtype=np.float64)
    p0 = np.array([-1.0 if human.p0 is None else human.p0,
                   -1.0 if av.p0 is None else av.p0], dtype=np.float64)
    h = np.array([human.h, av.h], dtype=np.int64)
    return p, p0, h


def _seed(seed, n):
    return (int(seed) * 1_000_003 + int(n)) % 4_294_967_295


# --------------------------------------------------------------------------- compiled core

@njit(cache=True)
def _step(pos, vel, cls, vnew, vmax, p, p0, h, L, ring):
    """Advance all vehicles by one step. Returns the sum of the new speeds."""
    n = pos.shape[0]
    for i in range(n):
        if i < n - 1:
            gap = pos[i + 1] - pos[i] - 1
        elif ring:
            gap = pos[0] + L - pos[i] - 1
        else:
            gap = OPEN_ROAD_GAP
        c = cls[i]
        if not ring and gap <= h[c]:
            # boxed in: the speed is 0 whatever the dice say. Open-road runs skip the draw to
            # save time (same statistics); ring runs keep it so all parameter sets share one
            # random stream.
            vnew[i] = 0
            continue
        v = min(vel[i] + 1, vmax)
        v = min(v, max(gap - h[c], 0))
        if vel[i] == 0 and p0[c] >= 0.0:
            p_brake = p0[c]
        else:
            p_brake = p[c]
        if np.random.random() < p_brake:
            v = max(v - 1, 0)
        vnew[i] = v
    total = 0
    for i in range(n):
        pos[i] += vnew[i]
        vel[i] = vnew[i]
        total += vnew[i]
    return total


@njit(cache=True)
def _assign_classes(n, n_av):
    """Exactly n_av of the n vehicles are AVs, at random places in the stream."""
    cls = np.zeros(n, dtype=np.int64)
    idx = np.arange(n)
    for k in range(n_av):
        j = np.random.randint(k, n)
        idx[k], idx[j] = idx[j], idx[k]
        cls[idx[k]] = 1
    return cls


@njit(cache=True)
def _init_ring(L, n, n_av, vmax):
    """n vehicles on distinct random cells with random speeds (same start as the original code)."""
    cells = np.arange(L)
    for k in range(n):
        j = np.random.randint(k, L)
        cells[k], cells[j] = cells[j], cells[k]
    pos = np.sort(cells[:n])
    vel = np.empty(n, dtype=np.int64)
    for i in range(n):
        vel[i] = np.random.randint(0, vmax + 1)
    cls = _assign_classes(n, n_av)
    return pos, vel, cls


@njit(cache=True)
def _init_queue(n, n_av, h):
    """A standing queue: all vehicles stopped, each at its own class's jam spacing (1 + h cells).
    The queue head sits at cell 0 and the queue extends upstream (negative positions)."""
    cls = _assign_classes(n, n_av)
    pos = np.empty(n, dtype=np.int64)
    pos[n - 1] = 0
    for i in range(n - 2, -1, -1):
        pos[i] = pos[i + 1] - 1 - h[cls[i]]
    vel = np.zeros(n, dtype=np.int64)
    return pos, vel, cls


@njit(parallel=True, cache=True)
def _ring_flow_batch(L, ns, n_avs, task_seeds, vmax, p, p0, h, steps, transient, order):
    # tasks are visited in a shuffled `order` so that every thread gets a mix of cheap
    # (low-density) and expensive (high-density) runs; results do not depend on the order
    n_rows, n_cols = task_seeds.shape
    out = np.empty((n_rows, n_cols))
    for k in prange(order.shape[0]):
        task = order[k]
        i = task // n_cols
        j = task % n_cols
        n = ns[i]
        np.random.seed(task_seeds[i, j])
        pos, vel, cls = _init_ring(L, n, n_avs[i], vmax)
        vnew = np.empty(n, dtype=np.int64)
        moved = 0
        for t in range(steps):
            s = _step(pos, vel, cls, vnew, vmax, p, p0, h, L, True)
            if t >= transient:
                moved += s
        out[i, j] = moved / (L * (steps - transient))
    return out


@njit(parallel=True, cache=True)
def _discharge_batch(n, n_av, seeds, vmax, p, p0, h, steps, detector):
    times = np.full((seeds.shape[0], n), -1, dtype=np.int64)
    for j in prange(seeds.shape[0]):
        np.random.seed(seeds[j])
        pos, vel, cls = _init_queue(n, n_av, h)
        vnew = np.empty(n, dtype=np.int64)
        nxt = n - 1                       # vehicles cross the detector head-first
        for t in range(steps):
            _step(pos, vel, cls, vnew, vmax, p, p0, h, 0, False)
            while nxt >= 0 and pos[nxt] >= detector:
                times[j, nxt] = t + 1
                nxt -= 1
            if nxt < 0:
                break
    return times


@njit(cache=True)
def _trajectory(L, n, n_av, vmax, p, p0, h, steps, seed, ring):
    np.random.seed(seed)
    if ring:
        pos, vel, cls = _init_ring(L, n, n_av, vmax)
    else:
        pos, vel, cls = _init_queue(n, n_av, h)
    vnew = np.empty(n, dtype=np.int64)
    P = np.empty((steps + 1, n), dtype=np.int64)
    V = np.empty((steps + 1, n), dtype=np.int64)
    P[0] = pos
    V[0] = vel
    for t in range(steps):
        _step(pos, vel, cls, vnew, vmax, p, p0, h, L, ring)
        P[t + 1] = pos
        V[t + 1] = vel
    return P, V, cls


# --------------------------------------------------------------------------- python interface

def ring_flow(L, n_vehicles, vmax, human, av=AV, penetration=0.0,
              steps=2000, transient=500, seeds=range(4)):
    """Stationary flow on a closed ring of L cells.

    Returns an array of shape (len(n_vehicles), len(seeds)): the flow in vehicles per step,
    averaged over all cells and over the steps after the first `transient` ones. Flow equals
    density x mean speed, so the fundamental diagram is flow against n / L.
    """
    ns = np.asarray(n_vehicles, dtype=np.int64).ravel()
    if ns.min() < 1 or ns.max() > L:
        raise ValueError("need 1 <= n_vehicles <= L")
    if not 0 <= transient < steps:
        raise ValueError("need 0 <= transient < steps")
    n_avs = np.rint(penetration * ns).astype(np.int64)
    p, p0, h = _class_arrays(human, av)
    task_seeds = np.array([[_seed(s, n) for s in seeds] for n in ns], dtype=np.int64)
    order = np.random.default_rng(0).permutation(task_seeds.size).astype(np.int64)
    return _ring_flow_batch(L, ns, n_avs, task_seeds, vmax, p, p0, h, steps, transient, order)


def queue_discharge_times(n_queue, vmax, human, av=AV, penetration=0.0,
                          detector_offset=50, steps=None, seeds=range(4)):
    """Release a standing queue of n_queue vehicles onto an empty open road.

    Returns an int array of shape (len(seeds), n_queue): the step at which each vehicle first
    reaches a detector placed `detector_offset` cells downstream of the queue head (-1 if it never
    did). Vehicle index n_queue - 1 is the queue head. Sorting a row gives the cumulative count
    curve N(t) at the detector; its slope while the queue is still discharging is the queue
    discharge rate.
    """
    p, p0, h = _class_arrays(human, av)
    n_av = int(round(penetration * n_queue))
    if steps is None:
        steps = 20 * n_queue + 10 * detector_offset
    seeds = np.array([_seed(s, n_queue) for s in seeds], dtype=np.int64)
    return _discharge_batch(n_queue, n_av, seeds, vmax, p, p0, h, steps, detector_offset)


def trajectory(n_vehicles, vmax, human, av=AV, penetration=0.0, steps=500, seed=0,
               L=None, ring=True):
    """Full positions and speeds of every vehicle (for tests and space-time plots).

    ring=True: random start on a ring of L cells. ring=False: a standing queue released onto an
    open road (L is ignored). Returns (positions[steps+1, n], speeds[steps+1, n], classes[n]).
    """
    p, p0, h = _class_arrays(human, av)
    n_av = int(round(penetration * n_vehicles))
    if ring and (L is None or n_vehicles > L):
        raise ValueError("a ring needs L >= n_vehicles")
    return _trajectory(0 if L is None else L, n_vehicles, n_av, vmax, p, p0, h, steps,
                       _seed(seed, n_vehicles), ring)
