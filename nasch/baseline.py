"""A simple, simulation-free benchmark: a triangular fundamental diagram fitted directly to the data.

Free branch q = vf * k through the origin; congested branch q = a + b * k (b < 0). The breakpoint
between the two is chosen by least squares. The queue-discharge capacity estimate is the apex, the
point where the congested branch meets the free branch. In kinematic-wave terms that is the state
leaving an active bottleneck: on the free branch, at the highest flow the queue can sustain.
Pre-breakdown peaks that sit above the congested line are therefore not counted as capacity.
"""
import numpy as np


def fit_triangle(k, q, n_breaks=50, min_points=8, k_jam=None):
    """Least-squares triangular FD. Returns a dict (vf, a, b, k_apex, capacity) or None.
    With k_jam given, the congested branch is forced through (k_jam, 0), as in the cellular
    automaton, where the jam density is fixed by the 7.5 m cell."""
    k, q = np.asarray(k, float), np.asarray(q, float)
    best = None
    for kc in np.quantile(k, np.linspace(0.10, 0.90, n_breaks)):
        free, cong = k < kc, k >= kc
        if free.sum() < min_points or cong.sum() < min_points:
            continue
        vf = (q[free] * k[free]).sum() / (k[free] ** 2).sum()
        if k_jam is None:
            b, a = np.polyfit(k[cong], q[cong], 1)
        else:
            x = k_jam - k[cong]
            w = (q[cong] * x).sum() / (x ** 2).sum()
            a, b = w * k_jam, -w
        if b >= 0:
            continue
        sse = ((q[free] - vf * k[free]) ** 2).sum() + ((q[cong] - (a + b * k[cong])) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, vf, a, b)
    if best is None:
        return None
    _, vf, a, b = best
    k_apex = a / (vf - b)
    return dict(vf=vf, a=a, b=b, k_apex=k_apex, capacity=vf * k_apex,
                jam_density=-a / b, wave_speed=-b)


def predict_triangle(fit, k):
    k = np.asarray(k, float)
    return np.clip(np.minimum(fit["vf"] * k, fit["a"] + fit["b"] * k), 0.0, None)
