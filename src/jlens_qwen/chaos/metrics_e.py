"""E — Layer curve shape: gradient, curvature, cliff, bimodality."""

from __future__ import annotations

from typing import Any


def layer_curve_metrics(series: dict[int, float]) -> dict[str, Any]:
    """Analyze scalar series keyed by layer index.

    Returns gradients, curvatures, cliff (max |g| + argmax layer), bimodality.
    """
    layers = sorted(series.keys())
    if len(layers) < 2:
        return {
            "layers": layers,
            "values": [series[l] for l in layers],
            "gradients": {},
            "curvatures": {},
            "cliff_mag": 0.0,
            "cliff_layer": layers[0] if layers else -1,
            "bimodal": False,
            "peak_layers": layers[:1],
        }
    vals = [series[l] for l in layers]
    grads: dict[int, float] = {}
    for i in range(len(layers) - 1):
        grads[layers[i]] = vals[i + 1] - vals[i]
    curves: dict[int, float] = {}
    for i in range(1, len(layers) - 1):
        curves[layers[i]] = vals[i + 1] - 2 * vals[i] + vals[i - 1]

    # cliff = max abs gradient
    cliff_layer = layers[0]
    cliff_mag = 0.0
    for L, g in grads.items():
        if abs(g) > abs(cliff_mag):
            cliff_mag = g
            cliff_layer = L

    # local maxima for bimodality
    peaks: list[int] = []
    for i in range(1, len(vals) - 1):
        if vals[i] >= vals[i - 1] and vals[i] >= vals[i + 1]:
            peaks.append(layers[i])
    if not peaks:
        # endpoints
        if vals[0] >= vals[1]:
            peaks.append(layers[0])
        if vals[-1] >= vals[-2]:
            peaks.append(layers[-1])
    bimodal = len(peaks) >= 2

    return {
        "layers": layers,
        "values": vals,
        "gradients": {str(k): v for k, v in grads.items()},
        "curvatures": {str(k): v for k, v in curves.items()},
        "cliff_mag": float(cliff_mag),
        "cliff_layer": int(cliff_layer),
        "bimodal": bimodal,
        "peak_layers": peaks,
    }
