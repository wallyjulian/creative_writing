"""
The "alpha-wave" / rhythmic decoding schedule from Will Kurt's article
(see /docs): sweeps the future-entropy sampler's alpha parameter along a
sine wave over generation steps instead of holding it fixed, so generated
text alternates between more predictable and more entropy-driven stretches.
"""

import math


def alpha_wave(step: int, period: int = 80, min_alpha: float = -0.7, max_alpha: float = 0.8):
    """
    Returns the alpha value for a given generation step, oscillating
    sinusoidally between min_alpha and max_alpha with the given period
    (in steps).
    """
    center = (max_alpha + min_alpha) / 2.0
    amplitude = (max_alpha - min_alpha) / 2.0

    return center + amplitude * math.sin(2.0 * math.pi * step / period)
