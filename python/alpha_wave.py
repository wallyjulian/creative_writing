def alpha_wave(step: int, period: int = 80, min_alpha: float = -0.7, max_alpha: float = 0.8):
    """
    Returns alpha that moves smoothly between min_alpha and max_alpha.
    """
    center = (max_alpha + min_alpha) / 2.0
    amplitude = (max_alpha - min_alpha) / 2.0

    return center + amplitude * math.sin(2.0 * math.pi * step / period)
