import numpy as np


def smooth_random_noise(size, scale=1.0, seed=None):
    """Generate smooth random variation."""
    rng = np.random.default_rng(seed)

    noise = rng.normal(0, scale, size)

    # Smooth the noise so traffic does not jump unrealistically
    kernel = np.ones(5) / 5
    return np.convolve(noise, kernel, mode="same")


def normal_traffic(time):
    """Normal background traffic."""
    base = 30 + 5 * np.sin(time / 20)
    noise = smooth_random_noise(len(time), scale=2, seed=10)
    return np.maximum(base + noise, 5)


def embb_burst(time):
    """High-bandwidth traffic with gradual increase and bursts."""
    traffic = normal_traffic(time).copy()

    # Gradual increase
    traffic += np.where((time >= 30) & (time < 60), 20, 0)

    # Large burst
    traffic += np.where((time >= 60) & (time < 80), 50, 0)

    # Recovery
    traffic += np.where((time >= 80) & (time < 100), 20, 0)

    return traffic


def urllc_spike(time):
    """Mostly low traffic with short latency-sensitive spikes."""
    traffic = 10 + smooth_random_noise(len(time), scale=1.5, seed=20)

    # Short bursts
    traffic += np.where((time >= 20) & (time < 30), 30, 0)
    traffic += np.where((time >= 45) & (time < 55), 45, 0)
    traffic += np.where((time >= 70) & (time < 85), 35, 0)

    return np.maximum(traffic, 2)


def iot_traffic(time):
    """Periodic IoT/sensor traffic with synchronized bursts."""
    traffic = 5 + 2 * np.sin(time / 5)

    # Sensor reporting bursts
    traffic += np.where((time % 30) < 5, 8, 0)

    noise = smooth_random_noise(len(time), scale=1, seed=30)

    return np.maximum(traffic + noise, 1)


def competition_traffic(time):
    """Multiple slices competing for resources."""
    embb = normal_traffic(time)

    # eMBB becomes dominant
    embb += np.where((time >= 20) & (time < 70), 40, 0)

    # URLLC suddenly becomes dominant
    urllc = 10 + np.where((time >= 50) & (time < 80), 45, 0)

    # IoT remains relatively small but fluctuates
    iot = iot_traffic(time)

    return embb, urllc, iot


def fluctuation_traffic(time):
    """Traffic fluctuating around decision thresholds."""
    base = 40 + 15 * np.sin(time / 4)

    noise = smooth_random_noise(len(time), scale=3, seed=40)

    return np.maximum(base + noise, 5)