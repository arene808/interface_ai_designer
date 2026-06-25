# -*- coding: utf-8 -*-
"""
精益求径 — 多材料3D打印界面傅里叶统一表示
============================================
将 F0(平面)、N(正弦)、S(锯齿)、D(燕尾) 四种界面类型
统一映射到傅里叶系数空间，为 AI 代理模型提供连续输入向量。
"""

import numpy as np

# ============================================================================
# 全局约束参数
# ============================================================================
SPECIMEN_WIDTH = 13.0       # mm, 标距段宽度 (ASTM D638 Type I)
PERIOD = 4.8                # mm, 所有周期性界面的统一周期
N_HARMONICS = 10
N_SAMPLE_POINTS = 1000
MIN_FEATURE_SIZE = 1.2      # mm, 2
FLAT_END_LENGTH = 1.7       # mm, flat ends at both edges (avoid edge stress concentration)
TRANSITION_LENGTH = 0.6     # mm, cosine-taper transition zone from flat to full curve

FLAT_END_LENGTH = 1.7       # mm, flat ends at both edges (avoid edge stress concentration)
TRANSITION_LENGTH = 0.6     # mm, cosine-taper transition zone from flat to full curve (avoid edge stress concentration)

# ============================================================================
# 界面曲线生成函数
# ============================================================================
def generate_curve(interface_type, amplitude=0, period=PERIOD, 
                   sawtooth_asymmetry=0.5, neck_width=1.5, head_width=2.9,
                   dovetail_depth=1.8, n_points=N_SAMPLE_POINTS, **kwargs):
    """
    Generate interface curve y=f(x) with 1.7mm flat ends.
    Phase-shifted so curve passes through y=0 with positive slope at x=FLAT_END_LENGTH.

    Types:
      flat       : y=0 (baseline)
      sinusoidal : y = A*sin(omega*x + phi), zero-crossing aligned
      sawtooth   : symmetric triangular wave, rising zero aligned
      dovetail   : trapezoidal alternating interlocking, units alternate +/- depth
    """
    x = np.linspace(0, SPECIMEN_WIDTH, n_points)
    omega = 2 * np.pi / period

    if interface_type == "flat":
        y = np.zeros_like(x)

    elif interface_type == "sinusoidal":
        phi = np.pi - omega * FLAT_END_LENGTH
        y = amplitude * np.sin(omega * x + phi)

    elif interface_type == "sawtooth":
        phi_rising = np.pi * sawtooth_asymmetry - omega * FLAT_END_LENGTH
        y = np.zeros_like(x)
        for i in range(len(x)):
            phase = (omega * x[i] + phi_rising) % (2 * np.pi)
            np_val = phase / (2 * np.pi)
            asym = sawtooth_asymmetry
            if np_val < asym:
                y[i] = amplitude * (2 * np_val / asym - 1)
            else:
                y[i] = amplitude * (1 - 2 * (np_val - asym) / (1 - asym))

    elif interface_type == "dovetail":
        # Trapezoidal dovetail with ALTERNATING interlocking direction.
        # Each period p = 4.8mm contains one trapezoidal unit.
        # Adjacent units alternate: +depth (PLA head in TPU) / -depth (TPU head in PLA).
        #
        # Unit structure (width fractions of period p):
        #   [neck/2] [slant_up] [head] [slant_down] [neck/2]
        #    y=0     0->+d     +d     +d->0       y=0
        #
        # Parameters from define_all_groups():
        #   D1: nw=2.0, hw=2.4, d=1.2  (hw/nw=1.2, weak lock)
        #   D2: nw=1.5, hw=2.9, d=1.8  (hw/nw=1.9, medium lock)
        #   D3: nw=1.2, hw=3.2, d=2.4  (hw/nw=2.7, strong lock)
        #
        p = period
        nw = neck_width
        hw = head_width
        d = dovetail_depth
        sw = (p - nw - hw) / 2.0
        if sw < 0.1:
            sw = 0.1

        zero_pos = nw / 2.0  # junction of neck (y=0) and rising slant
        zero_norm = zero_pos / p
        phi = 2.0 * np.pi * zero_norm - omega * FLAT_END_LENGTH

        y = np.zeros_like(x)
        seg1 = (nw / 2.0) / p
        seg2 = (nw / 2.0 + sw) / p
        seg3 = (nw / 2.0 + sw + hw) / p
        seg4 = (nw / 2.0 + sw + hw + sw) / p

        for i in range(len(x)):
            phase = (omega * x[i] + phi) % (2.0 * np.pi)
            norm = phase / (2.0 * np.pi)

            unit_index = int((omega * x[i] + phi) // (2.0 * np.pi))
            sign = 1.0 if (unit_index % 2 == 0) else -1.0

            if norm < seg1:
                y[i] = 0.0
            elif norm < seg2:
                frac = (norm - seg1) / (seg2 - seg1)
                y[i] = sign * d * (1.0 - np.cos(np.pi * frac)) / 2.0
            elif norm < seg3:
                y[i] = sign * d
            elif norm < seg4:
                frac = (norm - seg3) / (seg4 - seg3)
                y[i] = sign * d * (1.0 + np.cos(np.pi * frac)) / 2.0
            else:
                y[i] = 0.0

    else:
        raise ValueError(f"Unknown interface type: {interface_type}")

    # Apply flat ends (y=0) at both edges
    mask_left = x < FLAT_END_LENGTH
    mask_right = x > (SPECIMEN_WIDTH - FLAT_END_LENGTH)
    y = np.where(mask_left | mask_right, 0, y)

    return x, y


def compute_fourier(x, y, n_harmonics=N_HARMONICS):
    N = len(x)
    L = x[-1] - x[0]
    Y = np.fft.fft(y)
    freqs = np.fft.fftfreq(N, d=L/N)
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_Y = Y[pos_mask]
    amplitudes = 2 * np.abs(pos_Y) / N
    phases = np.angle(pos_Y)
    sort_idx = np.argsort(pos_freqs)
    freqs_sorted = pos_freqs[sort_idx]
    amps_sorted = amplitudes[sort_idx]
    phases_sorted = phases[sort_idx]
    total_power = float(np.sum(amps_sorted ** 2))
    cumsum = np.cumsum(amps_sorted[:n_harmonics] ** 2)
    cum_ratio = (cumsum / total_power).tolist() if total_power > 0 else [1.0]*n_harmonics

    # 衰减指数
    ns = np.arange(1, n_harmonics + 1)
    amps_fit = amps_sorted[:n_harmonics]
    valid = amps_fit > 1e-10
    if np.sum(valid) >= 3:
        log_ns = np.log10(ns[valid])
        log_amps = np.log10(amps_fit[valid])
        alpha, _ = np.polyfit(log_ns, log_amps, 1)
        decay = float(-alpha)
    else:
        decay = float('inf')

    return {
        'freqs': freqs_sorted[:n_harmonics].tolist(),
        'amplitudes': amps_sorted[:n_harmonics].tolist(),
        'phases': phases_sorted[:n_harmonics].tolist(),
        'dc_offset': float(np.mean(y)),
        'total_power': total_power,
        'cumulative_power_ratio': cum_ratio,
        'spectral_decay_exponent': decay,
    }


def compute_geometric_features(x, y):
    dx = x[1] - x[0]
    dy = np.gradient(y, dx)
    d2y = np.gradient(dy, dx)
    curvature = np.abs(d2y) / (1 + dy**2) ** 1.5
    return {
        'rms_roughness': float(np.sqrt(np.mean(y**2))),
        'max_amplitude': float(np.max(np.abs(y))),
        'curvature_max': float(np.max(curvature)),
        'curvature_rms': float(np.sqrt(np.mean(curvature**2))),
        'slope_max': float(np.max(np.abs(dy))),
        'slope_rms': float(np.sqrt(np.mean(dy**2))),
        'interface_length': float(np.sum(np.sqrt(1 + dy**2)) * dx),
        'tortuosity': float(np.sum(np.sqrt(1 + dy**2)) * dx / (x[-1] - x[0])),
        'zero_crossings': int(np.sum(np.diff(np.signbit(y.astype(np.float64))))),
    }


def build_unified_vector(amps, geo, n_harmonics=5):
    a = list(amps[:n_harmonics])
    while len(a) < n_harmonics:
        a.append(0.0)
    return a + [geo['rms_roughness'], geo['curvature_max'], geo['tortuosity']]


# ============================================================================
# 定义所有实验组
# ============================================================================
def define_all_groups():
    groups = []

    # 单材基线
    groups.append(('M-A', 'flat', 0, {}))
    groups.append(('M-B', 'flat', 0, {}))

    # F0 平面
    groups.append(('F0', 'flat', 0, {}))

    # N 正弦
    for a, name in [(1.2, 'N1'), (1.8, 'N2'), (2.4, 'N3')]:
        groups.append((name, 'sinusoidal', a, {'period': PERIOD}))

    # S 锯齿 (对称)
    for a, name in [(1.2, 'S1'), (1.8, 'S2'), (2.4, 'S3')]:
        groups.append((name, 'sawtooth', a, {'period': PERIOD, 'sawtooth_asymmetry': 0.5}))

    # D 燕尾 (trapezoidal dovetail, alternating interlocking)
    # p = nw + hw + 2*sw, sw >= 0.1mm
    # Lock ratio = hw/nw: >1 for undercut interlock
    groups.append(('D1', 'dovetail', 1.2, {
        'period': PERIOD, 'neck_width': 2.0, 'head_width': 2.4,
        'dovetail_depth': 1.2}))  # hw/nw=1.2, weak lock
    groups.append(('D2', 'dovetail', 1.8, {
        'period': PERIOD, 'neck_width': 1.5, 'head_width': 2.9,
        'dovetail_depth': 1.8}))  # hw/nw=1.9, medium lock [core group]
    groups.append(('D3', 'dovetail', 2.4, {
        'period': PERIOD, 'neck_width': 1.2, 'head_width': 3.2,
        'dovetail_depth': 2.4}))  # hw/nw=2.7, strong lock (process limit)

    return groups


# ============================================================================
# 主分析
# ============================================================================
def analyze():
    groups = define_all_groups()
    results = []

    for name, itype, amp, extra in groups:
        kwargs = {'amplitude': amp}
        kwargs.update(extra)
        x, y = generate_curve(itype, **kwargs)
        fourier = compute_fourier(x, y)
        geo = compute_geometric_features(x, y)
        vec = build_unified_vector(fourier['amplitudes'], geo)

        results.append({
            'name': name,
            'type': itype,
            'amplitude': amp,
            'extra': extra,
            'x': x, 'y': y,
            'fourier': fourier,
            'geo': geo,
            'unified_vector': vec,
        })

    return results




