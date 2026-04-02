import struct
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def read_iq_file(iq_path: Path):
    with open(iq_path, "rb") as f:
        f.seek(157)
        freq_bytes = f.read(4)
        center_freq_mhz = struct.unpack("<f", freq_bytes)[0] / 1e6 if len(freq_bytes) == 4 else 0.0
        f.seek(248)
        raw = f.read()
    if len(raw) % 2 != 0:
        raw = raw[:-1]
    data = np.frombuffer(raw, dtype="<i2")
    iq = data[1::2].astype(np.float32) + 1j * data[0::2].astype(np.float32)
    return center_freq_mhz, iq


def iq_to_spectrogram_png(iq, save_path: Path, fs_hz=153.6e6, nfft=1024):
    _, _, stft = signal.stft(iq, fs=fs_hz, window="hamming", nperseg=nfft, noverlap=nfft // 2, return_onesided=False)
    stft = np.fft.fftshift(stft, axes=0)
    amp_db = 20 * np.log10(np.abs(stft) + 1e-8)
    vmin = np.percentile(amp_db, 5)
    vmax = np.percentile(amp_db, 99.5)
    fig = plt.figure(figsize=(5.12, 5.12), dpi=100, frameon=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(amp_db, aspect="auto", origin="lower", cmap="jet", vmin=vmin, vmax=vmax)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close(fig)


def extract_rgb_frame(video_path: Path, target_ms: float, save_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_POS_MSEC, target_ms)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), frame)
    return True
