#!/usr/bin/env python3
"""
DeepShip 데이터 분석 및 전처리 확인
"""

import sys
import os
sys.path.insert(0, 'src')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import librosa.display

# DeepShip 샘플 로드
print("Loading DeepShip samples...")
audio_files = [
    'data/DeepShip/Cargo/1.wav',
    'data/DeepShip/Cargo/2.wav',
    'data/DeepShip/Cargo/3.wav'
]

fig, axes = plt.subplots(3, 3, figsize=(16, 12))

for idx, audio_file in enumerate(audio_files):
    print(f"\nAnalyzing: {audio_file}")

    # 원본 로드 (리샘플링 없이)
    audio_raw, sr_raw = librosa.load(audio_file, sr=None, duration=3.0)

    # 16kHz로 리샘플링
    audio_16k = librosa.resample(audio_raw, orig_sr=sr_raw, target_sr=16000)

    print(f"  Original SR: {sr_raw} Hz")
    print(f"  Min: {np.min(audio_raw):.6f}, Max: {np.max(audio_raw):.6f}")
    print(f"  Mean: {np.mean(audio_raw):.6f}, Std: {np.std(audio_raw):.6f}")

    # 1. 원본 파형
    ax = axes[idx, 0]
    time = np.arange(len(audio_raw)) / sr_raw
    ax.plot(time, audio_raw, linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Sample {idx+1}: Original Waveform')
    ax.grid(True, alpha=0.3)

    # 2. DC 제거 후
    ax = axes[idx, 1]
    audio_dc_removed = audio_raw - np.mean(audio_raw)
    ax.plot(time, audio_dc_removed, linewidth=0.5, color='orange')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Sample {idx+1}: DC Removed')
    ax.grid(True, alpha=0.3)

    # 3. 스펙트로그램
    ax = axes[idx, 2]
    D = librosa.stft(audio_16k, n_fft=2048, hop_length=512)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(S_db, sr=16000, hop_length=512,
                                   x_axis='time', y_axis='hz', ax=ax,
                                   cmap='viridis')
    ax.set_title(f'Sample {idx+1}: Spectrogram')
    plt.colorbar(img, ax=ax, format='%+2.0f dB')

plt.tight_layout()
plt.savefig('results/real_validation/deepship_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Analysis saved: results/real_validation/deepship_analysis.png")

# 통계 요약
print("\n" + "="*60)
print("DeepShip Data Characteristics")
print("="*60)

all_samples = []
for audio_file in audio_files:
    audio, sr = librosa.load(audio_file, sr=None, duration=3.0)
    all_samples.append(audio)

all_audio = np.concatenate(all_samples)

print(f"Overall statistics (3 samples):")
print(f"  Min: {np.min(all_audio):.6f}")
print(f"  Max: {np.max(all_audio):.6f}")
print(f"  Mean: {np.mean(all_audio):.6f}")
print(f"  Std: {np.std(all_audio):.6f}")
print(f"  RMS: {np.sqrt(np.mean(all_audio**2)):.6f}")

print("\nIssues detected:")
if np.min(all_audio) >= 0:
    print("  ⚠️  All values are non-negative (unusual for audio)")
if abs(np.mean(all_audio)) > 0.01:
    print(f"  ⚠️  Large DC offset: {np.mean(all_audio):.6f}")
if np.max(all_audio) < 0.1:
    print("  ⚠️  Very low amplitude (possibly normalized differently)")

print("\nRecommendations:")
print("  1. Remove DC offset: audio -= mean(audio)")
print("  2. Normalize: audio /= max(abs(audio))")
print("  3. Check if this is envelope/feature data rather than raw audio")
