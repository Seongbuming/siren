#!/usr/bin/env python3
"""
DeepShip 데이터 올바른 구간 선택 검증
선박이 지나가는 중간 구간 사용
"""

import sys
import os
sys.path.insert(0, 'src')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import warnings
warnings.filterwarnings('ignore')

from physics_simulator import ScenarioSynthesizer
from data_validation import SyntheticRealComparator, generate_comparison_report

print("="*80)
print("DeepShip Validation (Proper Time Segment)")
print("="*80)
print()

os.makedirs('results/deepship_validation', exist_ok=True)

SAMPLE_RATE = 16000
DURATION = 3.0

synthesizer = ScenarioSynthesizer(sample_rate=SAMPLE_RATE, duration=DURATION)
comparator = SyntheticRealComparator(sample_rate=SAMPLE_RATE)

# DeepShip 파일 - 선박이 지나가는 중간 구간 사용
deepship_files = [
    ('data/DeepShip/Cargo/1.wav', 180),    # 180초 오프셋
    ('data/DeepShip/Cargo/2.wav', 180),
    ('data/DeepShip/Cargo/3.wav', 180),
    ('data/DeepShip/Cargo/4.wav', 180),
    ('data/DeepShip/Cargo/5.wav', 180),
]

print("Loading DeepShip samples (ship passing segments)...")
print("-" * 60)

real_samples = []
for filepath, offset in deepship_files:
    try:
        # 선박이 지나가는 구간 로드
        audio, sr = librosa.load(filepath, sr=SAMPLE_RATE, offset=offset, duration=DURATION)

        # DC 제거 및 정규화
        audio = audio - np.mean(audio)
        max_val = np.max(np.abs(audio))
        if max_val > 1e-6:
            audio = audio / max_val

        real_samples.append(audio)

        # 간단한 분석
        spectrum = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1/SAMPLE_RATE)
        dominant_freq = freqs[np.argmax(spectrum)]

        print(f"  {os.path.basename(filepath)}: DominantFreq={dominant_freq:.1f}Hz")

    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
        continue

print(f"\n✓ Loaded {len(real_samples)} samples")
print()

# 합성 데이터 생성
print("Synthesizing comparable Cargo vessel data...")
print("-" * 60)

# Cargo 선박 파라미터
synthetic_samples = []
for i in range(len(real_samples)):
    speed = 15.0 + np.random.uniform(-2, 2)
    rpm = 150 + np.random.uniform(-10, 10)

    audio, _ = synthesizer.synthesize_highspeed_vessel(
        speed=speed,
        propeller_rpm=rpm,
        distance=1000.0
    )
    synthetic_samples.append(audio)

print(f"✓ Synthesized {len(synthetic_samples)} samples")
print()

# 비교 분석
print("Comparing synthetic vs real...")
print("-" * 60)

comparisons = []
similarities = []

for i, (synth, real) in enumerate(zip(synthetic_samples, real_samples)):
    comparison = comparator.compare_features(
        synth, real, label=f"Cargo Sample {i+1}"
    )
    comparisons.append(comparison)

    similarity = comparator.spectral_similarity(synth, real)
    similarities.append(similarity)

    print(f"  Sample {i+1}: Spectral Similarity = {similarity:.4f}")

avg_similarity = np.mean(similarities)
print(f"\nAverage Spectral Similarity: {avg_similarity:.4f}")
print()

# 주요 특징 비교
print("Feature statistics...")
print("-" * 60)

key_features = ['spectral_centroid', 'spectral_spread', 'spectral_rolloff', 'num_tonal_components']

print(f"\n{'Feature':<25} {'Synthetic':<20} {'Real':<20} {'Error'}")
print("-" * 80)

for feat in key_features:
    synth_vals = [c['synthetic_features'][feat] for c in comparisons]
    real_vals = [c['real_features'][feat] for c in comparisons]

    synth_avg = np.mean(synth_vals)
    real_avg = np.mean(real_vals)

    if abs(real_avg) > 1e-10:
        rel_error = abs(synth_avg - real_avg) / abs(real_avg)
    else:
        rel_error = abs(synth_avg - real_avg)

    print(f"{feat:<25} {synth_avg:<20.2f} {real_avg:<20.2f} {rel_error:.2%}")

print()

# 시각화
print("Generating visualizations...")
print("-" * 60)

# 상세 비교
fig = comparator.visualize_comparison(
    synthetic_samples[0],
    real_samples[0],
    title="Cargo Vessel: Synthetic vs DeepShip (Ship Passing)",
    save_path='results/deepship_validation/comparison_detailed.png'
)
plt.close(fig)
print("  ✓ Detailed: results/deepship_validation/comparison_detailed.png")

# 요약
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 평균 파워 스펙트럼
ax = axes[0, 0]
freqs = np.fft.rfftfreq(len(synthetic_samples[0]), 1/SAMPLE_RATE)

synth_spectra = [np.abs(np.fft.rfft(s)) for s in synthetic_samples]
real_spectra = [np.abs(np.fft.rfft(r)) for r in real_samples]

synth_avg = 20 * np.log10(np.mean(synth_spectra, axis=0) + 1e-10)
real_avg = 20 * np.log10(np.mean(real_spectra, axis=0) + 1e-10)

ax.plot(freqs, synth_avg, label='Synthetic', alpha=0.8)
ax.plot(freqs, real_avg, label='Real (DeepShip)', alpha=0.8)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Power (dB)')
ax.set_title('Average Power Spectrum')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 3000])

# 2. LOFAR 영역 비교
ax = axes[0, 1]
mask = freqs <= 1000
ax.plot(freqs[mask], synth_avg[mask], label='Synthetic', alpha=0.8, linewidth=2)
ax.plot(freqs[mask], real_avg[mask], label='Real (DeepShip)', alpha=0.8, linewidth=2)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Power (dB)')
ax.set_title('LOFAR Region (0-1000 Hz)')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Similarity 분포
ax = axes[1, 0]
ax.bar(range(len(similarities)), similarities, alpha=0.7, color='steelblue')
ax.axhline(y=avg_similarity, color='r', linestyle='--', label=f'Avg: {avg_similarity:.3f}')
ax.set_xlabel('Sample Index')
ax.set_ylabel('Spectral Similarity')
ax.set_title('Per-Sample Similarity')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1])

# 4. 요약
ax = axes[1, 1]
ax.axis('off')

if avg_similarity > 0.6:
    verdict = "✓ GOOD MATCH"
    color = 'green'
elif avg_similarity > 0.4:
    verdict = "~ MODERATE"
    color = 'orange'
else:
    verdict = "✗ NEEDS WORK"
    color = 'red'

summary = f"""
VALIDATION RESULTS
Dataset: DeepShip
Class: Cargo
Segment: Ship Passing (offset 180s)

Samples: {len(real_samples)}
Avg Similarity: {avg_similarity:.4f}

Overall: {verdict}

Key Features:
"""

for feat in ['spectral_centroid', 'spectral_spread'][:2]:
    synth_vals = [c['synthetic_features'][feat] for c in comparisons]
    real_vals = [c['real_features'][feat] for c in comparisons]
    synth_avg = np.mean(synth_vals)
    real_avg = np.mean(real_vals)

    if abs(real_avg) > 1e-10:
        rel_error = abs(synth_avg - real_avg) / abs(real_avg)
    else:
        rel_error = abs(synth_avg - real_avg)

    status = "✓" if rel_error < 0.5 else "~" if rel_error < 1.0 else "✗"
    summary += f"\n{status} {feat}: {rel_error:.0%}"

ax.text(0.1, 0.5, summary, fontsize=11, family='monospace', verticalalignment='center')

plt.suptitle("DeepShip Validation: Cargo Vessel", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/deepship_validation/summary.png', dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ Summary: results/deepship_validation/summary.png")
print()

# 리포트
generate_comparison_report(comparisons, 'results/deepship_validation/report.txt')
print("  ✓ Report: results/deepship_validation/report.txt")
print()

# 최종 평가
print("="*80)
print("FINAL VALIDATION RESULTS")
print("="*80)
print(f"Dataset: DeepShip (proper time segments)")
print(f"Class: Cargo")
print(f"Average Spectral Similarity: {avg_similarity:.4f}")
print()

if avg_similarity > 0.6:
    print("✓ VALIDATION PASSED")
    print("  Physics model captures realistic vessel acoustics.")
    print("\nNext steps:")
    print("  1. Validate other vessel classes (Tug, Tanker, Passengership)")
    print("  2. Proceed to Neural Audio Codec implementation")
elif avg_similarity > 0.4:
    print("~ VALIDATION MODERATE")
    print("  Model captures general characteristics but needs refinement.")
    print("\nNext steps:")
    print("  1. Fine-tune physical parameters")
    print("  2. Add more environmental effects")
    print("  3. Re-validate")
else:
    print("✗ VALIDATION NEEDS IMPROVEMENT")
    print("  Significant gaps between synthetic and real data.")
    print("\nNext steps:")
    print("  1. Analyze spectral differences")
    print("  2. Adjust physics models")
    print("  3. Consider hybrid approach")

print()
print("="*80)
