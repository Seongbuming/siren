#!/usr/bin/env python3
"""
실제 데이터셋과 합성 데이터 비교 검증 스크립트
Real dataset vs synthetic data validation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from physics_simulator import ScenarioSynthesizer
from data_validation import SyntheticRealComparator, generate_comparison_report
from real_data_loader import check_datasets, create_dummy_real_data, ShipsEarLoader, DeepShipLoader

print("="*80)
print("Real Dataset Validation")
print("Physics-Informed Underwater Acoustic Synthesis")
print("="*80)
print()

# 결과 디렉토리
os.makedirs('results/real_validation', exist_ok=True)

# 데이터셋 확인
print("Step 1: Checking available datasets...")
print("-" * 60)
datasets = check_datasets()

shipsear_available = datasets['shipsear']['available']
deepship_available = datasets['deepship']['available']

if shipsear_available:
    print("✓ ShipsEar dataset found!")
    loader = datasets['shipsear']['loader']
    dataset_name = "ShipsEar"
elif deepship_available:
    print("✓ DeepShip dataset found!")
    loader = datasets['deepship']['loader']
    dataset_name = "DeepShip"
else:
    print("✗ No real dataset found.")
    print()
    print("Creating dummy 'real' data for testing...")
    create_dummy_real_data(output_dir='data/dummy_real', n_samples=5)
    print()
    print("Note: Using synthetic data with variations as 'real' data.")
    print("      For actual validation, download ShipsEar or DeepShip.")
    print("      See: docs/DATA_DOWNLOAD_GUIDE.md")
    print()

    # 더미 데이터 로더 사용
    class DummyLoader:
        def __init__(self):
            self.data_dir = 'data/dummy_real'

        def get_available_classes(self):
            from pathlib import Path
            dummy_dir = Path(self.data_dir)
            if not dummy_dir.exists():
                return []
            return [d.name for d in dummy_dir.iterdir() if d.is_dir()]

        def load_class_samples(self, class_name, n_samples=5, target_sr=16000, duration=3.0):
            from pathlib import Path
            import librosa
            cls_dir = Path(self.data_dir) / class_name
            wav_files = sorted(list(cls_dir.glob("*.wav")))[:n_samples]

            samples = []
            for wav_file in wav_files:
                audio, _ = librosa.load(wav_file, sr=target_sr, duration=duration, mono=True)
                target_length = int(target_sr * duration)
                if len(audio) < target_length:
                    audio = np.pad(audio, (0, target_length - len(audio)))
                else:
                    audio = audio[:target_length]
                samples.append(audio)

            return samples

    loader = DummyLoader()
    dataset_name = "Dummy (synthetic)"

print()

# 시뮬레이터 초기화
SAMPLE_RATE = 16000
DURATION = 3.0

synthesizer = ScenarioSynthesizer(sample_rate=SAMPLE_RATE, duration=DURATION)
comparator = SyntheticRealComparator(sample_rate=SAMPLE_RATE)

print("Step 2: Selecting vessel class for comparison...")
print("-" * 60)

available_classes = loader.get_available_classes()
print(f"Available classes: {available_classes}")

# 비교할 클래스 선택
target_class = None
class_mapping = {
    'Cargo': {'speed': 15.0, 'rpm': 150},
    'Tug': {'speed': 10.0, 'rpm': 200},
    'Tanker': {'speed': 12.0, 'rpm': 120},
    'Passengership': {'speed': 20.0, 'rpm': 250}
}

for cls in ['Cargo', 'Tug', 'Tanker', 'Passengership']:
    if cls in available_classes:
        target_class = cls
        break

if target_class is None:
    target_class = available_classes[0] if available_classes else 'Cargo'
    if target_class not in class_mapping:
        class_mapping[target_class] = {'speed': 15.0, 'rpm': 150}

print(f"Selected class: {target_class}")
print()

# 실제 데이터 로드
print("Step 3: Loading real data samples...")
print("-" * 60)

try:
    real_samples = loader.load_class_samples(
        target_class,
        n_samples=5,
        target_sr=SAMPLE_RATE,
        duration=DURATION
    )
    print(f"✓ Loaded {len(real_samples)} real samples")
except Exception as e:
    print(f"✗ Error loading real data: {e}")
    print("Exiting...")
    sys.exit(1)

print()

# 합성 데이터 생성
print("Step 4: Synthesizing comparable data...")
print("-" * 60)

params = class_mapping[target_class]
print(f"Synthesis parameters: speed={params['speed']} knots, rpm={params['rpm']}")

synthetic_samples = []
for i in range(len(real_samples)):
    # 약간씩 변형
    speed = params['speed'] + np.random.uniform(-2, 2)
    rpm = params['rpm'] + np.random.uniform(-10, 10)

    audio, _ = synthesizer.synthesize_highspeed_vessel(
        speed=speed,
        propeller_rpm=rpm,
        distance=1000.0
    )
    synthetic_samples.append(audio)

print(f"✓ Synthesized {len(synthetic_samples)} samples")
print()

# 비교 분석
print("Step 5: Comparing synthetic vs real...")
print("-" * 60)

comparisons = []
similarities = []

for i, (synth, real) in enumerate(zip(synthetic_samples, real_samples)):
    comparison = comparator.compare_features(
        synth, real, label=f"{target_class} Sample {i+1}"
    )
    comparisons.append(comparison)

    similarity = comparator.spectral_similarity(synth, real)
    similarities.append(similarity)

    print(f"  Sample {i+1}: Spectral Similarity = {similarity:.4f}")

avg_similarity = np.mean(similarities)
print(f"\nAverage Spectral Similarity: {avg_similarity:.4f}")
print()

# 주요 특징 비교
print("Step 6: Feature statistics...")
print("-" * 60)

key_features = ['spectral_centroid', 'spectral_spread', 'low_high_ratio', 'num_tonal_components']

print(f"\n{'Feature':<25} {'Synthetic (avg)':<20} {'Real (avg)':<20} {'Rel. Error'}")
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
print("Step 7: Generating visualizations...")
print("-" * 60)

# 대표 샘플 비교 (첫 번째 샘플)
fig = comparator.visualize_comparison(
    synthetic_samples[0],
    real_samples[0],
    title=f"{target_class}: Synthetic vs Real ({dataset_name})",
    save_path='results/real_validation/comparison_detailed.png'
)
plt.close(fig)
print("  ✓ Detailed comparison: results/real_validation/comparison_detailed.png")

# 여러 샘플 집계 비교
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 평균 파워 스펙트럼
ax = axes[0, 0]
synth_spectra = []
real_spectra = []

for synth, real in zip(synthetic_samples, real_samples):
    synth_fft = np.abs(np.fft.rfft(synth))
    real_fft = np.abs(np.fft.rfft(real))
    synth_spectra.append(synth_fft)
    real_spectra.append(real_fft)

freqs = np.fft.rfftfreq(len(synthetic_samples[0]), 1/SAMPLE_RATE)
synth_avg = 20 * np.log10(np.mean(synth_spectra, axis=0) + 1e-10)
real_avg = 20 * np.log10(np.mean(real_spectra, axis=0) + 1e-10)

ax.plot(freqs, synth_avg, label='Synthetic (avg)', alpha=0.8)
ax.plot(freqs, real_avg, label='Real (avg)', alpha=0.8)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Power (dB)')
ax.set_title('Average Power Spectrum')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 5000])

# 2. 특징 분포
ax = axes[0, 1]
feature_names = ['Spectral\nCentroid', 'Spectral\nSpread', 'Spectral\nRolloff']
features_to_plot = ['spectral_centroid', 'spectral_spread', 'spectral_rolloff']

synth_means = []
real_means = []
for feat in features_to_plot:
    synth_vals = [c['synthetic_features'][feat] for c in comparisons]
    real_vals = [c['real_features'][feat] for c in comparisons]
    synth_means.append(np.mean(synth_vals))
    real_means.append(np.mean(real_vals))

x = np.arange(len(feature_names))
width = 0.35
ax.bar(x - width/2, synth_means, width, label='Synthetic', alpha=0.7)
ax.bar(x + width/2, real_means, width, label='Real', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(feature_names)
ax.set_ylabel('Value (Hz)')
ax.set_title('Feature Comparison')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 3. Similarity scores
ax = axes[1, 0]
ax.bar(range(len(similarities)), similarities, alpha=0.7, color='steelblue')
ax.axhline(y=avg_similarity, color='r', linestyle='--', label=f'Average: {avg_similarity:.3f}')
ax.set_xlabel('Sample Index')
ax.set_ylabel('Spectral Similarity')
ax.set_title('Per-Sample Spectral Similarity')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1])

# 4. Summary text
ax = axes[1, 1]
ax.axis('off')

summary_text = f"""
VALIDATION SUMMARY
{dataset_name} - {target_class}

Samples: {len(real_samples)}
Average Similarity: {avg_similarity:.4f}

Key Findings:
"""

# 유사도 평가
if avg_similarity > 0.7:
    verdict = "✓ GOOD"
    color = 'green'
elif avg_similarity > 0.5:
    verdict = "~ MODERATE"
    color = 'orange'
else:
    verdict = "✗ NEEDS IMPROVEMENT"
    color = 'red'

summary_text += f"\nOverall Match: {verdict}"

# 특징별 오차
for feat in key_features[:3]:
    synth_vals = [c['synthetic_features'][feat] for c in comparisons]
    real_vals = [c['real_features'][feat] for c in comparisons]
    synth_avg = np.mean(synth_vals)
    real_avg = np.mean(real_vals)

    if abs(real_avg) > 1e-10:
        rel_error = abs(synth_avg - real_avg) / abs(real_avg)
    else:
        rel_error = abs(synth_avg - real_avg)

    status = "✓" if rel_error < 0.5 else "~" if rel_error < 1.0 else "✗"
    summary_text += f"\n{status} {feat}: {rel_error:.1%} error"

ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
        verticalalignment='center')

plt.suptitle(f"Real Data Validation: {target_class} ({dataset_name})", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/real_validation/summary.png', dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ Summary: results/real_validation/summary.png")
print()

# 리포트 생성
print("Step 8: Generating report...")
report = generate_comparison_report(comparisons, 'results/real_validation/comparison_report.txt')
print("  ✓ Report: results/real_validation/comparison_report.txt")
print()

# 최종 평가
print("="*80)
print("VALIDATION RESULTS")
print("="*80)
print(f"Dataset: {dataset_name}")
print(f"Class: {target_class}")
print(f"Samples: {len(real_samples)}")
print(f"Average Spectral Similarity: {avg_similarity:.4f}")
print()

if avg_similarity > 0.7:
    print("✓ VALIDATION PASSED")
    print("  Physics model produces realistic acoustic signatures.")
elif avg_similarity > 0.5:
    print("~ VALIDATION MODERATE")
    print("  Physics model captures main characteristics but needs tuning.")
else:
    print("✗ VALIDATION NEEDS IMPROVEMENT")
    print("  Physics parameters require significant adjustment.")

print()
print("Next steps:")
if avg_similarity < 0.7:
    print("  1. Tune physical parameters (RPM, speed, cavitation strength)")
    print("  2. Add more realistic variations (multipath, bio-noise)")
    print("  3. Re-run validation")
else:
    print("  1. Validate on other vessel classes")
    print("  2. Proceed to Neural Audio Codec implementation")
    print("  3. Integrate with Flow Matching")

print()
print("="*80)
