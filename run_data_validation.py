#!/usr/bin/env python3
"""
실제 데이터 비교 검증 스크립트
(데모: 합성 데이터 + 노이즈를 "실제" 데이터로 사용)
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

print("="*80)
print("Synthetic vs Real Data Validation Demo")
print("="*80)
print()

# 결과 디렉토리
os.makedirs('results/validation', exist_ok=True)

# 초기화
SAMPLE_RATE = 16000
DURATION = 3.0

synthesizer = ScenarioSynthesizer(sample_rate=SAMPLE_RATE, duration=DURATION)
comparator = SyntheticRealComparator(sample_rate=SAMPLE_RATE)

print("Configuration:")
print(f"  Sample Rate: {SAMPLE_RATE} Hz")
print(f"  Duration: {DURATION} s")
print()

# 시나리오 합성
print("Synthesizing scenarios...")
print("-" * 60)

print("[1/2] High-Speed Vessel (synthetic)...")
audio_synthetic, params = synthesizer.synthesize_highspeed_vessel(
    speed=30.0, propeller_rpm=300, distance=1000.0
)

# "실제" 데이터 시뮬레이션
# 실제 데이터를 다운로드하면 이 부분을 실제 데이터 로드로 대체
print("[2/2] Simulating 'real' data (synthetic + realistic variations)...")

# 합성 신호에 현실적인 변형 추가
np.random.seed(42)
audio_real = audio_synthetic.copy()

# 1. 약간 다른 RPM (실제 측정 불확실성)
rpm_variation = synthesizer.sim.cavitation_noise(
    speed_knots=31.0,  # 약간 다른 속도
    propeller_rpm=305,  # 약간 다른 RPM
    n_blades=5,
    cavitation_strength=1.05
)

# 2. 추가 환경 노이즈 (해양 생물, 먼 선박 등)
environmental_noise = synthesizer.sim.ocean_ambient_noise(sea_state=4) * 0.3
bio_noise = np.random.randn(len(audio_real)) * 0.02

# 3. 다른 전파 경로 효과 (다중 경로)
delay_samples = int(0.05 * SAMPLE_RATE)  # 50ms 지연
multipath = np.zeros_like(audio_real)
multipath[delay_samples:] = audio_synthetic[:-delay_samples] * 0.15

# 조합
audio_real = (rpm_variation * 0.7 + audio_synthetic * 0.3 +
              environmental_noise + bio_noise + multipath)

# 정규화
audio_real = audio_real / (np.max(np.abs(audio_real)) + 1e-8) * 0.8

print("✓ Data prepared!")
print()

# 특징 비교
print("Comparing features...")
print("-" * 60)

comparison = comparator.compare_features(audio_synthetic, audio_real, "High-Speed Vessel")

print("Key features comparison:")
key_features = ['spectral_centroid', 'spectral_spread', 'low_high_ratio', 'num_tonal_components']

for feat in key_features:
    if feat in comparison['differences']:
        diff = comparison['differences'][feat]
        print(f"  {feat}:")
        print(f"    Synthetic: {diff['synthetic']:.2f}")
        print(f"    Real:      {diff['real']:.2f}")
        print(f"    Error:     {diff['relative_error']:.2%}")

print()

# 스펙트럼 유사도
similarity = comparator.spectral_similarity(audio_synthetic, audio_real)
print(f"Spectral Similarity (Cosine): {similarity:.4f}")
print()

# 시각화
print("Generating visualizations...")
print("-" * 60)

fig = comparator.visualize_comparison(
    audio_synthetic,
    audio_real,
    title="High-Speed Vessel: Synthetic vs Simulated Real",
    save_path='results/validation/synthetic_vs_real_comparison.png'
)
plt.close(fig)

print("✓ Visualization saved: results/validation/synthetic_vs_real_comparison.png")
print()

# 리포트 생성
print("Generating report...")
report = generate_comparison_report(
    [comparison],
    'results/validation/comparison_result.txt'
)
print("✓ Report saved: results/validation/comparison_result.txt")
print()

# 물리 파라미터 검증
print("=" * 60)
print("Physical Parameter Validation")
print("=" * 60)
print()

print("Expected characteristics for High-Speed Vessel:")
print("  ✓ Strong cavitation: Broadband noise at high frequencies")
print("  ✓ Tonal components: Propeller BRF and harmonics")
print("  ✓ Source Level: ~180 dB re 1μPa @ 1m")
print()

print("Observed in synthetic signal:")
print(f"  ✓ Spectral Centroid: {comparison['synthetic_features']['spectral_centroid']:.1f} Hz")
print(f"  ✓ Tonal Components: {comparison['synthetic_features']['num_tonal_components']}")
print(f"  ✓ Spectral Rolloff: {comparison['synthetic_features']['spectral_rolloff']:.1f} Hz")
print()

print("Validation status:")
if comparison['synthetic_features']['spectral_centroid'] > 500:
    print("  ✅ High spectral centroid (cavitation signature)")
else:
    print("  ⚠️  Low spectral centroid (check cavitation model)")

if comparison['synthetic_features']['num_tonal_components'] > 0:
    print("  ✅ Tonal components detected (propeller signature)")
else:
    print("  ⚠️  No tonal components (check machinery model)")

print()

print("=" * 80)
print("✓ Data validation completed!")
print("=" * 80)
print()
print("Next steps:")
print("  1. Download real dataset (ShipsEar or DeepShip)")
print("  2. Replace simulated data with actual recordings")
print("  3. Run this script again for real validation")
print("  4. Tune physical parameters based on comparison")
print()
print("For dataset download instructions, see: docs/DATASETS.md")
print()
