#!/usr/bin/env python3
"""
DeepShip 실제 데이터에 맞춰 물리 파라미터 튜닝
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa

from physics_simulator import ScenarioSynthesizer
from data_validation import SyntheticRealComparator

print("="*80)
print("Physics Parameter Tuning for DeepShip")
print("="*80)
print()

# 실제 데이터 로드
print("Loading real DeepShip sample...")
audio_real, sr = librosa.load('data/DeepShip/Cargo/1.wav', sr=16000, offset=180, duration=3.0)
audio_real = audio_real - np.mean(audio_real)
max_val = np.max(np.abs(audio_real))
if max_val > 1e-6:
    audio_real = audio_real / max_val

print(f"✓ Loaded")
print()

# 다양한 파라미터 조합 테스트
print("Testing parameter combinations...")
print("-" * 60)

synthesizer = ScenarioSynthesizer(sample_rate=16000, duration=3.0)
comparator = SyntheticRealComparator(sample_rate=16000)

# 파라미터 그리드
param_grid = [
    # (speed, rpm, cavitation_strength, distance, description)
    (30, 300, 1.0, 1000, "Original (high-speed)"),
    (15, 150, 0.5, 2000, "Lower speed, less cav"),
    (10, 100, 0.3, 3000, "Low speed, far away"),
    (12, 120, 0.4, 2500, "Medium-low, moderate"),
    (8, 80, 0.2, 4000, "Very low speed, very far"),
]

results = []

for speed, rpm, cav_strength, distance, desc in param_grid:
    # 합성
    audio_synth, _ = synthesizer.synthesize_highspeed_vessel(
        speed=speed,
        propeller_rpm=rpm,
        distance=distance
    )

    # 캐비테이션 강도 조정
    from physics_simulator import UnderwaterAcousticSimulator
    sim = UnderwaterAcousticSimulator(16000, 3.0)

    # 재합성 (캐비테이션 강도 조정)
    machinery = sim.machinery_noise(rpm=rpm, noise_level=0.5)
    cavitation = sim.cavitation_noise(
        speed_knots=speed,
        propeller_rpm=rpm,
        cavitation_strength=cav_strength
    )

    audio_synth = cavitation + machinery * 0.5
    audio_synth = sim.propagation_loss(audio_synth, distance, frequency_hz=500)
    audio_synth += sim.ocean_ambient_noise(sea_state=2)
    audio_synth = audio_synth / (np.max(np.abs(audio_synth)) + 1e-8) * 0.7

    # 비교
    comparison = comparator.compare_features(audio_synth, audio_real, label=desc)
    similarity = comparator.spectral_similarity(audio_synth, audio_real)

    results.append({
        'params': (speed, rpm, cav_strength, distance),
        'desc': desc,
        'similarity': similarity,
        'synth_centroid': comparison['synthetic_features']['spectral_centroid'],
        'real_centroid': comparison['real_features']['spectral_centroid'],
        'synth_tonal': comparison['synthetic_features']['num_tonal_components'],
        'real_tonal': comparison['real_features']['num_tonal_components'],
    })

    print(f"{desc:30s} Similarity: {similarity:.4f}")

print()

# 최적 파라미터 찾기
best_result = max(results, key=lambda x: x['similarity'])

print("="*80)
print("BEST PARAMETERS FOUND")
print("="*80)
print(f"Description: {best_result['desc']}")
print(f"Similarity: {best_result['similarity']:.4f}")
print(f"Parameters:")
speed, rpm, cav, dist = best_result['params']
print(f"  Speed: {speed} knots")
print(f"  RPM: {rpm}")
print(f"  Cavitation Strength: {cav}")
print(f"  Distance: {dist} m")
print()
print(f"Spectral Centroid: {best_result['synth_centroid']:.1f} Hz (target: {best_result['real_centroid']:.1f} Hz)")
print(f"Tonal Components: {best_result['synth_tonal']} (target: {best_result['real_tonal']})")
print()

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Similarity 비교
ax = axes[0, 0]
descriptions = [r['desc'] for r in results]
similarities = [r['similarity'] for r in results]
colors = ['red' if i == results.index(best_result) else 'steelblue' for i in range(len(results))]

ax.barh(range(len(results)), similarities, color=colors, alpha=0.7)
ax.set_yticks(range(len(results)))
ax.set_yticklabels(descriptions, fontsize=9)
ax.set_xlabel('Spectral Similarity')
ax.set_title('Parameter Tuning Results')
ax.grid(True, alpha=0.3, axis='x')
ax.set_xlim([0, 1])

# 2. Spectral Centroid 비교
ax = axes[0, 1]
synth_centroids = [r['synth_centroid'] for r in results]
real_centroid = results[0]['real_centroid']

ax.scatter(range(len(results)), synth_centroids, s=100, alpha=0.7, c=colors)
ax.axhline(y=real_centroid, color='green', linestyle='--', linewidth=2, label='Real Data Target')
ax.set_xlabel('Parameter Set')
ax.set_ylabel('Spectral Centroid (Hz)')
ax.set_title('Spectral Centroid Matching')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Parameter vs Similarity
ax = axes[1, 0]
speeds = [r['params'][0] for r in results]
ax.scatter(speeds, similarities, s=100, alpha=0.7)
for i, r in enumerate(results):
    ax.annotate(f"RPM={r['params'][1]}", (speeds[i], similarities[i]),
                fontsize=8, ha='right')
ax.set_xlabel('Speed (knots)')
ax.set_ylabel('Similarity')
ax.set_title('Speed vs Similarity')
ax.grid(True, alpha=0.3)

# 4. 요약
ax = axes[1, 1]
ax.axis('off')

summary = f"""
TUNING SUMMARY

Total combinations tested: {len(results)}

Best configuration:
  {best_result['desc']}

  Speed: {speed} knots
  RPM: {rpm}
  Cavitation: {cav}
  Distance: {dist} m

Similarity: {best_result['similarity']:.4f}

Improvement:
  From original: {results[0]['similarity']:.4f}
  To best: {best_result['similarity']:.4f}
  Gain: {(best_result['similarity']-results[0]['similarity']):.4f}

Recommendation:
  Update default parameters in
  physics_simulator.py with
  these values for Cargo vessels.
"""

ax.text(0.1, 0.5, summary, fontsize=10, family='monospace', verticalalignment='center')

plt.tight_layout()
plt.savefig('results/deepship_validation/parameter_tuning.png', dpi=150, bbox_inches='tight')
print("✓ Visualization saved: results/deepship_validation/parameter_tuning.png")
print()

print("="*80)
print("Recommended next steps:")
print("  1. Update ScenarioSynthesizer with best parameters")
print("  2. Re-run validation with tuned parameters")
print("  3. Test on other vessel classes (Tug, Tanker, Passengership)")
print("="*80)
