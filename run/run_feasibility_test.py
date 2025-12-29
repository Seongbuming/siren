#!/usr/bin/env python3
"""
실현 가능성 테스트 실행 스크립트
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 실행
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from physics_simulator import ScenarioSynthesizer
from visualization import AcousticVisualizer, create_summary_report

# 결과 디렉토리 생성
os.makedirs('results', exist_ok=True)
os.makedirs('results/audio', exist_ok=True)

print("="*80)
print("PILFM-UAS Feasibility Test")
print("Physics-Informed Latent Flow Matching for Underwater Acoustic Anomaly Synthesis")
print("="*80)
print()

# 초기화
SAMPLE_RATE = 16000
DURATION = 3.0

synthesizer = ScenarioSynthesizer(sample_rate=SAMPLE_RATE, duration=DURATION)
visualizer = AcousticVisualizer(sample_rate=SAMPLE_RATE)

print(f"Configuration:")
print(f"  Sample Rate: {SAMPLE_RATE} Hz")
print(f"  Duration: {DURATION} s")
print(f"  Total Samples: {int(SAMPLE_RATE * DURATION)}")
print()

# 시나리오 합성
print("Synthesizing scenarios...")
print("-" * 60)

print("[1/4] Covert Submarine...")
audio_sub, params_sub = synthesizer.synthesize_covert_submarine(
    speed=5.0, depth=150.0, distance=2000.0
)

print("[2/4] High-Speed Vessel...")
audio_highspeed, params_highspeed = synthesizer.synthesize_highspeed_vessel(
    speed=30.0, propeller_rpm=300, distance=1000.0
)

print("[3/4] Collision Event...")
audio_collision, params_collision = synthesizer.synthesize_collision(
    impact_energy='medium', resonance_freq=80
)

print("[4/4] Rapid Acceleration...")
audio_accel, params_accel = synthesizer.synthesize_rapid_acceleration(
    v_initial=5.0, v_final=25.0
)

print("✓ All scenarios synthesized successfully!")
print()

# 시나리오 리스트
scenarios = [
    (audio_sub, params_sub, "Covert Submarine"),
    (audio_highspeed, params_highspeed, "High-Speed Vessel"),
    (audio_collision, params_collision, "Collision Event"),
    (audio_accel, params_accel, "Rapid Acceleration")
]

# 시각화
print("Generating visualizations...")
print("-" * 60)

print("  [1/4] Scenario comparison grid...")
fig1 = visualizer.plot_scenario_comparison_grid(
    scenarios, save_path='results/scenario_comparison.png'
)
plt.close(fig1)

print("  [2/4] LOFAR spectrum analysis...")
fig2, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()
for idx, (audio, params, name) in enumerate(scenarios):
    visualizer.plot_lofar(audio, title=f"{name} - LOFAR", ax=axes[idx], max_freq=2000)
plt.tight_layout()
plt.savefig('results/lofar_comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig2)

print("  [3/4] Parameter comparison...")
fig3 = visualizer.plot_parameter_comparison(
    scenarios, save_path='results/parameter_comparison.png'
)
plt.close(fig3)

print("  [4/4] Power spectrum analysis...")
fig4, axes = plt.subplots(2, 2, figsize=(16, 8))
axes = axes.flatten()
for idx, (audio, params, name) in enumerate(scenarios):
    visualizer.plot_power_spectrum(audio, title=f"{name}", ax=axes[idx])
plt.tight_layout()
plt.savefig('results/power_spectrum_comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig4)

print("✓ All visualizations saved!")
print()

# 통계 분석
print("Computing signal statistics...")
print("-" * 60)
for audio, params, name in scenarios:
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    crest_factor = peak / (rms + 1e-10)

    spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1/SAMPLE_RATE)
    power = np.abs(spectrum) ** 2
    spectral_centroid = np.sum(freqs * power) / (np.sum(power) + 1e-10)

    print(f"  {name}:")
    print(f"    RMS: {rms:.4f}, Peak: {peak:.4f}, Crest: {crest_factor:.2f}")
    print(f"    Spectral Centroid: {spectral_centroid:.1f} Hz")

print()

# 리포트 생성
print("Generating summary report...")
report = create_summary_report(scenarios, output_dir='results')
with open('results/feasibility_result.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("✓ Report saved: results/feasibility_result.txt")
print()

# 오디오 저장
print("Saving audio files...")
try:
    import soundfile as sf
    for audio, params, name in scenarios:
        audio_normalized = audio / (np.max(np.abs(audio)) + 1e-8) * 0.9
        filename = name.lower().replace(' ', '_').replace('-', '_')
        filepath = f'results/audio/{filename}.wav'
        sf.write(filepath, audio_normalized, SAMPLE_RATE)
        print(f"  ✓ {filepath}")
    print()
except ImportError:
    print("  ! soundfile not installed, skipping audio export")
    print()

print("="*80)
print("✓ Feasibility test completed successfully!")
print("="*80)
print()
print("Results saved in:")
print("  - results/scenario_comparison.png")
print("  - results/lofar_comparison.png")
print("  - results/parameter_comparison.png")
print("  - results/power_spectrum_comparison.png")
print("  - results/feasibility_result.txt")
print("  - results/audio/*.wav")
print()
