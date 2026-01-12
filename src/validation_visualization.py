#!/usr/bin/env python3
"""
실제 데이터 vs 합성 데이터 직관적 비교 시각화
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path

from src.physics_simulator import ScenarioSynthesizer
from src.data_validation import SyntheticRealComparator
from src.validation.dataset_integration import IntegratedDatasetLoader


def plot_comparison(synth_audio, real_audio, dataset_name, vessel_type,
                   sample_rate=16000, save_path=None):
    """
    합성 vs 실제 데이터 직관적 비교

    Args:
        synth_audio: 합성 오디오
        real_audio: 실제 오디오
        dataset_name: 데이터셋 이름 (DeepShip, ShipsEar)
        vessel_type: 선박 유형
        sample_rate: 샘플링 레이트
        save_path: 저장 경로
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    time = np.arange(len(synth_audio)) / sample_rate

    # 1. 파형 비교 (좌: 합성, 우: 실제)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, synth_audio, linewidth=0.5, color='steelblue')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Synthetic Waveform', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-1, 1])

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time, real_audio, linewidth=0.5, color='coral')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title(f'Real Waveform ({dataset_name})', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-1, 1])

    # 2. 스펙트로그램 비교
    ax3 = fig.add_subplot(gs[1, 0])
    D_synth = librosa.stft(synth_audio, n_fft=2048, hop_length=512)
    S_db_synth = librosa.amplitude_to_db(np.abs(D_synth), ref=np.max)
    img1 = librosa.display.specshow(S_db_synth, sr=sample_rate, hop_length=512,
                                    x_axis='time', y_axis='hz', ax=ax3,
                                    cmap='viridis')
    ax3.set_title('Synthetic Spectrogram', fontweight='bold')
    ax3.set_ylim([0, 4000])
    plt.colorbar(img1, ax=ax3, format='%+2.0f dB')

    ax4 = fig.add_subplot(gs[1, 1])
    D_real = librosa.stft(real_audio, n_fft=2048, hop_length=512)
    S_db_real = librosa.amplitude_to_db(np.abs(D_real), ref=np.max)
    img2 = librosa.display.specshow(S_db_real, sr=sample_rate, hop_length=512,
                                    x_axis='time', y_axis='hz', ax=ax4,
                                    cmap='viridis')
    ax4.set_title(f'Real Spectrogram ({dataset_name})', fontweight='bold')
    ax4.set_ylim([0, 4000])
    plt.colorbar(img2, ax=ax4, format='%+2.0f dB')

    # 3. Power Spectrum 비교
    ax5 = fig.add_subplot(gs[2, :])

    # 합성 데이터 스펙트럼
    freqs_synth = np.fft.rfftfreq(len(synth_audio), 1/sample_rate)
    spectrum_synth = np.abs(np.fft.rfft(synth_audio))
    power_synth = 20 * np.log10(spectrum_synth + 1e-10)

    # 실제 데이터 스펙트럼
    freqs_real = np.fft.rfftfreq(len(real_audio), 1/sample_rate)
    spectrum_real = np.abs(np.fft.rfft(real_audio))
    power_real = 20 * np.log10(spectrum_real + 1e-10)

    ax5.plot(freqs_synth, power_synth, label='Synthetic', alpha=0.8,
            linewidth=1.5, color='steelblue')
    ax5.plot(freqs_real, power_real, label=f'Real ({dataset_name})', alpha=0.8,
            linewidth=1.5, color='coral')
    ax5.set_xlabel('Frequency (Hz)')
    ax5.set_ylabel('Power (dB)')
    ax5.set_title('Power Spectrum Comparison', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([0, 3000])

    # 유사도 계산
    comparator = SyntheticRealComparator(sample_rate=sample_rate)
    similarity = comparator.spectral_similarity(synth_audio, real_audio)

    # 전체 타이틀
    fig.suptitle(f'{vessel_type} Vessel: Synthetic vs Real ({dataset_name})\n'
                f'Spectral Similarity: {similarity:.4f}',
                fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")

    plt.close(fig)
    return similarity


def main():
    """메인 검증 시각화"""
    print("="*80)
    print("Physics Model Validation: Intuitive Comparison")
    print("="*80)
    print()

    # 초기화
    loader = IntegratedDatasetLoader()
    synthesizer = ScenarioSynthesizer(sample_rate=16000, duration=3.0)

    output_dir = Path('results/validation')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 검증할 선박 유형
    vessel_configs = [
        ('Cargo', 15.0, 150),
        ('Tug', 10.0, 200),
        ('Passengers', 20.0, 250)
    ]

    results = []

    for vessel_type, speed, rpm in vessel_configs:
        print(f"\n{'='*60}")
        print(f"Validating: {vessel_type}")
        print('='*60)

        # 실제 데이터 로드
        try:
            real_data = loader.get_combined_vessel_samples(
                vessel_type,
                n_samples_per_dataset=1,
                target_sr=16000,
                duration=3.0
            )

            deepship_samples = real_data['deepship']
            shipsear_samples = real_data['shipsear']

        except Exception as e:
            print(f"  ✗ Error loading data: {e}")
            continue

        # 합성 데이터 생성
        print(f"  Synthesizing {vessel_type} vessel...")
        audio_synth, _ = synthesizer.synthesize_highspeed_vessel(
            speed=speed + np.random.uniform(-1, 1),
            propeller_rpm=rpm + np.random.uniform(-5, 5),
            distance=1000.0 + np.random.uniform(-100, 100)
        )

        # DeepShip 비교
        if len(deepship_samples) > 0:
            print(f"  Comparing with DeepShip...")
            sim_deepship = plot_comparison(
                audio_synth,
                deepship_samples[0],
                'DeepShip',
                vessel_type,
                save_path=output_dir / f'{vessel_type.lower()}_deepship.png'
            )
            results.append({
                'vessel': vessel_type,
                'dataset': 'DeepShip',
                'similarity': sim_deepship
            })

        # ShipsEar 비교
        if len(shipsear_samples) > 0:
            print(f"  Comparing with ShipsEar...")
            sim_shipsear = plot_comparison(
                audio_synth,
                shipsear_samples[0],
                'ShipsEar',
                vessel_type,
                save_path=output_dir / f'{vessel_type.lower()}_shipsear.png'
            )
            results.append({
                'vessel': vessel_type,
                'dataset': 'ShipsEar',
                'similarity': sim_shipsear
            })

    # 요약
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"\n{'Vessel':<15} {'Dataset':<15} {'Similarity':<15}")
    print("-" * 45)

    for res in results:
        print(f"{res['vessel']:<15} {res['dataset']:<15} {res['similarity']:<15.4f}")

    if results:
        avg_sim = np.mean([r['similarity'] for r in results])
        print("\n" + "-" * 45)
        print(f"{'Average':<15} {'Overall':<15} {avg_sim:<15.4f}")

        if avg_sim > 0.5:
            verdict = "✓ VALIDATED"
        elif avg_sim > 0.4:
            verdict = "~ MODERATE"
        else:
            verdict = "✗ NEEDS WORK"

        print(f"\nResult: {verdict}")

    print("\n" + "="*80)
    print(f"✓ Visualizations saved in: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
