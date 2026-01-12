#!/usr/bin/env python3
"""
종합 검증 시스템
DeepShip + ShipsEar 통합 검증
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.physics_simulator import ScenarioSynthesizer
from src.data_validation import SyntheticRealComparator, generate_comparison_report
from src.validation.dataset_integration import IntegratedDatasetLoader


def validate_vessel_type(vessel_type: str,
                        loader: IntegratedDatasetLoader,
                        synthesizer: ScenarioSynthesizer,
                        comparator: SyntheticRealComparator,
                        output_dir: str = 'results/comprehensive_validation'):
    """
    특정 선박 유형에 대한 종합 검증

    Args:
        vessel_type: 선박 유형 ('Cargo', 'Tug', 'Passengers')
        loader: 데이터 로더
        synthesizer: 시뮬레이터
        comparator: 비교기
        output_dir: 출력 디렉터리
    """
    print(f"\n{'='*80}")
    print(f"Validating: {vessel_type}")
    print('='*80)

    # 파라미터 매핑
    params_mapping = {
        'Cargo': {'speed': 15.0, 'rpm': 150},
        'Tug': {'speed': 10.0, 'rpm': 200},
        'Passengers': {'speed': 20.0, 'rpm': 250},
        'Tanker': {'speed': 12.0, 'rpm': 120}
    }

    if vessel_type not in params_mapping:
        print(f"Warning: No parameters for {vessel_type}, using defaults")
        params_mapping[vessel_type] = {'speed': 15.0, 'rpm': 150}

    params = params_mapping[vessel_type]

    # 실제 데이터 로드
    print("\nLoading real data from both datasets...")
    try:
        real_data = loader.get_combined_vessel_samples(
            vessel_type,
            n_samples_per_dataset=3,
            target_sr=16000,
            duration=3.0
        )

        deepship_samples = real_data['deepship']
        shipsear_samples = real_data['shipsear']

        print(f"  DeepShip: {len(deepship_samples)} samples")
        print(f"  ShipsEar: {len(shipsear_samples)} samples")

        if len(deepship_samples) == 0 and len(shipsear_samples) == 0:
            print(f"  ✗ No data available for {vessel_type}")
            return None

    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        return None

    # 합성 데이터 생성
    n_synth = len(deepship_samples) + len(shipsear_samples)
    print(f"\nSynthesizing {n_synth} samples...")

    synthetic_samples = []
    for i in range(n_synth):
        speed = params['speed'] + np.random.uniform(-2, 2)
        rpm = params['rpm'] + np.random.uniform(-10, 10)

        audio, _ = synthesizer.synthesize_highspeed_vessel(
            speed=speed,
            propeller_rpm=rpm,
            distance=1000.0 + np.random.uniform(-200, 200)
        )
        synthetic_samples.append(audio)

    # DeepShip과 비교
    results = {
        'deepship': {'similarities': [], 'comparisons': []},
        'shipsear': {'similarities': [], 'comparisons': []}
    }

    if len(deepship_samples) > 0:
        print("\nComparing with DeepShip...")
        for i, (synth, real) in enumerate(zip(synthetic_samples[:len(deepship_samples)],
                                              deepship_samples)):
            comp = comparator.compare_features(synth, real, label=f"DeepShip-{vessel_type}-{i+1}")
            sim = comparator.spectral_similarity(synth, real)

            results['deepship']['comparisons'].append(comp)
            results['deepship']['similarities'].append(sim)

            print(f"  Sample {i+1}: {sim:.4f}")

    # ShipsEar와 비교
    if len(shipsear_samples) > 0:
        print("\nComparing with ShipsEar...")
        synth_offset = len(deepship_samples)
        for i, (synth, real) in enumerate(zip(synthetic_samples[synth_offset:],
                                              shipsear_samples)):
            comp = comparator.compare_features(synth, real, label=f"ShipsEar-{vessel_type}-{i+1}")
            sim = comparator.spectral_similarity(synth, real)

            results['shipsear']['comparisons'].append(comp)
            results['shipsear']['similarities'].append(sim)

            print(f"  Sample {i+1}: {sim:.4f}")

    # 통계
    deepship_avg = np.mean(results['deepship']['similarities']) if results['deepship']['similarities'] else 0
    shipsear_avg = np.mean(results['shipsear']['similarities']) if results['shipsear']['similarities'] else 0

    all_sims = results['deepship']['similarities'] + results['shipsear']['similarities']
    overall_avg = np.mean(all_sims) if all_sims else 0

    print(f"\n{'='*60}")
    print(f"Results Summary - {vessel_type}")
    print('='*60)
    print(f"DeepShip Avg Similarity: {deepship_avg:.4f}")
    print(f"ShipsEar Avg Similarity: {shipsear_avg:.4f}")
    print(f"Overall Avg Similarity:  {overall_avg:.4f}")

    # 시각화
    output_path = Path(output_dir) / vessel_type.lower()
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Similarity 비교
    ax = axes[0, 0]
    x = list(range(len(results['deepship']['similarities'])))
    y = list(range(len(results['shipsear']['similarities'])))

    if x:
        ax.bar([i - 0.2 for i in x], results['deepship']['similarities'],
               width=0.4, label='DeepShip', alpha=0.7, color='steelblue')
    if y:
        ax.bar([len(x) + i + 0.2 for i in y], results['shipsear']['similarities'],
               width=0.4, label='ShipsEar', alpha=0.7, color='coral')

    ax.axhline(y=overall_avg, color='red', linestyle='--', linewidth=2,
               label=f'Overall Avg: {overall_avg:.3f}')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Spectral Similarity')
    ax.set_title(f'{vessel_type}: Per-Sample Similarity')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])

    # 2. Dataset 비교
    ax = axes[0, 1]
    datasets = []
    avgs = []
    if deepship_avg > 0:
        datasets.append('DeepShip')
        avgs.append(deepship_avg)
    if shipsear_avg > 0:
        datasets.append('ShipsEar')
        avgs.append(shipsear_avg)

    colors_bar = ['steelblue', 'coral'][:len(datasets)]
    ax.bar(datasets, avgs, color=colors_bar, alpha=0.7)
    ax.set_ylabel('Average Similarity')
    ax.set_title(f'{vessel_type}: Dataset Comparison')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Feature 비교 (DeepShip)
    ax = axes[1, 0]
    if results['deepship']['comparisons']:
        features = ['spectral_centroid', 'spectral_spread', 'spectral_rolloff']
        synth_vals = []
        real_vals = []

        for feat in features:
            synth_avg = np.mean([c['synthetic_features'][feat] for c in results['deepship']['comparisons']])
            real_avg = np.mean([c['real_features'][feat] for c in results['deepship']['comparisons']])
            synth_vals.append(synth_avg)
            real_vals.append(real_avg)

        x = np.arange(len(features))
        width = 0.35
        ax.bar(x - width/2, synth_vals, width, label='Synthetic', alpha=0.7)
        ax.bar(x + width/2, real_vals, width, label='DeepShip', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=15, ha='right', fontsize=9)
        ax.set_ylabel('Value (Hz)')
        ax.set_title('Feature Comparison (DeepShip)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No DeepShip data', ha='center', va='center')
        ax.axis('off')

    # 4. Feature 비교 (ShipsEar)
    ax = axes[1, 1]
    if results['shipsear']['comparisons']:
        features = ['spectral_centroid', 'spectral_spread', 'spectral_rolloff']
        synth_vals = []
        real_vals = []

        for feat in features:
            synth_avg = np.mean([c['synthetic_features'][feat] for c in results['shipsear']['comparisons']])
            real_avg = np.mean([c['real_features'][feat] for c in results['shipsear']['comparisons']])
            synth_vals.append(synth_avg)
            real_vals.append(real_avg)

        x = np.arange(len(features))
        width = 0.35
        ax.bar(x - width/2, synth_vals, width, label='Synthetic', alpha=0.7)
        ax.bar(x + width/2, real_vals, width, label='ShipsEar', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=15, ha='right', fontsize=9)
        ax.set_ylabel('Value (Hz)')
        ax.set_title('Feature Comparison (ShipsEar)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No ShipsEar data', ha='center', va='center')
        ax.axis('off')

    plt.suptitle(f'Comprehensive Validation: {vessel_type}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Visualization saved: {output_path / 'summary.png'}")

    return {
        'vessel_type': vessel_type,
        'deepship_avg': deepship_avg,
        'shipsear_avg': shipsear_avg,
        'overall_avg': overall_avg,
        'n_deepship': len(deepship_samples),
        'n_shipsear': len(shipsear_samples)
    }


def main():
    """메인 검증 프로세스"""
    print("="*80)
    print("Comprehensive Validation System")
    print("DeepShip + ShipsEar Integration")
    print("="*80)

    # 초기화
    loader = IntegratedDatasetLoader()
    synthesizer = ScenarioSynthesizer(sample_rate=16000, duration=3.0)
    comparator = SyntheticRealComparator(sample_rate=16000)

    os.makedirs('results/comprehensive_validation', exist_ok=True)

    # 검증할 선박 유형
    vessel_types = ['Cargo', 'Tug', 'Passengers']

    all_results = []

    for vessel_type in vessel_types:
        result = validate_vessel_type(
            vessel_type,
            loader,
            synthesizer,
            comparator
        )

        if result:
            all_results.append(result)

    # 최종 요약
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    summary_data = []
    for res in all_results:
        print(f"\n{res['vessel_type']}:")
        print(f"  DeepShip:  {res['deepship_avg']:.4f} ({res['n_deepship']} samples)")
        print(f"  ShipsEar:  {res['shipsear_avg']:.4f} ({res['n_shipsear']} samples)")
        print(f"  Overall:   {res['overall_avg']:.4f}")

        summary_data.append(res)

    # 전체 평균
    if summary_data:
        overall_deepship = np.mean([r['deepship_avg'] for r in summary_data if r['deepship_avg'] > 0])
        overall_shipsear = np.mean([r['shipsear_avg'] for r in summary_data if r['shipsear_avg'] > 0])
        overall_all = np.mean([r['overall_avg'] for r in summary_data])

        print(f"\n{'='*60}")
        print("CROSS-DATASET AVERAGE")
        print('='*60)
        print(f"DeepShip Average:  {overall_deepship:.4f}")
        print(f"ShipsEar Average:  {overall_shipsear:.4f}")
        print(f"Overall Average:   {overall_all:.4f}")

        # 평가
        if overall_all > 0.6:
            verdict = "✓ EXCELLENT"
        elif overall_all > 0.5:
            verdict = "✓ GOOD"
        elif overall_all > 0.4:
            verdict = "~ MODERATE"
        else:
            verdict = "✗ NEEDS IMPROVEMENT"

        print(f"\nValidation Result: {verdict}")

        # 최종 시각화
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 1. 선박 유형별 비교
        ax = axes[0]
        vessel_names = [r['vessel_type'] for r in summary_data]
        deepship_avgs = [r['deepship_avg'] for r in summary_data]
        shipsear_avgs = [r['shipsear_avg'] for r in summary_data]

        x = np.arange(len(vessel_names))
        width = 0.35

        ax.bar(x - width/2, deepship_avgs, width, label='DeepShip', alpha=0.7, color='steelblue')
        ax.bar(x + width/2, shipsear_avgs, width, label='ShipsEar', alpha=0.7, color='coral')
        ax.set_xlabel('Vessel Type')
        ax.set_ylabel('Average Similarity')
        ax.set_title('Similarity by Vessel Type and Dataset')
        ax.set_xticks(x)
        ax.set_xticklabels(vessel_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])

        # 2. 전체 요약
        ax = axes[1]
        ax.axis('off')

        summary_text = f"""
COMPREHENSIVE VALIDATION RESULTS

Datasets Used:
  • DeepShip: {sum(r['n_deepship'] for r in summary_data)} samples
  • ShipsEar: {sum(r['n_shipsear'] for r in summary_data)} samples

Average Similarity:
  • DeepShip:  {overall_deepship:.4f}
  • ShipsEar:  {overall_shipsear:.4f}
  • Overall:   {overall_all:.4f}

Vessel Types Tested: {len(summary_data)}

Validation: {verdict}

Recommendation:
  {'Physics model is validated.' if overall_all > 0.5 else 'Requires further tuning.'}
  {'Ready for Neural Codec phase.' if overall_all > 0.5 else 'Additional parameter adjustment needed.'}
"""

        ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center')

        plt.tight_layout()
        plt.savefig('results/comprehensive_validation/overall_summary.png',
                   dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n✓ Overall summary saved: results/comprehensive_validation/overall_summary.png")

    print("\n" + "="*80)
    print("✓ Comprehensive validation completed!")
    print("="*80)


if __name__ == "__main__":
    main()
