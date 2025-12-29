"""
Real data validation and comparison utilities
실제 데이터 검증 및 비교 유틸리티
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy import signal, stats
from typing import Dict, Tuple, List
import os


class AcousticFeatureExtractor:
    """음향 특징 추출기"""

    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate

    def extract_features(self, audio: np.ndarray) -> Dict:
        """
        음향 신호에서 특징 추출

        Args:
            audio: 오디오 신호

        Returns:
            특징 딕셔너리
        """
        features = {}

        # 시간 도메인 특징
        features['rms'] = np.sqrt(np.mean(audio ** 2))
        features['peak'] = np.max(np.abs(audio))
        features['crest_factor'] = features['peak'] / (features['rms'] + 1e-10)
        features['zero_crossing_rate'] = np.mean(librosa.zero_crossings(audio))

        # 주파수 도메인 특징
        spectrum = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/self.sr)
        power = np.abs(spectrum) ** 2

        # 스펙트럼 중심
        features['spectral_centroid'] = np.sum(freqs * power) / (np.sum(power) + 1e-10)

        # 스펙트럼 확산 (대역폭)
        features['spectral_spread'] = np.sqrt(
            np.sum(((freqs - features['spectral_centroid']) ** 2) * power) /
            (np.sum(power) + 1e-10)
        )

        # 스펙트럼 롤오프 (95% 에너지 주파수)
        cumsum_power = np.cumsum(power)
        threshold = 0.95 * cumsum_power[-1]
        rolloff_idx = np.where(cumsum_power >= threshold)[0]
        features['spectral_rolloff'] = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]

        # 스펙트럼 플럭스 (변화율)
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        flux = np.sqrt(np.sum(np.diff(np.abs(stft), axis=1) ** 2, axis=0))
        features['spectral_flux_mean'] = np.mean(flux)
        features['spectral_flux_std'] = np.std(flux)

        # LOFAR 특징 (저주파 토널 성분)
        lofar_freqs = freqs[freqs <= 2000]
        lofar_power = power[:len(lofar_freqs)]

        if len(lofar_power) > 0:
            # 토널 성분 탐지 (피크)
            peaks, properties = signal.find_peaks(lofar_power,
                                                  height=np.mean(lofar_power) * 2,
                                                  distance=10)
            features['num_tonal_components'] = len(peaks)
            features['dominant_frequency'] = lofar_freqs[np.argmax(lofar_power)]

            # 저주파(<500Hz) vs 고주파 비율
            low_freq_mask = lofar_freqs <= 500
            low_freq_energy = np.sum(lofar_power[low_freq_mask])
            high_freq_energy = np.sum(lofar_power[~low_freq_mask]) + 1e-10
            features['low_high_ratio'] = low_freq_energy / high_freq_energy
        else:
            features['num_tonal_components'] = 0
            features['dominant_frequency'] = 0
            features['low_high_ratio'] = 0

        # 스펙트럼 기울기 (로그 스케일)
        log_power = 10 * np.log10(power + 1e-10)
        valid_mask = np.isfinite(log_power) & (freqs > 10) & (freqs < self.sr/2)
        if np.sum(valid_mask) > 10:
            slope, _, _, _, _ = stats.linregress(freqs[valid_mask], log_power[valid_mask])
            features['spectral_slope'] = slope
        else:
            features['spectral_slope'] = 0

        return features


class SyntheticRealComparator:
    """합성 신호와 실제 신호 비교기"""

    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
        self.extractor = AcousticFeatureExtractor(sample_rate)

    def compare_features(self,
                        synthetic: np.ndarray,
                        real: np.ndarray,
                        label: str = "Comparison") -> Dict:
        """
        합성 신호와 실제 신호의 특징 비교

        Args:
            synthetic: 합성 신호
            real: 실제 신호
            label: 비교 레이블

        Returns:
            비교 결과 딕셔너리
        """
        # 신호 길이 맞추기
        min_len = min(len(synthetic), len(real))
        synthetic = synthetic[:min_len]
        real = real[:min_len]

        # 특징 추출
        synthetic_features = self.extractor.extract_features(synthetic)
        real_features = self.extractor.extract_features(real)

        # 차이 계산
        comparison = {
            'label': label,
            'synthetic_features': synthetic_features,
            'real_features': real_features,
            'differences': {}
        }

        for key in synthetic_features:
            synth_val = synthetic_features[key]
            real_val = real_features[key]

            # 상대 오차 계산
            if isinstance(synth_val, (int, float)) and isinstance(real_val, (int, float)):
                if abs(real_val) > 1e-10:
                    rel_error = abs(synth_val - real_val) / abs(real_val)
                else:
                    rel_error = abs(synth_val - real_val)

                comparison['differences'][key] = {
                    'synthetic': synth_val,
                    'real': real_val,
                    'absolute_diff': synth_val - real_val,
                    'relative_error': rel_error
                }

        return comparison

    def spectral_similarity(self,
                           synthetic: np.ndarray,
                           real: np.ndarray) -> float:
        """
        스펙트럼 유사도 계산 (코사인 유사도)

        Args:
            synthetic: 합성 신호
            real: 실제 신호

        Returns:
            유사도 (0-1)
        """
        # FFT
        synth_spectrum = np.abs(np.fft.rfft(synthetic))
        real_spectrum = np.abs(np.fft.rfft(real))

        # 길이 맞추기
        min_len = min(len(synth_spectrum), len(real_spectrum))
        synth_spectrum = synth_spectrum[:min_len]
        real_spectrum = real_spectrum[:min_len]

        # 정규화
        synth_spectrum = synth_spectrum / (np.linalg.norm(synth_spectrum) + 1e-10)
        real_spectrum = real_spectrum / (np.linalg.norm(real_spectrum) + 1e-10)

        # 코사인 유사도
        similarity = np.dot(synth_spectrum, real_spectrum)

        return similarity

    def visualize_comparison(self,
                            synthetic: np.ndarray,
                            real: np.ndarray,
                            title: str = "Synthetic vs Real",
                            save_path: str = None) -> plt.Figure:
        """
        합성 신호와 실제 신호 시각화 비교

        Args:
            synthetic: 합성 신호
            real: 실제 신호
            title: 그래프 제목
            save_path: 저장 경로

        Returns:
            matplotlib figure
        """
        fig = plt.figure(figsize=(16, 12))

        # 1. 파형 비교
        ax1 = plt.subplot(4, 2, 1)
        time_synth = np.arange(len(synthetic)) / self.sr
        ax1.plot(time_synth, synthetic, linewidth=0.5, alpha=0.8, label='Synthetic')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Synthetic Waveform')
        ax1.grid(True, alpha=0.3)

        ax2 = plt.subplot(4, 2, 2)
        time_real = np.arange(len(real)) / self.sr
        ax2.plot(time_real, real, linewidth=0.5, alpha=0.8, label='Real', color='orange')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude')
        ax2.set_title('Real Waveform')
        ax2.grid(True, alpha=0.3)

        # 2. 스펙트로그램 비교
        ax3 = plt.subplot(4, 2, 3)
        D_synth = librosa.stft(synthetic, n_fft=2048, hop_length=512)
        S_db_synth = librosa.amplitude_to_db(np.abs(D_synth), ref=np.max)
        librosa.display.specshow(S_db_synth, sr=self.sr, hop_length=512,
                                x_axis='time', y_axis='hz', ax=ax3, cmap='viridis')
        ax3.set_title('Synthetic Spectrogram')

        ax4 = plt.subplot(4, 2, 4)
        D_real = librosa.stft(real, n_fft=2048, hop_length=512)
        S_db_real = librosa.amplitude_to_db(np.abs(D_real), ref=np.max)
        librosa.display.specshow(S_db_real, sr=self.sr, hop_length=512,
                                x_axis='time', y_axis='hz', ax=ax4, cmap='viridis')
        ax4.set_title('Real Spectrogram')

        # 3. 파워 스펙트럼 비교 (오버레이)
        ax5 = plt.subplot(4, 2, 5)
        freqs_synth = np.fft.rfftfreq(len(synthetic), 1/self.sr)
        spectrum_synth = np.abs(np.fft.rfft(synthetic))
        power_db_synth = 20 * np.log10(spectrum_synth + 1e-10)

        freqs_real = np.fft.rfftfreq(len(real), 1/self.sr)
        spectrum_real = np.abs(np.fft.rfft(real))
        power_db_real = 20 * np.log10(spectrum_real + 1e-10)

        ax5.plot(freqs_synth, power_db_synth, linewidth=0.8, alpha=0.7, label='Synthetic')
        ax5.plot(freqs_real, power_db_real, linewidth=0.8, alpha=0.7, label='Real')
        ax5.set_xlabel('Frequency (Hz)')
        ax5.set_ylabel('Power (dB)')
        ax5.set_title('Power Spectrum Overlay')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim([0, min(self.sr/2, 5000)])

        # 4. LOFAR 비교
        ax6 = plt.subplot(4, 2, 6)
        max_freq = 2000
        mask_synth = freqs_synth <= max_freq
        mask_real = freqs_real <= max_freq

        ax6.plot(freqs_synth[mask_synth], power_db_synth[mask_synth],
                linewidth=0.8, alpha=0.7, label='Synthetic')
        ax6.plot(freqs_real[mask_real], power_db_real[mask_real],
                linewidth=0.8, alpha=0.7, label='Real')
        ax6.set_xlabel('Frequency (Hz)')
        ax6.set_ylabel('Power (dB)')
        ax6.set_title('LOFAR Spectrum (0-2000 Hz)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_xlim([0, max_freq])

        # 5. 특징 비교 (바 차트)
        ax7 = plt.subplot(4, 2, 7)
        comparison = self.compare_features(synthetic, real)

        key_features = ['rms', 'spectral_centroid', 'spectral_spread', 'spectral_rolloff']
        synth_vals = [comparison['synthetic_features'][k] for k in key_features]
        real_vals = [comparison['real_features'][k] for k in key_features]

        x = np.arange(len(key_features))
        width = 0.35
        ax7.bar(x - width/2, synth_vals, width, label='Synthetic', alpha=0.7)
        ax7.bar(x + width/2, real_vals, width, label='Real', alpha=0.7)
        ax7.set_xticks(x)
        ax7.set_xticklabels(key_features, rotation=15, ha='right')
        ax7.set_ylabel('Value')
        ax7.set_title('Feature Comparison')
        ax7.legend()
        ax7.grid(True, alpha=0.3, axis='y')

        # 6. 스펙트럼 유사도
        ax8 = plt.subplot(4, 2, 8)
        similarity = self.spectral_similarity(synthetic, real)

        ax8.text(0.5, 0.6, f'Spectral Similarity',
                ha='center', va='center', fontsize=14, fontweight='bold')
        ax8.text(0.5, 0.4, f'{similarity:.4f}',
                ha='center', va='center', fontsize=32, color='blue')
        ax8.text(0.5, 0.2, '(Cosine Similarity: 0=different, 1=identical)',
                ha='center', va='center', fontsize=10, style='italic')
        ax8.set_xlim([0, 1])
        ax8.set_ylim([0, 1])
        ax8.axis('off')

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig


def load_real_audio(file_path: str, target_sr: int = 16000, duration: float = 3.0) -> np.ndarray:
    """
    실제 오디오 파일 로드

    Args:
        file_path: 오디오 파일 경로
        target_sr: 타겟 샘플링 레이트
        duration: 로드할 길이 (초)

    Returns:
        오디오 신호
    """
    try:
        audio, sr = librosa.load(file_path, sr=target_sr, duration=duration, mono=True)
        return audio
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def generate_comparison_report(comparisons: List[Dict], output_path: str):
    """
    비교 결과 리포트 생성

    Args:
        comparisons: 비교 결과 리스트
        output_path: 저장 경로
    """
    lines = []
    lines.append("=" * 80)
    lines.append("Synthetic vs Real Audio Comparison Report")
    lines.append("=" * 80)
    lines.append("")

    for idx, comp in enumerate(comparisons, 1):
        lines.append(f"{idx}. {comp['label']}")
        lines.append("-" * 60)

        # 주요 특징 비교
        key_features = ['spectral_centroid', 'spectral_spread', 'spectral_rolloff',
                       'low_high_ratio', 'num_tonal_components']

        for feat in key_features:
            if feat in comp['differences']:
                diff = comp['differences'][feat]
                lines.append(f"  {feat}:")
                lines.append(f"    Synthetic: {diff['synthetic']:.2f}")
                lines.append(f"    Real:      {diff['real']:.2f}")
                lines.append(f"    Rel Error: {diff['relative_error']:.2%}")

        lines.append("")

    lines.append("=" * 80)

    report = "\n".join(lines)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    return report
