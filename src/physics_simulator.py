"""
Physics-based Underwater Acoustic Simulator
물리 기반 수중 음향 시뮬레이터
"""

import numpy as np
from scipy import signal
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class UnderwaterAcousticSimulator:
    """수중 음향 전파 및 선박 소음 물리 시뮬레이터"""

    def __init__(self, sample_rate: int = 16000, duration: float = 3.0):
        """
        Args:
            sample_rate: 샘플링 레이트 (Hz)
            duration: 신호 길이 (초)
        """
        self.sr = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        self.t = np.linspace(0, duration, self.n_samples)

        # 물리 상수
        self.c_water = 1500  # 수중 음속 (m/s)
        self.rho_water = 1025  # 해수 밀도 (kg/m³)
        self.ref_pressure = 1e-6  # 기준 압력 1 μPa

    def cavitation_noise(self,
                        speed_knots: float,
                        propeller_rpm: float,
                        n_blades: int = 5,
                        cavitation_strength: float = 1.0) -> np.ndarray:
        """
        캐비테이션 소음 생성 (Brown's Formula 기반)

        Args:
            speed_knots: 선박 속도 (노트)
            propeller_rpm: 프로펠러 회전수
            n_blades: 프로펠러 날개 수
            cavitation_strength: 캐비테이션 강도 (0-1)

        Returns:
            캐비테이션 소음 신호
        """
        if cavitation_strength == 0:
            return np.zeros(self.n_samples)

        # Blade Rate Frequency
        brf = (propeller_rpm * n_blades) / 60.0

        # 광대역 캐비테이션 소음 (f^-2 스펙트럼)
        noise = np.random.randn(self.n_samples)

        # 주파수 영역에서 필터링 (f^-2 특성)
        freqs = np.fft.rfftfreq(self.n_samples, 1/self.sr)
        spectrum = np.fft.rfft(noise)

        # 캐비테이션 스펙트럼 형상 (저주파 우세)
        with np.errstate(divide='ignore', invalid='ignore'):
            shaping = np.where(freqs > 10, 1.0 / (freqs ** 1.5), 1.0)
        shaping[0] = 0  # DC 제거

        spectrum *= shaping * cavitation_strength
        broadband = np.fft.irfft(spectrum, n=self.n_samples)

        # BRF 토널 성분 추가
        tonal = np.zeros(self.n_samples)
        for harmonic in range(1, 8):
            freq = brf * harmonic
            if freq < self.sr / 2:
                amplitude = cavitation_strength / (harmonic ** 0.5)
                tonal += amplitude * np.sin(2 * np.pi * freq * self.t)

        # 속도 기반 소음 레벨 조정
        speed_factor = (speed_knots / 30.0) ** 2  # 속도 제곱 비례

        return (broadband + tonal) * speed_factor

    def machinery_noise(self,
                       rpm: float,
                       n_cylinders: int = 8,
                       noise_level: float = 0.5) -> np.ndarray:
        """
        기계 소음 생성 (엔진, 펌프 등)

        Args:
            rpm: 엔진 회전수
            n_cylinders: 실린더 수
            noise_level: 소음 레벨 (0-1)

        Returns:
            기계 소음 신호
        """
        # 점화 주파수 (4행정 엔진)
        firing_freq = (rpm * n_cylinders) / (60 * 2)

        # 기본 토널 + 고조파
        tonal = np.zeros(self.n_samples)
        for harmonic in range(1, 10):
            freq = firing_freq * harmonic
            if freq < self.sr / 2:
                amplitude = noise_level / harmonic
                phase = np.random.rand() * 2 * np.pi
                tonal += amplitude * np.sin(2 * np.pi * freq * self.t + phase)

        # 저주파 광대역 소음
        broadband = np.random.randn(self.n_samples) * noise_level * 0.3
        broadband = signal.filtfilt(*signal.butter(4, 500/(self.sr/2), 'low'), broadband)

        return tonal + broadband

    def doppler_shift(self,
                     audio: np.ndarray,
                     velocity_mps: float,
                     approaching: bool = True) -> np.ndarray:
        """
        도플러 효과 적용

        Args:
            audio: 입력 신호
            velocity_mps: 상대 속도 (m/s)
            approaching: True면 접근, False면 멀어짐

        Returns:
            도플러 효과가 적용된 신호
        """
        # 도플러 비율 계산
        if approaching:
            doppler_ratio = (self.c_water + velocity_mps) / self.c_water
        else:
            doppler_ratio = (self.c_water - velocity_mps) / self.c_water

        # 시간축 리샘플링으로 도플러 구현
        original_time = np.arange(len(audio)) / self.sr
        new_time = original_time / doppler_ratio

        # 보간
        shifted = np.interp(new_time, original_time, audio,
                          left=0, right=0)

        # 원래 길이로 맞춤
        if len(shifted) > len(audio):
            shifted = shifted[:len(audio)]
        else:
            shifted = np.pad(shifted, (0, len(audio) - len(shifted)))

        return shifted

    def propagation_loss(self,
                        audio: np.ndarray,
                        distance_m: float,
                        frequency_hz: float = 1000) -> np.ndarray:
        """
        음향 전파 손실 적용 (간단한 모델)

        Args:
            audio: 입력 신호
            distance_m: 전파 거리 (m)
            frequency_hz: 대표 주파수 (Hz)

        Returns:
            전파 손실이 적용된 신호
        """
        # 기하학적 확산 손실 (20 log r)
        spreading_loss_db = 20 * np.log10(max(distance_m, 1))

        # 흡수 손실 (Thorp's formula 간략화)
        f_khz = frequency_hz / 1000
        absorption_db_per_km = 0.11 * (f_khz**2) / (1 + f_khz**2) + 44 * (f_khz**2) / (4100 + f_khz**2)
        absorption_loss_db = absorption_db_per_km * (distance_m / 1000)

        # 총 손실
        total_loss_db = spreading_loss_db + absorption_loss_db
        attenuation = 10 ** (-total_loss_db / 20)

        return audio * attenuation

    def ocean_ambient_noise(self, sea_state: int = 3) -> np.ndarray:
        """
        해양 배경 소음 생성

        Args:
            sea_state: 해상 상태 (0-6)

        Returns:
            배경 소음 신호
        """
        # 해상 상태에 따른 소음 레벨
        noise_levels = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]
        noise_level = noise_levels[min(sea_state, 6)]

        # 저주파 우세 백색 소음
        noise = np.random.randn(self.n_samples) * noise_level

        # 스펙트럼 형상 (저주파 강조)
        noise = signal.filtfilt(*signal.butter(2, [50, 2000], 'band', fs=self.sr), noise)

        return noise

    def transient_event(self,
                       event_type: str = 'collision',
                       impact_energy: str = 'medium',
                       resonance_freq: float = 80) -> np.ndarray:
        """
        과도 사건 신호 생성 (충돌, 폭발 등)

        Args:
            event_type: 사건 유형
            impact_energy: 충격 에너지 ('low', 'medium', 'high')
            resonance_freq: 공진 주파수 (Hz)

        Returns:
            과도 사건 신호
        """
        energy_levels = {'low': 1.0, 'medium': 3.0, 'high': 10.0}
        amplitude = energy_levels.get(impact_energy, 3.0)

        # 초기 충격 펄스
        pulse_duration = 0.1  # 100ms
        pulse_samples = int(pulse_duration * self.sr)
        pulse = np.random.randn(pulse_samples) * amplitude

        # 구조 공진 (감쇠 사인파)
        tau_decay = 1.0  # 감쇠 시간 상수
        decay_envelope = np.exp(-self.t / tau_decay)
        resonance = amplitude * decay_envelope * np.sin(2 * np.pi * resonance_freq * self.t)

        # 조합
        event = np.zeros(self.n_samples)
        event[:pulse_samples] += pulse
        event += resonance * 0.5

        return event

    def time_varying_speed(self,
                          speed_profile: str = 'acceleration',
                          v_initial: float = 5.0,
                          v_final: float = 30.0) -> np.ndarray:
        """
        시간 가변 속도 프로파일 생성

        Args:
            speed_profile: 'acceleration', 'deceleration', 'constant'
            v_initial: 초기 속도 (knots)
            v_final: 최종 속도 (knots)

        Returns:
            시간에 따른 속도 배열 (knots)
        """
        if speed_profile == 'constant':
            return np.ones(self.n_samples) * v_initial
        elif speed_profile == 'acceleration':
            return np.linspace(v_initial, v_final, self.n_samples)
        elif speed_profile == 'deceleration':
            # 지수 감쇠
            tau = self.duration / 3
            decay = np.exp(-self.t / tau)
            return v_final + (v_initial - v_final) * decay
        else:
            return np.ones(self.n_samples) * v_initial


class ScenarioSynthesizer:
    """시나리오별 음향 신호 합성기"""

    def __init__(self, sample_rate: int = 16000, duration: float = 3.0):
        self.sim = UnderwaterAcousticSimulator(sample_rate, duration)

    def synthesize_covert_submarine(self,
                                   speed: float = 5.0,
                                   depth: float = 150.0,
                                   distance: float = 2000.0) -> Tuple[np.ndarray, Dict]:
        """잠수함 은밀 접근 시나리오"""

        # 극저속 기계 소음
        machinery = self.sim.machinery_noise(rpm=40, noise_level=0.15)

        # 캐비테이션 없음
        cavitation = np.zeros(self.sim.n_samples)

        # 유동 소음 (극소)
        flow_noise = np.random.randn(self.sim.n_samples) * 0.05
        flow_noise = signal.filtfilt(*signal.butter(2, 200/(self.sim.sr/2), 'low'), flow_noise)

        # 조합
        audio = machinery + cavitation + flow_noise

        # 전파 손실
        audio = self.sim.propagation_loss(audio, distance, frequency_hz=100)

        # 배경 소음
        audio += self.sim.ocean_ambient_noise(sea_state=2)

        # 정규화
        audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.3

        params = {
            'scenario': 'Covert Submarine',
            'speed_knots': speed,
            'depth_m': depth,
            'distance_m': distance,
            'cavitation': False,
            'source_level_db': 110,  # 추정값
        }

        return audio, params

    def synthesize_highspeed_vessel(self,
                                   speed: float = 30.0,
                                   propeller_rpm: float = 300,
                                   distance: float = 1000.0) -> Tuple[np.ndarray, Dict]:
        """고속 기동 선박 시나리오"""

        # 강한 캐비테이션
        cavitation = self.sim.cavitation_noise(
            speed_knots=speed,
            propeller_rpm=propeller_rpm,
            n_blades=5,
            cavitation_strength=1.0
        )

        # 고RPM 기계 소음
        machinery = self.sim.machinery_noise(rpm=200, noise_level=0.8)

        # 조합
        audio = cavitation * 2.0 + machinery

        # 전파 손실
        audio = self.sim.propagation_loss(audio, distance, frequency_hz=500)

        # 배경 소음
        audio += self.sim.ocean_ambient_noise(sea_state=3)

        # 정규화
        audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.8

        params = {
            'scenario': 'High-Speed Vessel',
            'speed_knots': speed,
            'propeller_rpm': propeller_rpm,
            'distance_m': distance,
            'cavitation': True,
            'source_level_db': 180,  # 추정값
        }

        return audio, params

    def synthesize_collision(self,
                           impact_energy: str = 'medium',
                           resonance_freq: float = 80) -> Tuple[np.ndarray, Dict]:
        """충돌/접촉 사고 시나리오"""

        # 충격 사건
        collision = self.sim.transient_event(
            event_type='collision',
            impact_energy=impact_energy,
            resonance_freq=resonance_freq
        )

        # 배경 소음
        audio = collision + self.sim.ocean_ambient_noise(sea_state=2)

        # 정규화
        audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.9

        params = {
            'scenario': 'Collision Event',
            'impact_energy': impact_energy,
            'resonance_freq_hz': resonance_freq,
            'source_level_db': 200,  # 추정값
        }

        return audio, params

    def synthesize_rapid_acceleration(self,
                                     v_initial: float = 5.0,
                                     v_final: float = 25.0) -> Tuple[np.ndarray, Dict]:
        """급가속 시나리오"""

        # 시간 가변 속도
        speed_profile = self.sim.time_varying_speed('acceleration', v_initial, v_final)

        # 시간에 따라 변하는 소음 생성
        audio = np.zeros(self.sim.n_samples)

        # 세그먼트별 처리
        n_segments = 10
        segment_length = len(speed_profile) // n_segments

        for i in range(n_segments):
            start_idx = i * segment_length
            end_idx = min((i + 1) * segment_length, len(audio))

            speed = speed_profile[start_idx]
            rpm = 50 + (speed / v_final) * 250  # RPM 증가

            # 세그먼트 길이에 맞는 시뮬레이터 생성
            segment_duration = (end_idx - start_idx) / self.sim.sr
            temp_sim = UnderwaterAcousticSimulator(self.sim.sr, segment_duration)

            # 캐비테이션 개시점 (15 knots 이상)
            cav_strength = max(0, (speed - 15) / 15)

            segment_audio = temp_sim.cavitation_noise(
                speed_knots=speed,
                propeller_rpm=rpm,
                cavitation_strength=cav_strength
            )

            audio[start_idx:end_idx] = segment_audio[:end_idx - start_idx]

        # 배경 소음
        audio += self.sim.ocean_ambient_noise(sea_state=3)

        # 정규화
        audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.7

        params = {
            'scenario': 'Rapid Acceleration',
            'v_initial_knots': v_initial,
            'v_final_knots': v_final,
            'acceleration_duration_s': self.sim.duration,
            'cavitation_onset': True,
        }

        return audio, params
