"""
Real underwater acoustic data loader
실제 수중 음향 데이터 로더
"""

import numpy as np
import librosa
import soundfile as sf
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class ShipsEarLoader:
    """ShipsEar 데이터셋 로더"""

    def __init__(self, data_dir: str = "data/shipsear"):
        """
        Args:
            data_dir: ShipsEar 데이터 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.classes = [
            'Dredger', 'Tug', 'Cargo', 'Tanker', 'Passengers',
            'Pilot', 'Fishing', 'Sailboat', 'MotorBoat', 'Ocean', 'Natural'
        ]

        # 데이터가 있는지 확인
        self.available = self._check_availability()

    def _check_availability(self) -> bool:
        """데이터셋 존재 여부 확인"""
        if not self.data_dir.exists():
            return False

        # 최소 한 개 클래스 디렉토리가 있는지 확인
        for cls in self.classes:
            cls_dir = self.data_dir / cls
            if cls_dir.exists():
                wav_files = list(cls_dir.glob("*.wav"))
                if len(wav_files) > 0:
                    return True

        return False

    def get_available_classes(self) -> List[str]:
        """사용 가능한 클래스 목록"""
        available = []
        for cls in self.classes:
            cls_dir = self.data_dir / cls
            if cls_dir.exists():
                wav_files = list(cls_dir.glob("*.wav"))
                if len(wav_files) > 0:
                    available.append(cls)
        return available

    def load_class_samples(self,
                          class_name: str,
                          n_samples: int = 10,
                          target_sr: int = 16000,
                          duration: float = 3.0) -> List[np.ndarray]:
        """
        특정 클래스의 샘플 로드

        Args:
            class_name: 클래스 이름
            n_samples: 로드할 샘플 수
            target_sr: 타겟 샘플링 레이트
            duration: 로드할 길이 (초)

        Returns:
            오디오 샘플 리스트
        """
        cls_dir = self.data_dir / class_name

        if not cls_dir.exists():
            raise ValueError(f"Class directory not found: {cls_dir}")

        wav_files = sorted(list(cls_dir.glob("*.wav")))

        if len(wav_files) == 0:
            raise ValueError(f"No WAV files found in {cls_dir}")

        # 샘플 로드
        samples = []
        for wav_file in wav_files[:n_samples]:
            try:
                audio, sr = librosa.load(wav_file, sr=target_sr, duration=duration, mono=True)

                # 길이 맞추기
                target_length = int(target_sr * duration)
                if len(audio) < target_length:
                    audio = np.pad(audio, (0, target_length - len(audio)))
                else:
                    audio = audio[:target_length]

                samples.append(audio)

            except Exception as e:
                print(f"Warning: Failed to load {wav_file}: {e}")
                continue

        return samples

    def get_statistics(self) -> Dict:
        """데이터셋 통계"""
        stats = {
            'available': self.available,
            'total_classes': len(self.classes),
            'available_classes': len(self.get_available_classes()),
            'classes': {}
        }

        for cls in self.get_available_classes():
            cls_dir = self.data_dir / cls
            wav_files = list(cls_dir.glob("*.wav"))
            stats['classes'][cls] = len(wav_files)

        return stats


class DeepShipLoader:
    """DeepShip 데이터셋 로더"""

    def __init__(self, data_dir: str = "data/DeepShip"):
        """
        Args:
            data_dir: DeepShip 데이터 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.classes = ['Cargo', 'Passengership', 'Tanker', 'Tug']
        self.available = self._check_availability()

    def _check_availability(self) -> bool:
        """데이터셋 존재 여부 확인"""
        if not self.data_dir.exists():
            return False

        for cls in self.classes:
            cls_dir = self.data_dir / cls
            if cls_dir.exists():
                wav_files = list(cls_dir.glob("*.wav"))
                if len(wav_files) > 0:
                    return True

        return False

    def get_available_classes(self) -> List[str]:
        """사용 가능한 클래스 목록"""
        available = []
        for cls in self.classes:
            cls_dir = self.data_dir / cls
            if cls_dir.exists():
                wav_files = list(cls_dir.glob("*.wav"))
                if len(wav_files) > 0:
                    available.append(cls)
        return available

    def load_class_samples(self,
                          class_name: str,
                          n_samples: int = 10,
                          target_sr: int = 16000,
                          duration: float = 3.0) -> List[np.ndarray]:
        """
        특정 클래스의 샘플 로드

        Args:
            class_name: 클래스 이름
            n_samples: 로드할 샘플 수
            target_sr: 타겟 샘플링 레이트
            duration: 로드할 길이 (초)

        Returns:
            오디오 샘플 리스트
        """
        cls_dir = self.data_dir / class_name

        if not cls_dir.exists():
            raise ValueError(f"Class directory not found: {cls_dir}")

        wav_files = sorted(list(cls_dir.glob("*.wav")))

        if len(wav_files) == 0:
            raise ValueError(f"No WAV files found in {cls_dir}")

        # 샘플 로드
        samples = []
        for wav_file in wav_files[:n_samples]:
            try:
                audio, sr = librosa.load(wav_file, sr=target_sr, duration=duration, mono=True)

                # 길이 맞추기
                target_length = int(target_sr * duration)
                if len(audio) < target_length:
                    audio = np.pad(audio, (0, target_length - len(audio)))
                else:
                    audio = audio[:target_length]

                samples.append(audio)

            except Exception as e:
                print(f"Warning: Failed to load {wav_file}: {e}")
                continue

        return samples

    def get_statistics(self) -> Dict:
        """데이터셋 통계"""
        stats = {
            'available': self.available,
            'total_classes': len(self.classes),
            'available_classes': len(self.get_available_classes()),
            'classes': {}
        }

        for cls in self.get_available_classes():
            cls_dir = self.data_dir / cls
            wav_files = list(cls_dir.glob("*.wav"))
            stats['classes'][cls] = len(wav_files)

        return stats


def check_datasets() -> Dict:
    """
    사용 가능한 데이터셋 확인

    Returns:
        데이터셋 정보 딕셔너리
    """
    info = {
        'shipsear': {'available': False, 'loader': None},
        'deepship': {'available': False, 'loader': None}
    }

    # ShipsEar 확인
    try:
        shipsear = ShipsEarLoader()
        info['shipsear']['available'] = shipsear.available
        if shipsear.available:
            info['shipsear']['loader'] = shipsear
            info['shipsear']['stats'] = shipsear.get_statistics()
    except Exception as e:
        print(f"ShipsEar check failed: {e}")

    # DeepShip 확인
    try:
        deepship = DeepShipLoader()
        info['deepship']['available'] = deepship.available
        if deepship.available:
            info['deepship']['loader'] = deepship
            info['deepship']['stats'] = deepship.get_statistics()
    except Exception as e:
        print(f"DeepShip check failed: {e}")

    return info


def create_dummy_real_data(output_dir: str = "data/dummy_real",
                          n_samples: int = 5) -> None:
    """
    테스트용 더미 "실제" 데이터 생성
    실제 데이터가 없을 때 프레임워크 테스트용

    Args:
        output_dir: 출력 디렉토리
        n_samples: 생성할 샘플 수
    """
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

    from physics_simulator import ScenarioSynthesizer

    output_path = Path(output_dir)

    # 클래스별 디렉토리 생성
    classes = {
        'Cargo': {'speed': 15.0, 'rpm': 150},
        'Tug': {'speed': 10.0, 'rpm': 200},
        'Tanker': {'speed': 12.0, 'rpm': 120}
    }

    synthesizer = ScenarioSynthesizer(sample_rate=16000, duration=3.0)

    print(f"Creating dummy real data in {output_dir}...")

    for cls_name, params in classes.items():
        cls_dir = output_path / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n_samples):
            # 약간씩 다른 파라미터로 생성
            speed = params['speed'] + np.random.uniform(-2, 2)
            rpm = params['rpm'] + np.random.uniform(-20, 20)

            audio, _ = synthesizer.synthesize_highspeed_vessel(
                speed=speed,
                propeller_rpm=rpm,
                distance=1000.0 + np.random.uniform(-200, 200)
            )

            # 추가 변형 (실제 데이터처럼 보이게)
            audio += np.random.randn(len(audio)) * 0.05  # 노이즈
            audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.8  # 정규화

            # 저장
            filename = cls_dir / f"{cls_name.lower()}_{i:03d}.wav"
            sf.write(filename, audio, 16000)

        print(f"  ✓ {cls_name}: {n_samples} samples")

    print(f"✓ Dummy data created in {output_dir}")


if __name__ == "__main__":
    # 데이터셋 확인
    print("Checking available datasets...")
    print("=" * 60)

    datasets = check_datasets()

    print("\n[ShipsEar]")
    if datasets['shipsear']['available']:
        print("  Status: ✓ Available")
        stats = datasets['shipsear']['stats']
        print(f"  Classes: {stats['available_classes']}/{stats['total_classes']}")
        for cls, count in stats['classes'].items():
            print(f"    - {cls}: {count} files")
    else:
        print("  Status: ✗ Not found")
        print("  Download: https://zenodo.org/records/2559221")

    print("\n[DeepShip]")
    if datasets['deepship']['available']:
        print("  Status: ✓ Available")
        stats = datasets['deepship']['stats']
        print(f"  Classes: {stats['available_classes']}/{stats['total_classes']}")
        for cls, count in stats['classes'].items():
            print(f"    - {cls}: {count} files")
    else:
        print("  Status: ✗ Not found")
        print("  Download: https://ieee-dataport.org/open-access/deepship")

    print("\n" + "=" * 60)

    # 데이터가 없으면 더미 데이터 생성 제안
    if not datasets['shipsear']['available'] and not datasets['deepship']['available']:
        print("\n⚠️  No real datasets found!")
        print("\nOptions:")
        print("  1. Download ShipsEar or DeepShip (see docs/DATA_DOWNLOAD_GUIDE.md)")
        print("  2. Create dummy data for testing:")
        print("     python src/real_data_loader.py --create-dummy")
