"""
통합 데이터셋 검증 시스템
DeepShip + ShipsEar 통합 활용
"""

import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class IntegratedDatasetLoader:
    """DeepShip과 ShipsEar 통합 로더"""

    def __init__(self):
        self.deepship_dir = Path("data/DeepShip")
        self.shipsear_dir = Path("data/ShipsEar")

        self.deepship_available = self._check_deepship()
        self.shipsear_available = self._check_shipsear()

        # ShipsEar 메타데이터 로드
        self.shipsear_metadata = self._load_shipsear_metadata()

    def _check_deepship(self) -> bool:
        """DeepShip 데이터 확인"""
        if not self.deepship_dir.exists():
            return False

        for cls in ['Cargo', 'Passengership', 'Tanker', 'Tug']:
            cls_dir = self.deepship_dir / cls
            if cls_dir.exists() and len(list(cls_dir.glob("*.wav"))) > 0:
                return True
        return False

    def _check_shipsear(self) -> bool:
        """ShipsEar 데이터 확인"""
        if not self.shipsear_dir.exists():
            return False

        for cls in ['Class_A', 'Class_B', 'Class_C', 'Class_D', 'Class_E']:
            cls_dir = self.shipsear_dir / cls
            if cls_dir.exists() and len(list(cls_dir.glob("*.wav"))) > 0:
                return True
        return False

    def _load_shipsear_metadata(self) -> Optional[pd.DataFrame]:
        """ShipsEar 메타데이터 로드"""
        csv_path = self.shipsear_dir / "ShipsEar.csv"
        if csv_path.exists():
            try:
                return pd.read_csv(csv_path)
            except:
                return None
        return None

    def get_shipsear_class_mapping(self) -> Dict[str, str]:
        """
        ShipsEar 클래스 매핑

        Returns:
            클래스 디렉터리명 -> 선박 유형 매핑
        """
        # ShipsEar 논문 기준
        return {
            'Class_A': 'Tugboat',          # 예인선
            'Class_B': 'Motorboat/Yacht',  # 모터보트/요트
            'Class_C': 'Passengers',       # 여객선
            'Class_D': 'Cargo/Tanker',     # 화물선/유조선
            'Class_E': 'Fishing'           # 어선
        }

    def load_deepship_samples(self,
                             vessel_class: str,
                             n_samples: int = 5,
                             target_sr: int = 16000,
                             duration: float = 3.0,
                             offset: float = 180.0) -> List[np.ndarray]:
        """
        DeepShip 샘플 로드 (선박 통과 구간)

        Args:
            vessel_class: 'Cargo', 'Passengership', 'Tanker', 'Tug'
            n_samples: 로드할 샘플 수
            target_sr: 샘플링 레이트
            duration: 길이 (초)
            offset: 시작 시간 (초) - 선박 통과 구간

        Returns:
            오디오 샘플 리스트
        """
        if not self.deepship_available:
            raise ValueError("DeepShip dataset not available")

        cls_dir = self.deepship_dir / vessel_class
        wav_files = sorted(list(cls_dir.glob("*.wav")))[:n_samples]

        samples = []
        for wav_file in wav_files:
            try:
                audio, sr = librosa.load(wav_file, sr=target_sr,
                                        offset=offset, duration=duration)

                # 전처리: DC 제거 및 정규화
                audio = audio - np.mean(audio)
                max_val = np.max(np.abs(audio))
                if max_val > 1e-6:
                    audio = audio / max_val

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

    def load_shipsear_samples(self,
                             class_dir: str,
                             n_samples: int = 5,
                             target_sr: int = 16000,
                             duration: float = 3.0) -> List[np.ndarray]:
        """
        ShipsEar 샘플 로드

        Args:
            class_dir: 'Class_A', 'Class_B', 'Class_C', 'Class_D', 'Class_E'
            n_samples: 로드할 샘플 수
            target_sr: 샘플링 레이트
            duration: 길이 (초)

        Returns:
            오디오 샘플 리스트
        """
        if not self.shipsear_available:
            raise ValueError("ShipsEar dataset not available")

        cls_dir = self.shipsear_dir / class_dir
        wav_files = sorted(list(cls_dir.glob("*.wav")))[:n_samples]

        samples = []
        for wav_file in wav_files:
            try:
                # ShipsEar는 52.7kHz, 짧은 길이 (1-2초)
                audio, sr = librosa.load(wav_file, sr=target_sr, duration=duration)

                # 전처리
                audio = audio - np.mean(audio)
                max_val = np.max(np.abs(audio))
                if max_val > 1e-6:
                    audio = audio / max_val

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

    def get_combined_vessel_samples(self,
                                   vessel_type: str,
                                   n_samples_per_dataset: int = 3,
                                   target_sr: int = 16000,
                                   duration: float = 3.0) -> Dict[str, List[np.ndarray]]:
        """
        두 데이터셋에서 동일 선박 유형 샘플 로드

        Args:
            vessel_type: 'Cargo', 'Tug', 'Passengers', 'Motorboat'
            n_samples_per_dataset: 데이터셋당 샘플 수
            target_sr: 샘플링 레이트
            duration: 길이

        Returns:
            {'deepship': [...], 'shipsear': [...]} 형태의 딕셔너리
        """
        result = {'deepship': [], 'shipsear': []}

        # DeepShip 매핑
        deepship_mapping = {
            'Cargo': 'Cargo',
            'Tug': 'Tug',
            'Passengers': 'Passengership',
            'Tanker': 'Tanker'
        }

        # ShipsEar 매핑
        shipsear_mapping = {
            'Tug': 'Class_A',
            'Motorboat': 'Class_B',
            'Passengers': 'Class_C',
            'Cargo': 'Class_D',
            'Fishing': 'Class_E'
        }

        # DeepShip 로드
        if vessel_type in deepship_mapping and self.deepship_available:
            try:
                result['deepship'] = self.load_deepship_samples(
                    deepship_mapping[vessel_type],
                    n_samples=n_samples_per_dataset,
                    target_sr=target_sr,
                    duration=duration
                )
            except Exception as e:
                print(f"DeepShip loading failed: {e}")

        # ShipsEar 로드
        if vessel_type in shipsear_mapping and self.shipsear_available:
            try:
                result['shipsear'] = self.load_shipsear_samples(
                    shipsear_mapping[vessel_type],
                    n_samples=n_samples_per_dataset,
                    target_sr=target_sr,
                    duration=duration
                )
            except Exception as e:
                print(f"ShipsEar loading failed: {e}")

        return result

    def get_dataset_statistics(self) -> Dict:
        """데이터셋 통계"""
        stats = {
            'deepship': {
                'available': self.deepship_available,
                'classes': {}
            },
            'shipsear': {
                'available': self.shipsear_available,
                'classes': {}
            }
        }

        # DeepShip 통계
        if self.deepship_available:
            for cls in ['Cargo', 'Passengership', 'Tanker', 'Tug']:
                cls_dir = self.deepship_dir / cls
                if cls_dir.exists():
                    wav_files = list(cls_dir.glob("*.wav"))
                    stats['deepship']['classes'][cls] = len(wav_files)

        # ShipsEar 통계
        if self.shipsear_available:
            class_mapping = self.get_shipsear_class_mapping()
            for cls_dir, vessel_type in class_mapping.items():
                cls_path = self.shipsear_dir / cls_dir
                if cls_path.exists():
                    wav_files = list(cls_path.glob("*.wav"))
                    stats['shipsear']['classes'][f"{cls_dir} ({vessel_type})"] = len(wav_files)

        return stats


def print_dataset_summary():
    """데이터셋 요약 출력"""
    loader = IntegratedDatasetLoader()
    stats = loader.get_dataset_statistics()

    print("="*80)
    print("Integrated Dataset Summary")
    print("="*80)
    print()

    print("[DeepShip]")
    if stats['deepship']['available']:
        print("  Status: ✓ Available")
        print("  Classes:")
        for cls, count in stats['deepship']['classes'].items():
            print(f"    - {cls}: {count} files")
    else:
        print("  Status: ✗ Not found")

    print()
    print("[ShipsEar]")
    if stats['shipsear']['available']:
        print("  Status: ✓ Available")
        print("  Classes:")
        for cls, count in stats['shipsear']['classes'].items():
            print(f"    - {cls}: {count} files")
    else:
        print("  Status: ✗ Not found")

    print()
    print("="*80)
    print()

    if stats['deepship']['available'] and stats['shipsear']['available']:
        print("✓ Both datasets available for comprehensive validation!")
    elif stats['deepship']['available'] or stats['shipsear']['available']:
        print("~ One dataset available - partial validation possible")
    else:
        print("✗ No datasets found")

    return loader, stats


if __name__ == "__main__":
    print_dataset_summary()
