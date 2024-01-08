from typing import Union
from pathlib import Path
from abc import ABC, abstractmethod

from speechbrain.pretrained import SpeakerRecognition
import torch
import torchaudio

from voiceclonar.utils import load_config

CFG_PATH = Path("voiceclonar/quality_assessment/cfg.yaml")


class FeatureExtractor(ABC):
    def __init__(self, cfg: Path = CFG_PATH):
        """
        Initialize the FeatureExtractor.

        Parameters
        ----------
        cfg : Path, optional
            The configuration file path, by default CFG_PATH.
        """
        self.cfg = load_config(cfg)

    @abstractmethod
    def process_audio(self, audio_path: Union[Path, str]) -> torch.tensor:
        """
        Extract features from the given audio file.

        This method must be implemented by any subclass of FeatureExtractor.

        Parameters
        ----------
        audio_path : Union[Path, str]
            Path to the audio file from which to extract features.

        Returns
        -------
        torch.tensor
            The extracted features as a torch tensor.
        """
        pass


class XVectorExtractor(FeatureExtractor):
    def process_audio(self, audio_path: Union[Path, str]) -> torch.tensor:
        """
        Extract x-vector features from the given audio file.

        Parameters
        ----------
        audio_path : Union[Path, str]
            Path to the audio file from which to extract x-vector features.

        Returns
        -------
        torch.tensor
            The extracted x-vector features as a torch tensor.
        """
        xvector_model = SpeakerRecognition.from_hparams(
            source=self.cfg.features.xvector.source,
            savedir=f"weights/{self.cfg.features.xvector.source}",
        )
        signal, _ = torchaudio.load(audio_path)
        return xvector_model.encode_batch(signal)
