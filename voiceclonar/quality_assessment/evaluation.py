from typing import Union, Dict, List, Tuple
from pathlib import Path

import torch
import torchaudio
import numpy as np
import pandas as pd
from scipy.spatial import distance
from torchaudio.pipelines import SQUIM_OBJECTIVE

from voiceclonar.quality_assessment.feature_extraction import (
    XVectorExtractor,
    TitaNetEmbeddingExtractor,
)
from voiceclonar.quality_assessment.nisqa.NISQA_model import nisqaModel
from voiceclonar.quality_assessment import utils as qa_utils
from voiceclonar.utils import load_config

CFG_PATH = Path("voiceclonar/quality_assessment/cfg.yaml")


class SyntheticSpeechQA:
    def __init__(self, cfg: Path = CFG_PATH):
        self.cfg = load_config(cfg)
        self.feature_extractor = TitaNetEmbeddingExtractor()

    def set_hparams(self, hparams: Dict):
        for param, value in hparams.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                print(
                    f"[WARNING] Parameter {param} is not a class attribute. Skipping."
                )

    def get_hparams(self) -> None:
        return self.__dict__

    def calculate_similarity(
        self, embed_1: torch.tensor, embed_2: torch.tensor
    ) -> float:
        if embed_1.ndim > 1:
            embed_1 = embed_1.squeeze()

        if embed_2.ndim > 1:
            embed_2 = embed_2.squeeze()

        similarity_score = torch.dot(embed_1, embed_2) / (
            (torch.dot(embed_1, embed_1) * torch.dot(embed_2, embed_2)) ** 0.5
        )
        similarity_score = (similarity_score + 1) / 2

        return similarity_score.item()

    def calculate_squim_objective_params(self, audio_path: Union[Path, str]) -> Tuple:
        audio_array, sr = torchaudio.load(audio_path)

        if sr != SQUIM_OBJECTIVE.sample_rate:
            audio_array = torchaudio.functional.resample(
                audio_array, sr, SQUIM_OBJECTIVE.sample_rate
            )

        squim_model = SQUIM_OBJECTIVE.get_model()
        stoi, pesq, sisdr = squim_model(audio_array)
        return stoi.item(), pesq.item(), sisdr.item()

    def calculate_nisqa(self, audio_path: Union[Path, str], model_name: "str"):
        if isinstance(audio_path, Path):
            audio_path = str(audio_path)

        model_args = self.cfg.metrics.nisqa.__dict__
        model_args["pretrained_model"] = f"{model_args['weights']}/{model_name}.tar"
        model_args["deg"] = audio_path
        nisqa_results = nisqaModel(model_args).predict()
        return nisqa_results.iloc[0, 1]

    def _evaluate_batch(
        self, audio_paths: Dict, reference_paths: Dict = None
    ) -> pd.DataFrame:
        batch_evaluation = []
        for audio_i, path_i in audio_paths.items():
            # Similarity
            reference_id = qa_utils.get_audio_reference(audio_i, reference_paths)
            if isinstance(reference_id, Path):
                # Extracting embed
                audio_i_embed = self.feature_extractor.process_audio(path_i)
                reference_embed = self.feature_extractor.process_audio(reference_id)

                similarity = self.calculate_similarity(audio_i_embed, reference_embed)
            else:
                similarity = np.nan

            # SQUIM objective parameters
            stoi, pesq, sisdr = self.calculate_squim_objective_params(path_i)

            # NISQA
            nisqa_mos = self.calculate_nisqa(path_i, "nisqa_mos_only")
            nisqa_natmos = self.calculate_nisqa(path_i, "nisqa_tts")

            batch_evaluation.append(
                {
                    "audio_name": audio_i,
                    "STOI": stoi,
                    "PESQ": pesq,
                    "SISDR": sisdr,
                    "MOS (NISQA)": nisqa_mos,
                    "NatMOS (NISQA)": nisqa_natmos,
                    "Similarity": similarity,
                }
            )

        return pd.DataFrame(batch_evaluation)

    def evaluate_folder(
        self,
        audios_folder: Union[Path, str],
        reference_suffix: str = None,
        selected_speakers: List[str] = None,
    ) -> pd.DataFrame:
        if isinstance(audios_folder, str):
            audios_folder = Path(audios_folder)

        audio_paths = qa_utils.sort_dict_keys(
            {audio_i.stem: audio_i for audio_i in audios_folder.iterdir()}
        )

        if selected_speakers:
            audio_paths = qa_utils.get_keys_matching_str(audio_paths, selected_speakers)

        if reference_suffix:
            reference_paths = qa_utils.get_keys_matching_str(
                audio_paths, [reference_suffix]
            )
            audio_paths = {
                name: path
                for name, path in audio_paths.items()
                if name not in reference_paths.keys()
            }
        else:
            reference_paths = None

        return self._evaluate_batch(audio_paths, reference_paths)

    def evaluate_dict(self, audios_dict: Dict, reference_suffix: str = None):
        audio_paths = qa_utils.sort_dict_keys(audios_dict)

        if reference_suffix:
            reference_paths = qa_utils.get_keys_matching_str(
                audio_paths, reference_suffix
            )
            audio_paths = {
                name: path
                for name, path in audio_paths.items()
                if name not in reference_paths.keys()
            }
        else:
            reference_paths = None

        return self._evaluate_batch(audio_paths, reference_paths)
