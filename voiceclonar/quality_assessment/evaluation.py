from typing import Union, Dict, List
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from scipy.spatial import distance

from voiceclonar.quality_assessment.feature_extraction import XVectorExtractor
from voiceclonar.quality_assessment import utils as qa_utils


class SyntheticSpeechQA:
    def __init__(self):
        self.feature_extractor = XVectorExtractor()
        self.similarity_by = "cosine"

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

    def measure_similarity(
        self, embed_1: torch.tensor, embed_2: torch.tensor, by: str = None
    ) -> float:
        if not by:
            by = self.similarity_by

        if embed_1.ndim > 1:
            embed_1 = embed_1.squeeze()

        if embed_2.ndim > 1:
            embed_2 = embed_2.squeeze()

        if hasattr(distance, by):
            distance_func = getattr(distance, by)
            return 1 - distance_func(embed_1, embed_2)
        else:
            raise ValueError(f"Unknown method for calculate distance: {by}")

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

                similarity = self.measure_similarity(audio_i_embed, reference_embed)
            else:
                similarity = np.nan

            batch_evaluation.append({"audio_name": audio_i, "similarity": similarity})

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
