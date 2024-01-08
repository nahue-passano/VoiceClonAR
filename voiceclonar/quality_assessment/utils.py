from typing import Dict, List
from pathlib import Path

from natsort import natsorted


def sort_dict_keys(dictionary: Dict) -> Dict:
    """
    Sort a dictionary by its keys using natural sort order.

    Parameters
    ----------
    dictionary : Dict
        The dictionary to be sorted by keys.

    Returns
    -------
    Dict
        A new dictionary where the keys are sorted in natural order.
    """
    sorted_keys = natsorted(dictionary.keys())
    return {k: dictionary[k] for k in sorted_keys}


def get_keys_matching_str(dictionary: Dict, str_pattern: List[str]):
    """
    Filter the dictionary to include only those items whose keys match the given string pattern.

    Parameters
    ----------
    dictionary : Dict
        The dictionary to be filtered.
    str_pattern : List[str] or str
        A list of string patterns to match against the keys. If a single string is provided,
        it is converted into a list containing that string.

    Returns
    -------
    Dict
        A filtered dictionary containing only the key-value pairs where the key matches
        the given string pattern(s).
    """
    if isinstance(str_pattern, str):
        str_pattern = [str_pattern]
    filtered_dict = {
        name: path
        for name, path in dictionary.items()
        if all(id in name for id in str_pattern)
    }
    return filtered_dict


def get_audio_reference(audio_name: str, references_dict: Dict) -> Path:
    """
    Get the path reference for a given audio name from a dictionary of references.

    Parameters
    ----------
    audio_name : str
        The name of the audio file for which the reference is required.
    references_dict : Dict
        A dictionary containing audio references, where keys are reference names.

    Returns
    -------
    Path
        The path corresponding to the audio name, if found in the references dictionary.
    """
    for reference in references_dict.keys():
        id = "_".join(reference.split("_")[:-1])
        if id in audio_name:
            return references_dict[reference]
