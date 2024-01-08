"""Utilities module for VoiceClonAR"""
from pathlib import Path
from typing import Union, List

import yaml


class AttributeDict:
    """
    AttributeDict class provides a dictionary-like interface with attribute-style access for nested dictionaries.

    An AttributeDict allows you to access dictionary values using dot notation, similar to object attributes.
    It recursively converts nested dictionaries into AttributeDict instances, providing a more convenient way to
    work with complex nested data structures.

    Parameters
    ----------
    dictionary : dict
        The dictionary to initialize the AttributeDict with.

    Example
    -------
    dict_data = {'user': {'name': 'John', 'age': 30}, 'address': {'city': 'New York'}}
    attr_dict = AttributeDict(dict_data)

    # Access values using dot notation
    name = attr_dict.user.name  # Equivalent to dict_data['user']['name']

    # Nested dictionaries are also converted to AttributeDict instances
    city = attr_dict.address.city  # Equivalent to dict_data['address']['city']

    Methods
    -------
    __init__(self, dictionary: dict):
        Initializes an AttributeDict instance with a given dictionary. Nested dictionaries are recursively
        converted into AttributeDict instances.

    merge(self, other, keys_to_merge: Union[str, List[str]] = None):
        Merges the specified attributes of another AttributeDict into this one.

        Parameters
        ----------
        other : AttributeDict
            Another AttributeDict instance to merge with.
        keys_to_merge : str or list of str, optional
            A list of attribute names to merge. If None, all attributes are merged. Default is None.

        Returns
        -------
        AttributeDict
            The merged AttributeDict object.

        Raises
        ------
        ValueError
            If the `other` object is not an AttributeDict.
    """

    def __init__(self, dictionary: dict):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, AttributeDict(value))
            else:
                setattr(self, key, value)

    def merge(self, other, keys_to_merge: Union[str, List[str]] = None):
        """Merge the specified attributes of another AttributeDict into this one.

        Parameters
        ----------
        other : AttributeDict
            Another AttributeDict instance to merge with.
        keys_to_merge : str or list of str, optional
            A list of attribute names to merge. If None, all attributes are merged. Default is None.

        Raises
        ------
        ValueError
            If the `other` object is not an AttributeDict.
        """
        if not isinstance(other, AttributeDict):
            raise ValueError("Can only merge with another AttributeDict")

        if keys_to_merge is None:
            # If keys_to_merge is not specified, merge all attributes
            keys_to_merge = other.__dict__.keys()

        if isinstance(keys_to_merge, str):
            keys_to_merge = [keys_to_merge]

        for key in keys_to_merge:
            value = other.__dict__.get(key)
            if value is not None:
                if isinstance(value, AttributeDict):
                    # If the value is another AttributeDict, merge it recursively
                    if hasattr(self, key) and isinstance(
                        getattr(self, key), AttributeDict
                    ):
                        getattr(self, key).merge(value, keys_to_merge=keys_to_merge)
                    else:
                        setattr(self, key, value)
                else:
                    # Otherwise, simply set the attribute
                    setattr(self, key, value)

        return self


def load_config(config_path: Union[str, Path]) -> AttributeDict:
    """
    Load a YAML configuration file and convert it into an AttributeDict.

    This function reads a YAML configuration file from the specified path and converts it into an AttributeDict
    for easy access to configuration parameters using dot notation.

    Parameters
    ----------
    config_path : Union[str, Path]
        The path to the YAML configuration file to load.

    Returns
    -------
    AttributeDict
        An AttributeDict instance representing the configuration data from the YAML file.

    Raises
    ------
    AssertionError
        If the `config_path` is not an instance of string or pathlib.Path.

    Example
    -------
    Given a YAML configuration file 'config.yml' with the following content:
    ```
    user:
        name: John
        age: 30
    server:
        host: localhost
        port: 8080
    ```

    You can load the configuration as follows:
    ```
    config = load_config('config.yml')

    # Access configuration parameters using dot notation
    username = config.user.name  # Equivalent to 'John'
    port_number = config.server.port  # Equivalent to 8080
    ```
    """
    assert isinstance(
        config_path, (str, Path)
    ), f"YAML path {config_path} must be an instance of string or pathlib.Path"

    with open(config_path, "r") as yaml_file:
        yaml_dict = yaml.safe_load(yaml_file)

    return AttributeDict(yaml_dict)
