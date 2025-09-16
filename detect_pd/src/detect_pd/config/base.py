"""Base configuration utilities for the DETECT-PD project."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Type, TypeVar

import yaml
from pydantic import BaseModel, Field, root_validator

_T = TypeVar("_T", bound="BaseConfig")


class BaseConfig(BaseModel):
    """Base configuration class for all pipeline configs.

    Provides common settings such as random seed and logging level, and helper
    methods for loading configuration objects from YAML files. Environment
    variable based overrides can be added in later iterations if needed.
    """

    random_seed: int = Field(42, description="Random seed used across pipeline components.")
    logging_level: str = Field(
        "INFO", description="Python logging level for the component using this config."
    )

    class Config:
        extra = "forbid"
        allow_mutation = False

    @root_validator(pre=True)
    def _normalize_keys(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Normalise incoming keys to snake_case.

        YAML config files may contain hyphenated keys; convert them to snake_case
        so that Pydantic field validation works as expected.
        """

        normalized: Dict[str, Any] = {}
        for key, value in values.items():
            snake_key = key.replace("-", "_")
            normalized[snake_key] = value
        return normalized

    @classmethod
    def from_yaml(cls: Type[_T], path: Path | str) -> _T:
        """Load a configuration object from a YAML file."""

        path = Path(path)
        with path.open("r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp) or {}
        return cls.parse_obj(data)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable representation of the config."""

        return self.dict()


def load_configs_from_directory(directory: Path | str, registry: Dict[str, Type[BaseConfig]]) -> Dict[str, BaseConfig]:
    """Load multiple configuration objects from a directory of YAML files.

    Args:
        directory: Directory containing configuration YAML files.
        registry: Mapping of stem names to configuration classes. Each file's
            stem (without extension) must exist in the registry.

    Returns:
        Dict mapping the same keys as the registry to instantiated config objects.
    """

    directory = Path(directory)
    configs: Dict[str, BaseConfig] = {}
    for name, config_cls in registry.items():
        file_path = directory / f"{name}.yaml"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing configuration file: {file_path}")
        configs[name] = config_cls.from_yaml(file_path)
    return configs
