"""Base configuration utilities for the DETECT-PD project."""

from pathlib import Path
from typing import Any, Dict, Type, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

_T = TypeVar("_T", bound="BaseConfig")


class BaseConfig(BaseModel):
    """Base configuration class for all pipeline configs."""

    random_seed: int = Field(42, description="Random seed used across pipeline components.")
    logging_level: str = Field(
        "INFO", description="Python logging level for the component using this config."
    )

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        protected_namespaces=(),
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_keys(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Normalise incoming keys to snake_case for YAML compatibility."""

        normalized: Dict[str, Any] = {}
        for key, value in (values or {}).items():
            snake_key = key.replace("-", "_")
            normalized[snake_key] = value
        return normalized

    @classmethod
    def from_yaml(cls: Type[_T], path: Path | str) -> _T:
        """Load a configuration object from a YAML file."""

        path = Path(path)
        with path.open("r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp) or {}
        return cls.model_validate(data)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable representation of the config."""

        return self.model_dump()


def load_configs_from_directory(directory: Path | str, registry: Dict[str, Type[BaseConfig]]) -> Dict[str, BaseConfig]:
    """Load multiple configuration objects from a directory of YAML files."""

    directory = Path(directory)
    configs: Dict[str, BaseConfig] = {}
    for name, config_cls in registry.items():
        file_path = directory / f"{name}.yaml"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing configuration file: {file_path}")
        configs[name] = config_cls.from_yaml(file_path)
    return configs
