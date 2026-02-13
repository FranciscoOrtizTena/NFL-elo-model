from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    repo_root: Path

    @property
    def data_dir(self) -> Path:
        return self.repo_root / "data"

    @property
    def schedule_features_path(self) -> Path:
        return self.data_dir / "schedule_features_2016_2025.parquet"

    @property
    def model_table_path(self) -> Path:
        return self.data_dir / "final_first_model.parquet"


def get_paths(repo_root: str | Path | None = None) -> ProjectPaths:
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    return ProjectPaths(repo_root=root.resolve())

