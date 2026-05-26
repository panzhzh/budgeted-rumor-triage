from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PREFERRED_DATASET_ROOT = Path("/scr/user/ipanzhzh/datasets/rumor-intervention-sim")
DEFAULT_OUTPUT_ROOT = Path("/scr/user/ipanzhzh/pythoncode/rumor-intervention-sim/outputs")
_FALLBACK_DATASET_ROOT = PROJECT_ROOT / "data"
DEFAULT_DATASET_ROOT = _PREFERRED_DATASET_ROOT if _PREFERRED_DATASET_ROOT.exists() else _FALLBACK_DATASET_ROOT
DEFAULT_CHECKPOINT_ROOT = DEFAULT_OUTPUT_ROOT / "checkpoints"
DEFAULT_RESULT_ROOT = DEFAULT_OUTPUT_ROOT / "results"
DEFAULT_TEXT_FEATURE_ROOT = DEFAULT_OUTPUT_ROOT / "text_features"
DEFAULT_DATASET_TIMEZONES = {
    "CHECKED": "Asia/Shanghai",
    "CSDC-Rumor": "Asia/Shanghai",
    "PHEME": "UTC",
}

DATASET_NAMES = ("CHECKED", "CSDC-Rumor", "PHEME")
DATASET_LANGUAGES = {
    "CHECKED": "zh",
    "CSDC-Rumor": "zh",
    "PHEME": "en",
}
DEFAULT_WINDOWS_HOURS = (1, 6, 24, 48)
DEFAULT_RESPONSE_TAXONOMY = ("deny", "query", "evidence", "emotion", "other")
DEFAULT_SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_TRANSFORMER_MODEL = "joeddav/xlm-roberta-large-xnli"
DEFAULT_TEXT_MODEL_BATCH_SIZE = 32
DEFAULT_TEXT_MODEL_LEVEL = "light"
IGNORED_NAMES = frozenset({".DS_Store", "__MACOSX"})
IGNORED_PREFIXES = ("._",)
STAGES = ("normalize", "audit", "graph", "features", "text_models", "triage", "reporting")


@dataclass(frozen=True)
class PipelineConfig:
    project_root: Path = PROJECT_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    dataset_root: Path = DEFAULT_DATASET_ROOT
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT
    result_root: Path = DEFAULT_RESULT_ROOT
    text_feature_root: Path = DEFAULT_TEXT_FEATURE_ROOT
    windows_hours: tuple[int, ...] = DEFAULT_WINDOWS_HOURS
    response_taxonomy: tuple[str, ...] = DEFAULT_RESPONSE_TAXONOMY
    sentence_transformer_model: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL
    transformer_model: str = DEFAULT_TRANSFORMER_MODEL
    text_model_batch_size: int = DEFAULT_TEXT_MODEL_BATCH_SIZE
    text_model_level: str = DEFAULT_TEXT_MODEL_LEVEL
    dataset_languages: dict[str, str] = field(default_factory=lambda: dict(DATASET_LANGUAGES))
    dataset_timezones: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_DATASET_TIMEZONES))
    ignored_names: frozenset[str] = IGNORED_NAMES
    ignored_prefixes: tuple[str, ...] = IGNORED_PREFIXES

    def dataset_path(self, dataset_name: str) -> Path:
        return self.dataset_root / dataset_name

    def dataset_timezone(self, dataset_name: str) -> str:
        return self.dataset_timezones.get(dataset_name, "UTC")

    def checkpoint_stage_path(self, stage: str, dataset_name: str | None = None) -> Path:
        base = self.checkpoint_root / stage
        return base if dataset_name is None else base / dataset_name

    def text_feature_path(self, dataset_name: str, filename: str | None = None) -> Path:
        base = self.text_feature_root / dataset_name
        return base if filename is None else base / filename

    def result_stage_path(self, stage: str, dataset_name: str | None = None) -> Path:
        base = self.result_root / stage
        return base if dataset_name is None else base / dataset_name

    def report_path(self, dataset_name: str, filename: str) -> Path:
        return self.result_stage_path("reports", dataset_name) / filename

    def checkpoint_path(self, dataset_name: str, filename: str) -> Path:
        return self.checkpoint_stage_path("normalized", dataset_name) / filename

    def feature_path(self, dataset_name: str, filename: str) -> Path:
        return self.result_stage_path("features", dataset_name) / filename

    def log_path(self, dataset_name: str, filename: str) -> Path:
        return self.checkpoint_stage_path("logs", dataset_name) / filename

    def status_path(self, dataset_name: str, filename: str) -> Path:
        return self.checkpoint_stage_path("status", dataset_name) / filename


def build_config(
    dataset_root: str | os.PathLike[str] | None = None,
    output_root: str | os.PathLike[str] | None = None,
    checkpoint_root: str | os.PathLike[str] | None = None,
    result_root: str | os.PathLike[str] | None = None,
    text_feature_root: str | os.PathLike[str] | None = None,
    text_model_level: str | None = None,
) -> PipelineConfig:
    resolved_output_root = Path(
        output_root or os.getenv("RUMOR_SIM_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    )
    resolved_dataset_root = Path(
        dataset_root or os.getenv("RUMOR_SIM_DATASET_ROOT", DEFAULT_DATASET_ROOT)
    )
    resolved_checkpoint_root = Path(
        checkpoint_root
        or os.getenv("RUMOR_SIM_CHECKPOINT_ROOT")
        or resolved_output_root / "checkpoints"
    )
    resolved_result_root = Path(
        result_root
        or os.getenv("RUMOR_SIM_RESULT_ROOT")
        or resolved_output_root / "results"
    )
    resolved_text_feature_root = Path(
        text_feature_root
        or os.getenv("RUMOR_SIM_TEXT_FEATURE_ROOT")
        or resolved_output_root / "text_features"
    )
    _validate_output_roots(
        dataset_root=resolved_dataset_root,
        output_roots={
            "checkpoint_root": resolved_checkpoint_root,
            "result_root": resolved_result_root,
            "text_feature_root": resolved_text_feature_root,
        },
    )
    return PipelineConfig(
        output_root=resolved_output_root,
        dataset_root=resolved_dataset_root,
        checkpoint_root=resolved_checkpoint_root,
        result_root=resolved_result_root,
        text_feature_root=resolved_text_feature_root,
        text_model_level=text_model_level or os.getenv("RUMOR_SIM_TEXT_MODEL_LEVEL", DEFAULT_TEXT_MODEL_LEVEL),
    )


def _validate_output_roots(*, dataset_root: Path, output_roots: dict[str, Path]) -> None:
    resolved_dataset_root = dataset_root.expanduser().resolve(strict=False)
    for name, root in output_roots.items():
        resolved_root = root.expanduser().resolve(strict=False)
        if resolved_root == resolved_dataset_root or _is_relative_to(resolved_root, resolved_dataset_root):
            raise ValueError(
                f"{name}={resolved_root} is inside dataset_root={resolved_dataset_root}; "
                "write outputs under RUMOR_SIM_OUTPUT_ROOT instead."
            )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
