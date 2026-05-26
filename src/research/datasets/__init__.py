from __future__ import annotations

from .checked import parse_checked
from .csdc_rumor import parse_csdc_rumor
from .pheme import parse_pheme


PARSERS = {
    "CHECKED": parse_checked,
    "CSDC-Rumor": parse_csdc_rumor,
    "PHEME": parse_pheme,
}


def parse_dataset(*, dataset_name: str, **kwargs):
    try:
        parser = PARSERS[dataset_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {dataset_name}") from exc
    return parser(**kwargs)
