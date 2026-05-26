# Data Sources

This repository does not redistribute raw datasets. Download each dataset from the original provider, follow the provider's license and terms, and place the extracted files under a common dataset root.

## Expected root

```text
<dataset-root>/
├── CHECKED/
├── CSDC-Rumor/
└── PHEME/
```

Pass this root through `--dataset-root` or set:

```bash
export RUMOR_SIM_DATASET_ROOT=/path/to/datasets
```

## CHECKED

Source:

- Dataset repository: <https://github.com/cyang03/CHECKED>
- Paper DOI: <https://doi.org/10.1007/s13278-021-00766-8>

Expected local structure:

```text
CHECKED/
└── dataset/
    ├── fake_news/
    │   └── *.json
    └── real_news/
        └── *.json
```

The parser reads verified COVID-19 Weibo microblogs, comments, reposts, metadata, and fact-check analysis fields where available.

## CSDC-Rumor

Source:

- Dataset page: <https://covid19.thunlp.org/archives/5/>
- Download URL listed by the provider: <https://data.thunlp.org/covid19/rumor.zip>
- Related paper DOI: <https://doi.org/10.1177/00027642211003153>

Expected local structure:

```text
CSDC-Rumor/
├── fact.json
├── rumor_weibo/
│   └── *.json
└── rumor_forward_comment/
    └── *.json
```

The parser reads Weibo rumor posts, comment/forward records, and Tencent/DXY fact-check records from `fact.json`.

## PHEME

Source:

- Dataset page: <https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078>
- DOI: <https://doi.org/10.6084/m9.figshare.6392078.v1>

Expected local structure:

```text
PHEME/
└── all-rnr-annotated-threads/
    └── <event>/
        ├── rumours/
        │   └── <thread-id>/
        │       ├── annotation.json
        │       ├── structure.json
        │       ├── source-tweets/
        │       │   └── *.json
        │       └── reactions/
        │           └── *.json
        └── non-rumours/
            └── <thread-id>/
                └── ...
```

The parser reads source tweets, reaction tweets, conversation structure files, and veracity annotations.

## Output data

Do not write generated outputs into the dataset root. Use a separate output directory:

```bash
export RUMOR_SIM_OUTPUT_ROOT=/path/to/outputs
```

Generated artifacts include normalized canonical files, audit logs, feature matrices, text-model caches, triage metrics, figures, and summary tables.
