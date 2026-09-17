# Experiment configuration

Version: `v1.1.0`.

The fixed-age target is recorded remaining growth, `max(R - r(t), 0)`. Calendar-time replay counts reactions strictly after actual review. Observation windows are 1, 6 and 24 hours.

## Features and text models

- [Feature families](../../reproduction/v1.1.0/feature_provenance.json): raw inputs, construction, observation cutoffs and fitting scopes, with source-function references.
- [MiniLM configuration](../../reproduction/v1.1.0/minilm.json): pretrained revision, 128-token input limit, 384-dimensional mean pooling, normalization and text assembly.
- [Text rules](../../src/research/text_rules.py): language-specific stance and correction cues. `source_stance_type_deny` indicates that the source-text stance rule returns `deny`.
- [Text descriptors](../../src/research/text_models.py): fixed prototypes and post embeddings. The experiment configuration is `light`.
- [Feature assembly and ranking](../../src/research/triage.py): window-specific columns, training-side categorical layouts, model fitting and budget metrics.

For each outer split, K-means is fitted on embeddings from the outer training threads. Source-post and six-hour thread-text embeddings are clustered separately with up to eight clusters, seed 42 and `n_init=auto`. Training and held-out topic labels and centroid distances are then obtained from those training centroids.

XLM-R regression uses `xlm-roberta-base` revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`, a 6000-character cap, 192 tokens, batch size 16, AdamW learning rate 2e-5, weight decay 0.01 and gradient clipping 1. Validation holds out 20% of outer training threads with seed plus 1000. Patience is three and the maximum is 20 epochs. The minimum validation observation period is three epochs, or ten for CSDC-Rumor and CHECKED rumor-only. The best validation epoch determines final training duration after reinitialization. Training uses BF16 autocast; validation and final scoring use FP32 with TF32 disabled. Split seeds are 11, 13, 17, 23, 29, 37, 42, 53, 71 and 101.

## Calendar-time replay

[Replay configuration](../../reproduction/v1.1.0/replay.json) specifies the chronological event sequence, seeds 11/13/17, historical fitting rule and capacity multipliers 0.05/0.10/0.20. Replay excludes topic descriptors from model inputs.

Service starts one hour after the first source arrival. Slots are spaced at `1 / capacity` hours and end no later than 48 hours after the last arrival. Threads are eligible from age 1h through 48h and can be reviewed once. Empty slots are unused. The latest available 1h/6h/24h score determines learned priority; ties use earlier arrival, then string thread ID. Reactions at service time count as observed, and later reactions receive capture credit. The denominator counts event reactions after age 1h.

The [replay module](../../src/research/replay.py) applies these rules to saved priority scores:

```bash
python scripts/replay_saved_scores.py /path/to/event_scores.json \
  --capacity 0.5 --output /path/to/new/replay.json
```

Input JSON fields are `thread_ids`, absolute `source_hours`, sorted relative-hour arrays `reaction_times`, and an `N x 3` `scores` array in 1h/6h/24h order. The command writes summaries and decisions for learned priority, current volume and FIFO. Model scores are supplied as input.
