"""Checks for split-specific topic fitting."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research.text_models import PROTOTYPE_TEXTS, _aggregate_thread_text_features
from research.triage import (
    _feature_frame,
    _fit_transform_topic_features,
    _merge_source_semantic_features,
)


class TrainingOnlyTopicTests(unittest.TestCase):
    @staticmethod
    def frame(values):
        values = np.asarray(values, dtype=float)
        return pd.DataFrame(
            {
                "thread_id": [f"t{index}" for index in range(len(values))],
                **{
                    f"__topic_source_embedding_{dimension:03d}": values[:, dimension]
                    for dimension in range(values.shape[1])
                },
                **{
                    f"thread_text_topic_embedding_{dimension:03d}": values[:, dimension]
                    for dimension in range(values.shape[1])
                },
            }
        )

    def test_heldout_values_do_not_change_training_fit(self):
        training = self.frame([[index, index % 3] for index in range(12)])
        heldout_a = self.frame([[100, 100], [200, 200]])
        heldout_b = self.frame([[-1000, 500], [3000, -700]])

        train_a, _ = _fit_transform_topic_features(training, heldout_a)
        train_b, _ = _fit_transform_topic_features(training, heldout_b)

        for column in (
            "source_sem_topic_cluster",
            "source_sem_topic_distance_to_center",
            "thread_text_source_topic_cluster",
            "thread_text_source_topic_distance",
        ):
            pd.testing.assert_series_equal(train_a[column], train_b[column])

    def test_heldout_uses_training_centroids(self):
        values = np.asarray([[index, index % 3] for index in range(12)], dtype=float)
        training = self.frame(values)
        heldout_values = np.asarray([[0.25, 0.75], [20.0, 2.0]], dtype=float)
        heldout = self.frame(heldout_values)

        _, transformed = _fit_transform_topic_features(training, heldout)
        expected_model = KMeans(n_clusters=8, random_state=42, n_init="auto").fit(values)
        expected_labels = expected_model.predict(heldout_values)
        expected_distances = np.linalg.norm(
            heldout_values - expected_model.cluster_centers_[expected_labels], axis=1
        )

        np.testing.assert_array_equal(
            transformed["source_sem_topic_cluster"].to_numpy(),
            expected_labels.astype(str),
        )
        np.testing.assert_allclose(
            transformed["source_sem_topic_distance_to_center"].to_numpy(float),
            expected_distances,
        )

    def test_raw_embeddings_are_not_ranker_columns(self):
        frame = self.frame([[0, 1], [1, 0]])
        frame["source_sem_embedding_mean"] = [0.1, 0.2]
        frame["source_sem_topic_cluster"] = ["0", "1"]
        output = _feature_frame(frame, 1, "semantic")

        self.assertIn("source_sem_embedding_mean", output.columns)
        self.assertTrue(any("source_sem_topic_cluster" in column for column in output.columns))
        self.assertFalse(any(column.startswith("__topic_") for column in output.columns))
        self.assertFalse(any(column.startswith("thread_text_topic_embedding_") for column in output.columns))

    def test_precomputed_embeddings_reach_split_transform(self):
        rows = [
            {
                "dataset": "example",
                "thread_id": "thread-1",
                "post_id": "source-1",
                "post_type": "source",
                "text": "source",
                "timestamp_relative_hours": 0,
            },
            {
                "dataset": "example",
                "thread_id": "thread-1",
                "post_id": "correction-1",
                "post_type": "correction",
                "text": "correction",
                "timestamp_relative_hours": 1,
            },
        ]
        post_features = pd.DataFrame(
            [
                {
                    "dataset": "example",
                    "thread_id": "thread-1",
                    "post_id": "source-1",
                    "post_type": "source",
                    "embedding_000": 1.0,
                    "embedding_001": 2.0,
                    "embedding_mean": 1.5,
                    "embedding_std": 0.5,
                    **{f"sim_{label}": 0.0 for label in PROTOTYPE_TEXTS},
                },
                {
                    "dataset": "example",
                    "thread_id": "thread-1",
                    "post_id": "correction-1",
                    "post_type": "correction",
                    "embedding_000": 3.0,
                    "embedding_001": 4.0,
                    "embedding_mean": 3.5,
                    "embedding_std": 0.5,
                    **{f"sim_{label}": 0.0 for label in PROTOTYPE_TEXTS},
                },
            ]
        )
        thread_features = _aggregate_thread_text_features(
            rows=rows,
            dataset_name="example",
            post_feature_frame=post_features,
            include_zero_shot=False,
        )
        self.assertEqual(thread_features.loc[0, "topic_embedding_000"], 2.0)
        self.assertEqual(thread_features.loc[0, "topic_embedding_001"], 3.0)
        self.assertEqual(thread_features.loc[0, "source_topic_embedding_000"], 1.0)
        self.assertEqual(thread_features.loc[0, "source_topic_embedding_001"], 2.0)

        with tempfile.TemporaryDirectory() as directory:
            feature_dir = Path(directory)
            post_features[
                [
                    column
                    for column in post_features.columns
                    if not column.startswith("embedding_")
                    or not column.removeprefix("embedding_").isdigit()
                ]
            ].to_csv(feature_dir / "post_semantic_features.csv", index=False)
            thread_features.to_csv(feature_dir / "thread_text_features.csv", index=False)
            merged = _merge_source_semantic_features(
                pd.DataFrame({"dataset": ["example"], "thread_id": ["thread-1"]}),
                feature_dir,
            )

        self.assertEqual(merged.loc[0, "__topic_source_embedding_000"], 1.0)
        self.assertEqual(merged.loc[0, "thread_text_topic_embedding_000"], 2.0)


if __name__ == "__main__":
    unittest.main()
