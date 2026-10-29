"""
Data Loading and Preprocessing Pipeline for Microarray Gene-Expression Datasets.
Part of the hyperfs pipeline.
"""

import os
import re
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


class MicroarrayDataLoader:
    """
    Modular loader and preprocessor for high-dimensional microarray data.
    """
    def __init__(self, file_path: str, label_column: str = 'target'):
        self.file_path = file_path
        self.label_column = label_column
        self.df: Optional[pd.DataFrame] = None
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[np.ndarray] = None

    def load_raw_data(
        self,
        *,
        feature_ids: Optional[Sequence[str]] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load features and labels, optionally restoring IDs from a verified source.

        ``feature_ids`` must be in the same order as the feature columns in the
        input CSV. Generic ``GENE_1``-style placeholders are rejected unless
        verified identifiers are supplied explicitly.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset file not found at: {self.file_path}")

        self.df = pd.read_csv(self.file_path)

        if self.label_column in self.df.columns:
            self.y = self.df[self.label_column].values
            self.X = self.df.drop(columns=[self.label_column])
        else:
            raise KeyError(f"Label column '{self.label_column}' not found in dataset.")

        if feature_ids is not None:
            ids = [str(feature_id).strip() for feature_id in feature_ids]
            if len(ids) != self.X.shape[1]:
                raise ValueError(
                    f"Expected {self.X.shape[1]} feature IDs, received {len(ids)}."
                )
            if any(not feature_id for feature_id in ids):
                raise ValueError("Feature IDs must not be empty.")
            if len(set(ids)) != len(ids):
                raise ValueError("Feature IDs must be unique.")
            self.X.columns = ids
        elif any(re.fullmatch(r"GENE_\d+", str(column)) for column in self.X.columns):
            raise ValueError(
                "Generic GENE_n feature labels are not identifiers. Pass the verified "
                "feature_ids in source-column order or use a source file with real IDs."
            )

        print(f"Loaded raw dataset: {self.X.shape[0]} samples × {self.X.shape[1]} features.")
        return self.X, self.y

    def apply_log2_transform(self, offset: float = 1.0) -> pd.DataFrame:
        """
        Applies log2 transformation: log2(x + offset) to stabilize expression variance.
        """
        # Ensure values are non-negative before applying log2
        min_val = self.X.values.min()
        if min_val < 0:
            self.X = self.X - min_val

        self.X = np.log2(self.X + offset)
        print("Applied log2 transformation.")
        return self.X

    def handle_missing_values(self) -> pd.DataFrame:
        """Fills missing values with column-wise median."""
        missing_count = self.X.isnull().sum().sum()
        if missing_count > 0:
            self.X = self.X.fillna(self.X.median())
            print(f"Imputed {missing_count} missing values using median strategy.")
        else:
            print("No missing values found.")
        return self.X

    def apply_variance_filter(self, top_k: int = 1000) -> pd.DataFrame:
        """
        Pre-screens features by keeping the top_k features with the highest variance.
        Crucial for mitigating high-dimensionality overhead (p >> n).
        """
        if self.X.shape[1] <= top_k:
            print(f"Feature count ({self.X.shape[1]}) <= top_k ({top_k}). Skipping variance filter.")
            return self.X

        variances = self.X.var(axis=0)
        top_feature_names = variances.nlargest(top_k).index
        self.X = self.X[top_feature_names]
        print(f"Variance Filter Applied: Filtered from {len(variances)} to top {top_k} features.")
        return self.X

    def map_probes(self, mapping_dict: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Renames probe IDs to official Gene Symbols if mapping dictionary is provided.
        """
        if mapping_dict:
            self.X = self.X.rename(columns=mapping_dict)
            # Aggregate duplicate gene columns by mean expression value
            self.X = self.X.groupby(level=0, axis=1).mean()
            print(f"Probe mapping applied. Retained {self.X.shape[1]} unique mapped genes.")
        else:
            print("No mapping dictionary provided. Keeping original probe IDs.")
        return self.X

    def save_processed(self, output_path: str):
        """Saves the preprocessed X matrix and target y into processed directory."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        processed_df = self.X.copy()
        processed_df[self.label_column] = self.y
        processed_df.to_csv(output_path, index=False)
        print(f"Saved preprocessed matrix to: {output_path}")
