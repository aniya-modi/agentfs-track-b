"""Run the metadata profiler on a tiny synthetic matrix."""

import pandas as pd

from agentfs.metadata_profiler import profile_dataset, profile_to_json

expression = pd.DataFrame(
    {
        "probe_a": [1.0, 2.0, 3.0, 4.0],
        "probe_b": [5.0, 5.0, 5.0, 5.0],
        "probe_c": [0.2, 0.3, None, 0.5],
    }
)
labels = pd.Series(["ALL", "ALL", "AML", "AML"], name="class")
print(profile_to_json(profile_dataset(expression, labels, dataset_id="synthetic-example")))
