"""Run the metadata profiler on a tiny synthetic matrix."""

import json

import pandas as pd

from agentfs.metadata_profiler import profile_dataset

expression = pd.DataFrame(
    {
        "probe_a": [1.0, 2.0, 3.0, 4.0],
        "probe_b": [5.0, 5.0, 5.0, 5.0],
        "probe_c": [0.2, 0.3, None, 0.5],
    }
)
labels = pd.Series(["ALL", "ALL", "AML", "AML"], name="class")
print(json.dumps(profile_dataset(expression, labels), indent=2))
