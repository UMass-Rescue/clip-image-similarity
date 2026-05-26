#!/usr/bin/env python3
"""Run data_workflow.sh for every (model, dataset) combination in config.json."""

import argparse
import json
import subprocess
import sys

from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    runs = [
        (model, dataset)
        for model in cfg["models"]
        for dataset in cfg["datasets"]
    ]

    for model, dataset in tqdm(runs, desc="Workflows"):
        output_dir = f"{cfg['output_base_dir']}/{model['name']}/{dataset['name']}"
        print(f"\n{'='*60}")
        print(f"{model['name']} × {dataset['name']}")
        print(f"Output: {output_dir}")
        print(f"{'='*60}\n")

        result = subprocess.run(
            [
                "bash", "data_workflow.sh",
                "--model",               model["id"],
                "--series-image-path",   dataset["series_image_path"],
                "--series-labels-path",  dataset["series_labels_path"],
                "--output-base-dir",     output_dir,
                "--min-samples",         str(cfg["min_samples"]),
                "--distance-threshold",  str(dataset["distance_threshold"]),
            ],
            stdin=subprocess.DEVNULL,
        )

        if result.returncode != 0:
            tqdm.write(
                f"\nERROR: run failed for {model['name']} × {dataset['name']}",
                file=sys.stderr,
            )
            sys.exit(result.returncode)

    print(f"\nAll {len(runs)} runs completed.")


if __name__ == "__main__":
    main()
