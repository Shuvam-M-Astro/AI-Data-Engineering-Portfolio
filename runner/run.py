import argparse
from pathlib import Path
from omegaconf import OmegaConf

# Import experiment entrypoints
from Clustering.Customer_Segmentation.advanced_segmentation import run_experiment as run_advanced_segmentation
from shared_utils.reproducibility import set_global_seed


def main():
    parser = argparse.ArgumentParser(description="Unified experiment runner")
    parser.add_argument("--config", type=str, default="configs/customer_segmentation.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    task = cfg.get("task", "advanced_segmentation")

    # Global reproducibility
    try:
        seed = int(cfg.get("seed", 42))
        set_global_seed(seed)
    except Exception:
        pass

    if task == "advanced_segmentation":
        run_advanced_segmentation(cfg)
    else:
        raise ValueError(f"Unsupported task: {task}")


if __name__ == "__main__":
    main()
