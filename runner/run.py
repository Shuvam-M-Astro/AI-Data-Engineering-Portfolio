import argparse
from pathlib import Path
from omegaconf import OmegaConf

# Import experiment entrypoints
from Clustering.Customer_Segmentation.advanced_segmentation import run_experiment as run_advanced_segmentation
from shared_utils.reproducibility import set_global_seed


def main():
    parser = argparse.ArgumentParser(description="Unified experiment runner")
    parser.add_argument("--config", type=str, default="configs/customer_segmentation.yaml", help="Path to YAML config")
    parser.add_argument("--task", type=str, default=None, help="Override task from config (e.g., advanced_segmentation)")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg = OmegaConf.load(str(config_path))

    task = args.task or cfg.get("task", "advanced_segmentation")

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
