import argparse
import logging
from pathlib import Path
from omegaconf import OmegaConf

# Import experiment entrypoints
from Clustering.Customer_Segmentation.advanced_segmentation import run_experiment as run_advanced_segmentation
from shared_utils.reproducibility import set_global_seed


AVAILABLE_TASKS = {
    "advanced_segmentation": run_advanced_segmentation,
}


def main():
    parser = argparse.ArgumentParser(description="Unified experiment runner")
    parser.add_argument("--config", type=str, default="configs/customer_segmentation.yaml", help="Path to YAML config")
    parser.add_argument("--task", type=str, default=None, help="Override task from config (e.g., advanced_segmentation)")
    parser.add_argument("--list-tasks", action="store_true", help="List available tasks and exit")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed from config")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print resolved settings without executing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("runner")

    if args.list_tasks:
        print("Available tasks:")
        for name in sorted(AVAILABLE_TASKS.keys()):
            print(f"- {name}")
        return

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg = OmegaConf.load(str(config_path))

    task = args.task or cfg.get("task", "advanced_segmentation")
    if task not in AVAILABLE_TASKS:
        raise ValueError(f"Unsupported task: {task}. Use --list-tasks to see options.")

    # Global reproducibility
    try:
        seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
        set_global_seed(int(seed))
        logger.info(f"Global seed set to {seed}")
    except Exception:
        logger.warning("Failed to set global seed; proceeding without deterministic setup")

    if args.dry_run:
        logger.info("Dry run enabled. Resolved configuration:")
        print(OmegaConf.to_yaml(cfg))
        return

    # Execute task
    logger.info(f"Starting task: {task}")
    AVAILABLE_TASKS[task](cfg)
    logger.info("Task completed successfully")


if __name__ == "__main__":
    main()
