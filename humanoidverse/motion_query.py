import os

os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import rich
import time

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.envs.g1_env_helper.queries import load_query
from humanoidverse.reward_inference import make_reward_agent


def main(model_folder: Path, query: Path, sort_by: str | None = None, top_k: int = 20, num_samples: int = 150_000, output: Path | None = None, device="cuda"):
    """Select replay-buffer frames with a QueryFunction from envs/g1_env_helper/queries.

    Example queries are in humanoidverse/config/queries.

    Only needs the model and the replay buffer, no simulator.

    Args:
        query: YAML file with the query class under 'type' and its fields, e.g.
            type: LocomotionQuery / min_speed: 0.5 / max_speed: 1.0.
        sort_by: metric to sort the printed table by (descending).
        top_k: number of rows to print.
        num_samples: frames drawn from the buffer; at least the buffer size uses every frame once.
        output: MotionDatabase pickle (default: model_folder/motion_query/motion_db.pkl).
    """
    model_folder = Path(model_folder)
    query = load_query(query)

    model = load_model_from_checkpoint_dir(model_folder / "checkpoint", device=device)
    model.to(device)
    model.eval()

    reward_eval_agent = make_reward_agent(model, model_folder, num_samples)
    start_t = time.time()
    db = reward_eval_agent.query_motions(query)
    print(f"{len(db)} frames match {query} (done in {time.time()-start_t}s)")
    output = Path(output) if output is not None else model_folder / "motion_query" / "motion_db.pkl"
    output.parent.mkdir(parents=True, exist_ok=True)
    db.save(output)
    print(f"Saved MotionDatabase at {output}")
    if sort_by is not None:
        db = db.sort(sort_by, descending=True)
    rich.print(db.to_dataframe().head(top_k).to_string())


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
