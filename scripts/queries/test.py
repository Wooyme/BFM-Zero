from pathlib import Path

from humanoidverse.query.motion_query import Source
import numpy as np

def main(source: Source):
    q = source.scan()
    q.where(
        'base_height<0.8 greatest(left_wrist_above_shoulder, right_wrist_above_shoulder) <= 0 abs(base_vx) <= 1.05 AND abs(base_vy) <= 1.05 AND abs(yaw_rate) <= 1.5')
    q.with_stability({"d_score": "distance_score"})
    q.where("d_score>96")
    result = q.collect()
    print(np.max(result.base_height))
    print(result)


if __name__ == "__main__":
    main(Source(model_folder=Path("new_model_for_training_code_inference"), lafan="rec*"))
