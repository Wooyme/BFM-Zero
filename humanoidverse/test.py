import h5py
import torch
import gc

from torch.utils._pytree import tree_map

from humanoidverse.agents.buffers.trajectory import find_start_stop_traj, TrajectoryDictBufferMultiDim, tree_concat
from humanoidverse.agents.buffers.transition import DictBuffer


def test1():
    buffer_path = "new_model_for_training_code_inference/checkpoint/buffers/train"
    dataset = TrajectoryDictBufferMultiDim.load(buffer_path)
    dataset.get_full_buffer()


def test2():
    hf = h5py.File("new_model_for_training_code_inference/checkpoint/buffers/train/buffer.hdf5", "r")
    next_keys = ["observation", "qpos", "qvel"]
    cur_keys = ["action"]
    truncated = torch.from_numpy(hf['truncated'][:])
    start_idx, stop_idx, lengths = find_start_stop_traj(
        truncated.squeeze()[:], at_capacity=True, cursor=1249
    )
    del truncated
    gc.collect()
    storage = {"next": {}}

    for next_key in next_keys:
        data = {}
        for k, v in hf.items():
            if next_key in k:
                splits = k.split('-')
                current_dict = data
                for i in range(len(splits) - 1):
                    key = splits[i]
                    if key not in current_dict:
                        current_dict[key] = {}
                    current_dict = current_dict[key]
                current_dict[splits[-1]] = torch.from_numpy(v[:])
        for start, length in zip(start_idx, lengths):
            idxs = (start[0] + 1 + torch.arange(length - 1, device='cpu')) % 5000
            if next_key not in storage["next"]:
                storage["next"][next_key] = []
            storage["next"][next_key].append(tree_map(lambda x: x[idxs, start[1]], data))
        storage["next"][next_key] = tree_concat(storage["next"][k])
        print(data.keys())
        del data
    print(storage)


# test1()
test2()