import random

import numpy as np
import torch


def fix_random_state(seed: int = 42, benchmark=False):
    seed = get_random_seed(seed)

    # Python library
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = benchmark
    if benchmark:
        print('Benchmark Mode')
    # torch.use_deterministic_algorithms(True)  # RuntimeError
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.allow_tf32 = False

    print('Fixed seed: {}'.format(seed))


def get_random_seed(seedval=None):
    if seedval is None:
        return random.randint(1, 1000000)
    return seedval


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_weights(model, load_state_dict, auto_slice: bool = True):
    state_dict = {}
    # convert data_parallal to model
    for k in load_state_dict:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = load_state_dict[k]
        else:
            state_dict[k] = load_state_dict[k]

    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    for k in model_state_dict.keys():
        if k in state_dict:
            load_shape = state_dict[k].shape
            model_shape = model_state_dict[k].shape
            if load_shape != model_shape:
                if auto_slice and load_shape[1:] == model_shape[1:]:
                    # If the shape is just different the output size (of conv),
                    # slice the weights so that the weight matches the model.
                    print(
                        'The shape {} is automatically changed '.format(k)
                        + 'from {} to {}.'.format(
                            str(load_shape), str(model_shape)))
                    state_dict[k] = state_dict[k][:model_shape[0]]
                else:
                    print('Different shape {}.'.format(k))
                    state_dict[k] = model_state_dict[k]
        else:
            print('Loaded state_dict does not have {}.'.format(k))
            state_dict[k] = model_state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    # print('Model Loaded: ', filename)
    return model
