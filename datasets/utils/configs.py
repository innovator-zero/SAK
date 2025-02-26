# Configurations for the datasets
PASCAL_OUT_CHANNELS = {
    "semseg": 21,
    "human_parts": 7,
    "normals": 3,
    "edge": 1,
    "sal": 2,
}

NYUD_OUT_CHANNELS = {"semseg": 40, "normals": 3, "edge": 1, "depth": 1}

INPUT_SIZE = {"pascalcontext": (512, 512), "nyud": (448, 576), "imagenet": (384, 384)}

NUM_TRAIN_IMAGES = {"pascalcontext": 4998, "nyud": 795}

NUM_TEST_IMAGES = {"pascalcontext": 5105, "nyud": 654}


def get_output_num(task, dataname):
    if dataname == "pascalcontext":
        return PASCAL_OUT_CHANNELS[task]
    elif dataname == "nyud":
        return NYUD_OUT_CHANNELS[task]
    else:
        raise NotImplementedError
