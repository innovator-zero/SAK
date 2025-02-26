import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class VOCSegmentationMaskDecoder(object):

    def __init__(self, n_classes):
        self.labels = np.asarray(
            [
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
            ]
        )[0:n_classes]

    def __call__(self, label_mask):
        # Initialize the segmentation map using the "void" colour
        r = np.ones_like(label_mask).astype(np.uint8) * 224
        g = np.ones_like(label_mask).astype(np.uint8) * 223
        b = np.ones_like(label_mask).astype(np.uint8) * 192

        for ind, label_colour in enumerate(self.labels):
            r[label_mask == ind] = label_colour[0]
            g[label_mask == ind] = label_colour[1]
            b[label_mask == ind] = label_colour[2]

        rgb = np.stack([r, g, b], axis=0)
        rgb = np.swapaxes(np.swapaxes(rgb, 0, 1), 1, 2)  # 3, H, W => H, W, 3
        return rgb


class NYUDSegmentationMaskDecoder(object):

    def __init__(self, n_classes):
        self.labels = self.colormap(n_classes)

    def colormap(self, N):

        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7 - j)
                g = g | (bitget(c, 1) << 7 - j)
                b = b | (bitget(c, 2) << 7 - j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

        return cmap

    def __call__(self, label_mask):
        # Initialize the segmentation map using the "void" colour
        r = np.ones_like(label_mask).astype(np.uint8) * 224
        g = np.ones_like(label_mask).astype(np.uint8) * 223
        b = np.ones_like(label_mask).astype(np.uint8) * 192

        for ind, label_colour in enumerate(self.labels):
            r[label_mask == ind] = label_colour[0]
            g[label_mask == ind] = label_colour[1]
            b[label_mask == ind] = label_colour[2]

        rgb = np.stack([r, g, b], axis=0)
        rgb = np.swapaxes(np.swapaxes(rgb, 0, 1), 1, 2)  # 3, H, W => H, W, 3
        return rgb


def save_img(dataset, file_names, preds, task, save_dir):
    """
    Save the predictions as images
    :param dataset: Dataset name
    :param file_names: List of image names
    :param preds: List of predictions
    :param task: Task name
    :param save_dir: Directory to save the predictions
    """
    for i in range(len(preds)):
        img_name = file_names[i]

        # Save predictions for the different tasks
        if task in {"edge", "sal"}:
            pred_image = Image.fromarray(np.uint8(preds[i]), mode="L")
        elif task == "semseg":
            if dataset == "pascalcontext":
                decoder = VOCSegmentationMaskDecoder(21)
            elif dataset == "nyud":
                decoder = NYUDSegmentationMaskDecoder(41)
                preds[i] += 1
            else:
                raise NotImplementedError
            pred_image = Image.fromarray(decoder(preds[i]), mode="RGB")
        elif task == "human_parts":
            decoder = VOCSegmentationMaskDecoder(7)
            pred_image = Image.fromarray(decoder(preds[i]), mode="RGB")
        elif task == "normals":
            pred_image = Image.fromarray(np.uint8(preds[i]), mode="RGB")
        elif task == "depth":
            pred = preds[i]
            np.save(os.path.join(save_dir, task, img_name), pred)

            # visualize with matplotlib
            # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            # plt.imshow(pred)
            # plt.axis("off")
            # plt.savefig(os.path.join(save_dir, task, img_name + ".png"))
            continue
        else:
            raise ValueError("Image Saving Task {} is Not Supported!".format(task))

        if task == "edge":
            pred_image.save(os.path.join(save_dir, "edge", "img", img_name + ".png"))
        else:
            pred_image.save(os.path.join(save_dir, task, img_name + ".png"))
