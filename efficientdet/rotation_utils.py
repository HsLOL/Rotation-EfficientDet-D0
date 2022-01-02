"""This rotation_utils.py is used to Rotation Detection, which is modified from efficientdet/utils.py"""
# coding=utf-8

import torch
import torch.nn as nn
import numpy as np
import math


class BBoxAddScores(nn.Module):
    def forward(self, anchors, scores):
        """

        Args:
            anchors: [xmin, ymin, xmax, ymax, theta]
            scores: 存放每个anchor对应的最大分类得分

        Returns:
            anchors: [xmin, ymin, xmax, ymax, theta, scores]
        """

        single_xmin = anchors[:, 0]
        single_ymin = anchors[:, 1]
        single_xmax = anchors[:, 2]
        single_ymax = anchors[:, 3]
        single_theta = anchors[:, 4]
        singel_scores = scores[:, 0]

        return torch.stack([single_xmin, single_ymin, single_xmax, single_ymax, single_theta, singel_scores]).t()


class Rotation_BBoxTransform(nn.Module):
    def forward(self, anchors, regression):
        """
        decode_box_outputs adapted from https://github.com/google/automl/blob/master/efficientdet/anchors.py

        Args:
            ori format:
            anchors: [batchsize, boxes, (y1, x1, y2, x2)]
            regression: [batchsize, boxes, (dy, dx, dh, dw)]

            rotation format:
            anchors: [batchsize, boxes, (x1, y1, x2, y2, theta)]
            regression: [batchsize, boxes, (dx, dy, dw, dh, dtheta)]

        Returns:

        """
        x_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
        y_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2
        ha = anchors[..., 3] - anchors[..., 1]
        wa = anchors[..., 2] - anchors[..., 0]
        ta = anchors[..., 4]

        pw = regression[..., 2].exp() * wa  # w of predicted box
        ph = regression[..., 3].exp() * ha  # h of predicted box

        px_centers = regression[..., 0] * wa + x_centers_a  # y_center of predicted box
        py_centers = regression[..., 1] * ha + y_centers_a  # x_center of predicted box
        pt = ta + (regression[..., 4] / math.pi) * 180  # radin -> rotation value

        xmin = px_centers - pw / 2.
        ymin = py_centers - ph / 2.
        xmax = px_centers + pw / 2.
        ymax = py_centers + ph / 2.

        return torch.stack([xmin, ymin, xmax, ymax, pt], dim=2)
        # return torch.stack([xmin, ymin, xmax, ymax], dim=2)  # return predicted box(ymin, xmin, ymax, xmax)


class ClipBoxes(nn.Module):

    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        # ori code
        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width - 1)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height - 1)

        return boxes


def generate_anchors(base_size, ratios, scales, rotations):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """
    num_anchors = len(ratios) * len(scales) * len(rotations)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 5))

    # scale base_size
    anchors[:, 2:4] = base_size * np.tile(scales, (2, len(scales) * len(rotations))).T
    for idx in range(len(ratios)):
        anchors[3 * idx: 3 * (idx + 1), 2] = anchors[3 * idx: 3 * (idx + 1), 2] * ratios[idx][0]
        anchors[3 * idx: 3 * (idx + 1), 3] = anchors[3 * idx: 3 * (idx + 1), 3] * ratios[idx][1]

    anchors[:, 4] = np.tile(np.repeat(rotations, len(scales)), (1, len(ratios))).T[:, 0]

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0:3:2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1:4:2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    # 返回每层对应的anchors ( x1, y1, x2, y2, theta=0)
    return anchors


def shift(shape, stride, anchors):
    shift_x = np.arange(stride / 2, shape[1], stride)
    shift_y = np.arange(stride / 2, shape[0], stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel(),
        np.zeros(shift_x.ravel().shape)
    )).transpose()
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]  # A = 9
    K = shifts.shape[0]  # K = 64 * 64 = 4096
    all_anchors = (anchors.reshape((1, A, 5)) + shifts.reshape((1, K, 5)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 5))
    return all_anchors


class Anchors(nn.Module):
    """
    adapted and modified from https://github.com/google/automl/blob/master/efficientdet/anchors.py by Zylo117
    2021/04/19 modified the function of the class Anchors to the Rotation Anchors
    """

    def __init__(self, anchor_scale=4., pyramid_levels=None, rotations=None, **kwargs):
        super().__init__()
        self.anchor_scale = anchor_scale

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels

        self.strides = kwargs.get('strides', [2 ** x for x in self.pyramid_levels])
        self.scales = np.array(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        self.ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])

        self.base_sizes = [x * anchor_scale for x in self.strides]
        # add rotation
        if rotations is None:
            self.rotations = np.array([-90])

        self.last_anchors = {}
        self.last_shape = None

    def forward(self, image, dtype=torch.float32):
        """Generates multiscale rotation anchor boxes.

        Args:
          image_size: integer number of input image size. The input image has the
            same dimension for width and height. The image_size should be divided by
            the largest feature stride 2^max_level.
          anchor_scale: float number representing the scale of size of the base
            anchor to the feature stride 2^level.
          anchor_configs: a dictionary with keys as the levels of anchors and
            values as a list of anchor configuration.

        Returns:
          rotation_anchor_boxes: a numpy array with shape [N, 5](x1, y1, x2, y2, theta=0),
          which stacks anchors on all feature levels.
        Raises:
          ValueError: input size must be the multiple of largest feature stride.
        """
        image_shape = image.shape[2:]
        if image_shape == self.last_shape and image.device in self.last_anchors:
            return self.last_anchors[image.device]

        if self.last_shape is None or self.last_shape != image_shape:
            self.last_shape = image_shape

        if dtype == torch.float16:
            dtype = np.float16
        else:
            dtype = np.float32

        # add rotation bbox
        all_anchors = np.zeros((0, 5)).astype(np.float32)
        for idx, p in enumerate(self.pyramid_levels):
            rotation_anchors = generate_anchors(
                base_size=self.base_sizes[idx],
                ratios=self.ratios,
                scales=self.scales,
                rotations=self.rotations
            )

            shifted_anchors = shift(image_shape, self.strides[idx], rotation_anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)
        all_anchors = np.expand_dims(all_anchors, axis=0)
        all_anchors = np.tile(all_anchors, (image.shape[0], 1, 1))
        all_anchors = torch.from_numpy(all_anchors.astype(dtype))
        if torch.is_tensor(image) and image.is_cuda:
            all_anchors = all_anchors.cuda()
        return all_anchors
