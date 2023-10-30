import random
from typing import Tuple

import albumentations as alb
from albumentations.augmentations.crops import functional as AF
import cv2
import numpy as np
import torch
from dataset.moco.augmentations import functional as MF


class ColorJitter(alb.ImageOnlyTransform):
    r"""
    Randomly change brightness, contrast, hue and saturation of the image. This
    class behaves exactly like :class:`torchvision.transforms.ColorJitter` but
    is slightly faster (uses OpenCV) and compatible with rest of the transforms
    used here (albumentations-style). This class works only on ``uint8`` images.
    .. note::
        Unlike torchvision variant, this class follows "garbage-in, garbage-out"
        policy and does not check limits for jitter factors. User must ensure
        that ``brightness``, ``contrast``, ``saturation`` should be ``float``
        in ``[0, 1]`` and ``hue`` should be a ``float`` in ``[0, 0.5]``.
    Parameters
    ----------
    brightness: float, optional (default = 0.4)
        How much to jitter brightness. ``brightness_factor`` is chosen
        uniformly from ``[1 - brightness, 1 + brightness]``.
    contrast: float, optional (default = 0.4)
        How much to jitter contrast. ``contrast_factor`` is chosen uniformly
        from ``[1 - contrast, 1 + contrast]``
    saturation: float, optional (default = 0.4)
        How much to jitter saturation. ``saturation_factor`` is chosen
        uniformly from ``[1 - saturation, 1 + saturation]``.
    hue: float, optional (default = 0.4)
        How much to jitter hue. ``hue_factor`` is chosen uniformly from
        ``[-hue, hue]``.
    always_apply: bool, optional (default = False)
        Indicates whether this transformation should be always applied.
    p: float, optional (default = 0.5)
        Probability of applying the transform.
    """

    def __init__(
        self,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.4,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def apply(self, img, **params):
        original_dtype = img.dtype

        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
        hue_factor = random.uniform(-self.hue, self.hue)

        # Convert arguments as required by albumentations functional interface.
        # "gain" = contrast and "bias" = (brightness_factor - 1)
        img = alb.augmentations.functional.brightness_contrast_adjust(
            img, alpha=contrast_factor, beta=brightness_factor - 1
        )
        # Hue and saturation limits are required to be integers.
        img = alb.augmentations.functional.shift_hsv(
            img,
            hue_shift=int(hue_factor * 255),
            sat_shift=int(saturation_factor * 255),
            val_shift=0,
        )
        img = img.astype(original_dtype)
        return img

    def get_transform_init_args_names(self):
        return ("brightness", "contrast", "saturation", "hue")


class OverlappingPairRandomResizedCrop(alb.ImageOnlyTransform):
    r"""Take two overlapping random crops from an input image."""

    def __init__(
        self,
        height: int,
        width: int,
        min_areacover: float = 0.2,
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (0.75, 1.3333333333333333),
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: bool = False,
        p: float = 1.0,
    ):

        super().__init__(always_apply=always_apply, p=p)
        self.height = height
        self.width = width
        self.min_areacover = min_areacover
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def apply(self, img, **params):
        first_crop_coords = MF.random_crop_dry(img, self.scale, self.ratio)
        second_crop_coords = MF.get_random_crop_coords_bbox_cover(
            img, first_crop_coords, self.min_areacover, self.scale, self.ratio
        )

        
        first_crop = alb.augmentations.geometric.resize(
            AF.bbox_crop(img, *first_crop_coords),
            height=self.height,
            width=self.width,
            interpolation=self.interpolation,
        )
        second_crop = alb.augmentations.geometric.resize(
            AF.bbox_crop(img, *second_crop_coords),
            height=self.height,
            width=self.width,
            interpolation=self.interpolation,
        )
        # Get overlapping mask.
        mask = MF.get_reference_crop_covering_mask(
            img, first_crop_coords, second_crop_coords
        )
        mask = alb.augmentations.geometric.resize(mask, self.height, self.width)
        # mask = mask.astype(bool)
        return first_crop, second_crop, mask

    def get_transform_init_args_names(self):
        return ("height", "width", "min_areacover", "scale", "ratio", "interpolation")


class MaskConstraintRandomResizedPairCrop(alb.ImageOnlyTransform):
    r"""
    Take two independent random crops from an input image which cover with
    a reference (ground truth) mask.
    """

    def __init__(
        self,
        height: int,
        width: int,
        min_areacover: float = 0.2,
        second_constraint: str = "all",
        output_mask_region: str = "all",
        scale: Tuple[float, float] = (0.2, 1.0),
        ratio: Tuple[float, float] = (3 / 4, 4 / 3),
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        if second_constraint not in {"ref", "all"}:
            raise ValueError(
                f"second_constraint should be either `ref` or `all`,"
                f"found {second_constraint}."
            )

        if output_mask_region not in {"ref", "all"}:
            raise ValueError(
                f"output_mask_region should be either `ref` or `all`,"
                f"found {output_mask_region}."
            )

        super().__init__(always_apply=always_apply, p=p)
        self.height = height
        self.width = width
        self.min_areacover = min_areacover
        self.second_constraint = second_constraint
        self.output_mask_region = output_mask_region
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def apply(self, img: np.ndarray, reference_mask: np.ndarray, **params):
        first_crop_coords = MF.get_random_crop_coords_mask_cover(
            img,
            reference_mask,
            other_coords=None,
            min_areacover=self.min_areacover,
            scale=self.scale,
            ratio=self.ratio,
        )
        # Check whether first crop is a constraint for second crop. Reference
        # mask will be a constraint by default.

        second_crop_coords = MF.get_random_crop_coords_mask_cover(
            img,
            reference_mask,
            other_coords=(
                [] if self.second_constraint == "ref" else [first_crop_coords]
            ),
            min_areacover=self.min_areacover,
            scale=self.scale,
            ratio=self.ratio,
        )

        print(*first_crop_coords)
        # Crop image using these coordinates and resize them.
        first_crop = alb.augmentations.geometric.resize(
            AF.bbox_crop(img, *first_crop_coords),
            height=self.height,
            width=self.width,
            interpolation=self.interpolation,
        )
        second_crop = alb.augmentations.geometric.resize(
            AF.bbox_crop(img, *second_crop_coords),
            height=self.height,
            width=self.width,
            interpolation=self.interpolation,
        )
        # Get mask for supervision (ALWAYS with respect to first crop).
        # Check whether the mask should be the region inside just first crop,
        # or the intersection of both crops.
        mask_query = MF.get_reference_crop_covering_mask(
            reference_mask,
            reference_coords=first_crop_coords,
            other_coords=(
                [second_crop_coords] if self.output_mask_region == "all" else []
            ),
        )
        mask_key = MF.get_reference_crop_covering_mask(
            reference_mask,
            reference_coords=second_crop_coords,
            other_coords=(
                [first_crop_coords] if self.output_mask_region == "all" else []
            ),
        )

        mask_query = alb.augmentations.geometric.resize(mask_query, self.height, self.width)
        mask_key = alb.augmentations.geometric.resize(mask_key, self.height, self.width)
        # mask = mask.astype(bool)
        return first_crop, second_crop, mask_query, mask_key

    @property
    def targets_as_params(self):
        return ["reference_mask"]

    def get_params_dependent_on_targets(self, params):
        return params


class BboxConstraintRandomResizedPairCrop(alb.ImageOnlyTransform):
    r"""
    Take two independent random crops from an input image which cover with
    a reference (ground truth) mask.
    """

    def __init__(
        self,
        height: int,
        width: int,
        min_areacover: float = 0.2,
        second_constraint: str = "all",
        output_mask_region: str = "all",
        scale: Tuple[float, float] = (0.2, 1.0),
        ratio: Tuple[float, float] = (3 / 4, 4 / 3),
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        if second_constraint not in {"ref", "all"}:
            raise ValueError(
                f"second_constraint should be either `ref` or `all`,"
                f"found {second_constraint}."
            )

        if output_mask_region not in {"ref", "all"}:
            raise ValueError(
                f"output_mask_region should be either `ref` or `all`,"
                f"found {output_mask_region}."
            )

        super().__init__(always_apply=always_apply, p=p)
        self.height = height
        self.width = width
        self.min_areacover = min_areacover
        self.second_constraint = second_constraint
        self.output_mask_region = output_mask_region
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def apply(
        self, img: np.ndarray, reference_coords: Tuple[int, int, int, int], **params
    ):
        first_crop_coords = MF.get_random_crop_coords_bbox_cover(
            img,
            reference_coords,
            min_areacover=self.min_areacover,
            scale=self.scale,
            ratio=self.ratio,
        )
        # Check whether first crop is a constraint for second crop. Reference
        # mask will be a constraint by default.
        second_crop_coords = MF.get_random_crop_coords_mask_cover(
            img,
            reference_coords=(
                [reference_coords]
                if self.second_constraint == "ref" else
                [reference_coords, first_crop_coords]
            ),
            min_areacover=self.min_areacover,
            scale=self.scale,
            ratio=self.ratio,
        )
        # Crop image using these coordinates and resize them.
        first_crop = alb.augmentations.geometric.resize(
            AF.bbox_crop(img, *first_crop_coords),
            height=self.height,
            width=self.width,
            interpolation=self.interpolation,
        )
        second_crop = alb.augmentations.geometric.resize(
            AF.bbox_crop(img, *second_crop_coords),
            height=self.height,
            width=self.width,
            interpolation=self.interpolation,
        )
        # Get mask for supervision (ALWAYS with respect to first crop).
        # Check whether the mask should be the region inside just first crop,
        # or the intersection of both crops.
        mask = MF.get_reference_crop_covering_mask(
            img,
            reference_coords=first_crop_coords,
            other_coords=(
                [second_crop_coords] if self.output_mask_region == "all" else []
            ),
        )
        mask = alb.augmentations.geometric.resize(mask, self.height, self.width)
        # mask = mask.astype(bool)
        return first_crop, second_crop, mask

    @property
    def targets_as_params(self):
        return ["reference_coords"]

    def get_params_dependent_on_targets(self, params):
        return params


class ToTensorV2(alb.BasicTransform):
    """Convert image and mask to `torch.Tensor`."""

    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask}

    def apply(self, img, **params):  # skipcq: PYL-W0613
        return torch.from_numpy(img.transpose(2, 0, 1))

    def apply_to_mask(self, mask, **params):  # skipcq: PYL-W0613
        return torch.from_numpy(mask)

    def get_transform_init_args_names(self):
        return []

    def get_params_dependent_on_targets(self, params):
        return {}


# =============================================================================

IMAGENET_COLOR_MEAN = (0.485, 0.456, 0.406)
r"""ImageNet color normalization mean in RGB format (values in 0-1)."""

IMAGENET_COLOR_STD = (0.229, 0.224, 0.225)
r"""ImageNet color normalization std in RGB format (values in 0-1)."""
