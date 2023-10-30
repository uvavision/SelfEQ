import math
import random
from typing import List, Optional, Tuple, Union

from albumentations.augmentations import functional as F
import numpy as np
from scipy.ndimage import center_of_mass


def random_crop_dry(
    img: np.ndarray,
    scale: Tuple[float, float] = (0.2, 1.0),
    ratio: Tuple[float, float] = (3 / 4, 4 / 3),
) -> Tuple[int, int, int, int]:
    r"""
    Compute ``[x1, y1, x2, y2]`` coordinates of a random crop satisfying the
    given scale and aspect ratio constraints. 
    """
    height, width = img.shape[0], img.shape[1]
    area = height * width

    # Make at most 10 attempts to do a random crop.
    for _attempt in range(10):
        target_area = random.uniform(*scale) * area
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        aspect_ratio = math.exp(random.uniform(*log_ratio))

        crop_height = int(round(math.sqrt(target_area / aspect_ratio)))
        crop_width = int(round(math.sqrt(target_area * aspect_ratio)))

        if 0 < crop_height <= img.shape[0] and 0 < crop_width <= img.shape[1]:
            i = random.randint(0, img.shape[0] - crop_height)
            j = random.randint(0, img.shape[1] - crop_width)

            h_start = i * 1.0 / (img.shape[0] - crop_height + 1e-10)
            w_start = j * 1.0 / (img.shape[1] - crop_width + 1e-10)

            # Get unnormalized crop bbox coordinates.
            x1, y1, x2, y2 = F.get_random_crop_coords(
                height, width, crop_height, crop_width, h_start, w_start
            )
            return (x1, y1, x2, y2)

    # If random crop could not work, then fallback to central crop.
    in_ratio = img.shape[1] / img.shape[0]
    if in_ratio < min(ratio):
        crop_width = img.shape[1]
        crop_height = int(round(crop_width / min(ratio)))
    elif in_ratio > max(ratio):
        crop_height = img.shape[0]
        crop_width = int(round(crop_height * max(ratio)))
    else:
        # Take whole image if aspect ratio doesn't match.
        crop_height = img.shape[0]
        crop_width = img.shape[1]

    i = (img.shape[0] - crop_height) // 2
    j = (img.shape[1] - crop_width) // 2

    h_start = i * 1.0 / (img.shape[0] - crop_height + 1e-10)
    w_start = j * 1.0 / (img.shape[1] - crop_width + 1e-10)

    # Get unnormalized crop bbox coordinates.
    x1, y1, x2, y2 = F.get_random_crop_coords(
        height, width, crop_height, crop_width, h_start, w_start
    )
    return (x1, y1, x2, y2)


def get_random_crop_coords_bbox_cover(
    img: np.ndarray,
    reference_coords: List[Tuple[int, int, int, int]],
    min_areacover: float = 0.2,
    scale: Tuple[float, float] = (0.2, 1.0),
    ratio: Tuple[float, float] = (3 / 4, 4 / 3),
):
    r"""
    Compute ``(x1, y1, x2, y2)`` coordinates of a random crop satisfying the
    following constraints:
        1. Must cover ``min_areacover`` area of crop defined by each coordinates
           in ``reference_coords`` list.
        2. Must have ``crop area / image area`` within ``scale`` limits.
        3. Must have aspect ratio within ``ratio`` limits.
    """
    if isinstance(reference_coords, tuple):
        reference_coords = [reference_coords]

    # Rename for consistent naming with ``get_random_crop_coords_mask_cover``.
    all_constraint_coords = reference_coords

    # Calculate the area of input image.
    image_height, image_width = img.shape[0], img.shape[1]
    image_area = image_height * image_width

    # Adjust ``min_areacover`` if area of biggest crop is too large. Else this
    # the generated crop may be too small to cover ``min_areacover`` area of
    # the biggest crop despite completely lying inside it.
    biggest_ref_crop_area = max([crop_area(c) for c in all_constraint_coords])
    min_areacover = min(
        min_areacover, scale[0] * image_area / biggest_ref_crop_area
    )
    # Make at most 10 attempts to do a random crop which also satisfies area
    # cover condition.
    for _attempt in range(10):
        target_area = random.uniform(*scale) * image_area
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        aspect_ratio = math.exp(random.uniform(*log_ratio))

        crop_height = int(round(math.sqrt(target_area / aspect_ratio)))
        crop_width = int(round(math.sqrt(target_area * aspect_ratio)))

        if 0 < crop_height <= img.shape[0] and 0 < crop_width <= img.shape[1]:
            i = random.randint(0, img.shape[0] - crop_height)
            j = random.randint(0, img.shape[1] - crop_width)

            h_start = i * 1.0 / (img.shape[0] - crop_height + 1e-10)
            w_start = j * 1.0 / (img.shape[1] - crop_width + 1e-10)

            # Get random crop coordinates.
            x1, y1, x2, y2 = F.get_random_crop_coords(
                image_height, image_width, crop_height, crop_width, h_start, w_start
            )
            # Check if this crop satisfies constraints with all reference crops.
            all_constraints_satisfied: bool = True
            for c_coords in all_constraint_coords:
                # Calculate the area of constraint crop (in square pixels).
                c_area = crop_area(c_coords)

                # Determine the coordinates of intersection between computed
                # crop coordinates and this constraint crop coordinates.
                inter_x1 = max(x1, c_coords[0])
                inter_y1 = max(y1, c_coords[1])
                inter_x2 = min(x2, c_coords[2])
                inter_y2 = min(y2, c_coords[3])
                inter_bbox = (inter_x1, inter_y1, inter_x2, inter_y2)

                # Compute area of mask inside intersection and check if this
                # covers enough area of mask inside this constraint crop.
                area_of_intersection = crop_area(inter_bbox)

                # Crop is successful if it satisfies area cover constraint.
                if area_of_intersection / c_area < min_areacover:
                    all_constraints_satisfied = False

            if all_constraints_satisfied:
                return (x1, y1, x2, y2)

    # If random crop could not work, then fallback to a crop which either
    # is entirely covering the reference crop.
    ref_x1, ref_y1, ref_x2, ref_y2 = all_constraint_coords[0]
    x1 = ref_x1 - random.randint(0, ref_x1)
    y1 = ref_y1 - random.randint(0, ref_y1)
    x2 = ref_x2 + random.randint(0, image_width - ref_x2)
    y2 = ref_y2 + random.randint(0, image_height - ref_y2)

    # Now this crop must satisfy the `scale` and `ratio` limits. We assume
    # that the reference crop also satisfies `scale` and `ratio` limits.
    # Since second crop is larger than reference crop, it will satisfy scale.
    aspect_ratio = (x2 - x1) / (y2 - y1)
    if aspect_ratio > ratio[1]:
        # Too wide, shrink width.
        x2 = x1 + int((y2 - y1) * ratio[1])
        # Translate x coordinates if falling short of 100% cover.
        if x2 < ref_x2:
            # Will definitely satisfy aspect ratio and cover 100% area.
            x1 += (ref_x2 - x2)
            x2 = ref_x2

    elif aspect_ratio < ratio[0]:
        # Too long, shrink height.
        y2 = y1 + int((x2 - x1) / ratio[0])
        # Translate y coordinates if falling short of 100% cover.
        if y2 < ref_y2:
            # Will definitely satisfy aspect ratio and cover 100% area.
            y1 += (ref_y2 - y2)
            y2 = ref_y2

    return (x1, y1, x2, y2)


def get_random_crop_coords_mask_cover(
    img: np.ndarray,
    reference_mask: np.ndarray,
    other_coords: Union[Tuple, List[Tuple]] = [],
    min_areacover: float = 0.2,
    scale: Tuple[float, float] = (0.2, 1.0),
    ratio: Tuple[float, float] = (3 / 4, 4 / 3),
):
    r"""
    Compute ``(x1, y1, x2, y2)`` coordinates of a random crop satisfying the
    following constraints:
        1. Must cover ``min_areacover`` area of the full ``reference_mask``.
        2. Must cover ``min_areacover`` area of the ``reference_mask`` inside
           each boxes in ``other_coords`` (list of ``(x1, y1, x2, y2)``).
        3. Must have ``crop area / image area`` within ``scale`` limits.
        4. Must have aspect ratio within ``ratio`` limits.
    """
    # If ``other_coords`` is a tuple representing only one box, make it a list
    # of tuples.
    if not isinstance(other_coords, list):
        other_coords = [other_coords] if other_coords is not None else []

    # Add the bounding box of the reference mask to the list of coordinates.
    all_constraint_coords = [bbox_from_mask(reference_mask)] + other_coords
    
    # Calculate the area of input image.
    image_height, image_width = img.shape[0], img.shape[1]
    image_area = image_height * image_width

    # Calculate the area and centroid coordinates (float, float) of mask.
    reference_area = mask_area(reference_mask)
    ref_centroid_y, ref_centroid_x = center_of_mass(reference_mask)

    # Adjust ``min_areacover`` if reference mask area is too large. Else this
    # the generated crop may be too small to cover ``min_areacover`` area of
    # mask despite completely lying inside it.
    min_areacover = min(
        min_areacover, scale[0] * image_area / reference_area
    )
    # Make at most 10 attempts to do a random crop which also satisfies area
    # cover condition.
    for _attempt in range(10):
        target_area = random.uniform(*scale) * image_area
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        aspect_ratio = math.exp(random.uniform(*log_ratio))

        crop_height = int(round(math.sqrt(target_area / aspect_ratio)))
        crop_width = int(round(math.sqrt(target_area * aspect_ratio)))

        if 0 < crop_height <= img.shape[0] and 0 < crop_width <= img.shape[1]:

            # Sample the center coordinates of the crop from truncated gaussian
            # with mean = centroid of reference mask, stdev = image dim.
            crop_center_x = _trunc_gauss(
                mu=ref_centroid_x,
                sigma=image_width,
                lower=crop_width / 2,
                upper=image_width - crop_width / 2
            )
            crop_center_y = _trunc_gauss(
                mu=ref_centroid_y,
                sigma=image_height,
                lower=crop_height / 2,
                upper=image_height - crop_height / 2
            )
            # Get random crop coordinates.
            x1 = int(crop_center_x - crop_width / 2)
            y1 = int(crop_center_y - crop_height / 2)
            x2 = int(crop_center_x + crop_width / 2)
            y2 = int(crop_center_y + crop_height / 2)
            # Check if crop satisfies constraints with all constraint crops.
            all_constraints_satisfied: bool = True
            for c_coords in all_constraint_coords:
                # Calculate the mask area inside coordinates.
                c_area = mask_area(reference_mask, c_coords)
                # Determine the coordinates of intersection between computed
                # crop coordinates and this constraint crop coordinates.
                inter_x1 = max(x1, c_coords[0])
                inter_y1 = max(y1, c_coords[1])
                inter_x2 = min(x2, c_coords[2])
                inter_y2 = min(y2, c_coords[3])
                inter_bbox = (inter_x1, inter_y1, inter_x2, inter_y2)

                # Compute area of mask inside intersection and check if this
                # covers enough area of mask inside this constraint crop.
                area_of_intersection = mask_area(reference_mask, inter_bbox)

                
                # Crop is successful if it satisfies area cover constraint.
                if area_of_intersection / c_area < min_areacover:
                    all_constraints_satisfied = False

            if all_constraints_satisfied:
                return (x1, y1, x2, y2)

    # If random crop could not work, then fallback to a crop which either
    # is entirely covering the reference crop.
    ref_x1, ref_y1, ref_x2, ref_y2 = all_constraint_coords[0]


    x1 = ref_x1 - random.randint(0, ref_x1)
    y1 = ref_y1 - random.randint(0, ref_y1)
    x2 = ref_x2 + random.randint(0, image_width - ref_x2)
    y2 = ref_y2 + random.randint(0, image_height - ref_y2)

    # Now this crop must satisfy the `scale` and `ratio` limits. We assume
    # that the reference crop also satisfies `scale` and `ratio` limits.
    # Since second crop is larger than reference crop, it will satisfy scale.
    aspect_ratio = (x2 - x1) / (y2 - y1)
    if aspect_ratio > ratio[1]:
        # Too wide, shrink width.
        x2 = x1 + int((y2 - y1) * ratio[1])
        # Translate x coordinates if falling short of 100% cover.
        if x2 < ref_x2:
            # Will definitely satisfy aspect ratio and cover 100% area.
            x1 += (ref_x2 - x2)
            x2 = ref_x2

    elif aspect_ratio < ratio[0]:
        # Too long, shrink height.
        y2 = y1 + int((x2 - x1) / ratio[0])
        # Translate y coordinates if falling short of 100% cover.
        if y2 < ref_y2:
            # Will definitely satisfy aspect ratio and cover 100% area.
            y1 += (ref_y2 - y2)
            y2 = ref_y2

    return (x1, y1, x2, y2)


def get_reference_crop_covering_mask(
    img_or_reference_mask: np.ndarray,
    reference_coords: Tuple[int, int, int, int],
    other_coords: Union[Tuple, List[Tuple]] = [],
) -> np.ndarray:
    r"""
    Get a mask of the crop defined by ``reference_coords`` which shows the
    region covered by this crop in the ``img_or_reference_mask``, intersecting
    with all crops in ``other_coords``.
    This method assumes that all crops (``reference_coords`` and each tuple in
    ``other_coords``) already have overlap with each other.
    Parameters
    ----------
    img_or_reference_mask: np.ndarray
    reference_coords: Tuple[int, int, int, int]
    other_coords: Union[Tuple, List[Tuple]]
    """

    # Convert ``img_or_reference_mask`` to an array of 1 and 0 integers. If
    # it has range [0-1] uint8, it would be a reference (GT) mask, else it
    # would have range [0-255] uint8 representing the image.
    if np.amax(img_or_reference_mask) == 1:
        height, width = img_or_reference_mask.shape

        mask = img_or_reference_mask.astype(np.uint8)
    else:
        # Just make an array or 1's for now, and later turn everything outside
        # area of intersection to 0's.
        height, width, _ = img_or_reference_mask.shape
        mask = np.ones_like(img_or_reference_mask, (height, width), dtype=np.uint8)

    # temporary fixes
    ref_x1, ref_y1, ref_x2, ref_y2 = reference_coords

    
    # If ``other_coords`` is a tuple representing only one box, make it a list
    # of tuples.
    if not isinstance(other_coords, list):
        other_coords = [other_coords] if other_coords is not None else []

    # Find an intersection box among all boxes.
    all_x1s = [ref_x1] + [other[0] for other in other_coords]
    all_y1s = [ref_y1] + [other[1] for other in other_coords]
    all_x2s = [ref_x2] + [other[2] for other in other_coords]
    all_y2s = [ref_y2] + [other[3] for other in other_coords]

    # Find intersection coordinates and shift them by treating
    # (ref_x1, ref_y1) as origin.
    inter_x1 = max(all_x1s) - ref_x1
    inter_y1 = max(all_y1s) - ref_y1
    inter_x2 = min(all_x2s) - ref_x1
    inter_y2 = min(all_y2s) - ref_y1

    # Check if all boxes did intersect.
    # Take image crop using reference coordinates. Wrap in np.array
    # constructor to make a separate copy.
    mask = np.array(F.crop(mask, *reference_coords))

    # Make region outside of intersection area in mask as black.
    mask[:inter_y1, :] = 0
    mask[:, :inter_x1] = 0
    mask[inter_y2:, :] = 0
    mask[:, inter_x2:] = 0
    return mask


def bbox_from_mask(mask: np.ndarray):
    r"""
    Compute bounding box of the input mask, assumes mask is not all ``False``.
    Parameters
    ----------
    mask: np.ndarray
        Boolean mask of shape ``(height, width)`` with masked pixels ``True``.
    Returns
    -------
    Tuple[int, int, int, int]
        Absolute coordinates of bounding box, ``(x1, y1, x2, y2)``.
    """
    indices_height, indices_width = np.where(mask)
    top_left = (indices_width.min(), indices_height.min())
    bottom_right = (indices_width.max() + 1, indices_height.max() + 1)

    return (*top_left, *bottom_right)


def crop_area(coords: Tuple[int, int, int, int]) -> int:
    r"""
    Compute area of a crop defined by absolute coords.
    Parameters
    ----------
    coords: Tuple[int, int, int, int]
        A crop specified by absolute coordinate of top-left and bottom-right
        ``(x1, y1, x2, y2)``.
    Returns
    -------
    int
        Area of bounding box.
    """
    return (coords[2] - coords[0]) * (coords[3] - coords[1])


def mask_area(mask: np.ndarray, bbox: Tuple[int, int, int, int] = None) -> int:
    r"""
    Compute area of a binary mask, either full (default) or optionally
    enclosed by a bounding box.
    Parameters
    ----------
    mask: np.ndarray
        Boolean mask of shape ``(height, width)`` with masked pixels ``True``.
    bbox: Tuple[int, int, int, int], optional (default = None)
        An optional bounding box specified by absolute coordinates of top-left
        and bottom-right ``(x1, y1, x2, y2)``. Default is ``None``, where full
        area of mask is computed.
    Returns
    -------
    int
        Area of mask (number of ``True`` values).
    """
    if bbox is None:
        return mask.sum()
    else:
        x1, y1, x2, y2 = bbox
        return mask[y1:y2, x1:x2].sum()


def _trunc_gauss(mu: float, sigma: float, lower: float, upper: float) -> float:
    r"""
    Sample from truncated normal distribution ``(mu, sigma)`` within the
    given upper and lower limits.
    Parameters
    ----------
    mu: float
        Mean of the normal distribution.
    sigma: float
        Standard deviation of the normal distribution.
    lower: float
        Lower limit of the truncated normal distribution.
    upper: float
        Upper limit of the truncated normal distribution.
    Returns
    -------
    float
        An observation sampled from the given distribution.
    """
    if lower==upper:
        return lower
    while True:
        number = random.gauss(mu, sigma)
        if number >= lower and number <= upper:
            break

    return number