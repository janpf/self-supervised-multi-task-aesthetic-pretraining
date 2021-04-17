import hashlib
from edit_image import parameter_range
import math

parameter_range["shadows"]["range"] = [-100, -60, -20, 20, 40, 50, 60, 80, 100]
parameter_range["highlights"]["range"] = [-100, -80, -60, -50, -40, -20, 20, 60, 100]
parameter_range["temperature"]["range"] = [2000, 3000, 5000, 6000, 6500, 7000, 8000, 10000, 14000, 18000]

mapping = dict()
mapping["styles"] = dict()

for style in parameter_range.keys():
    if style == "lcontrast":
        continue
    mapping["styles"][style] = dict()
    mapping["styles"][style]["neg"] = [
        f"{style};{i}" for i in parameter_range[style]["range"] if i < parameter_range[style]["default"]
    ]
    mapping["styles"][style]["pos"] = [
        f"{style};{i}" for i in parameter_range[style]["range"] if i > parameter_range[style]["default"]
    ]

mapping["technical"] = dict()
mapping["technical"]["jpeg_compression"] = dict()
mapping["technical"]["jpeg_compression"]["pos"] = [f"jpeg_compression;{i}" for i in range(1, 6)]
mapping["technical"]["defocus_blur"] = dict()
mapping["technical"]["defocus_blur"]["pos"] = [f"defocus_blur;{i}" for i in range(1, 6)]
mapping["technical"]["motion_blur"] = dict()
mapping["technical"]["motion_blur"]["pos"] = [f"motion_blur;{i}" for i in range(1, 6)]
mapping["technical"]["pixelate"] = dict()
mapping["technical"]["pixelate"]["pos"] = [f"pixelate;{i}" for i in range(1, 6)]
mapping["technical"]["gaussian_noise"] = dict()
mapping["technical"]["gaussian_noise"]["pos"] = [f"gaussian_noise;{i}" for i in range(1, 6)]
mapping["technical"]["impulse_noise"] = dict()
mapping["technical"]["impulse_noise"]["pos"] = [f"impulse_noise;{i}" for i in range(1, 6)]

mapping["composition"] = dict()
mapping["composition"]["rotate"] = dict()
mapping["composition"]["rotate"]["neg"] = [f"rotate;{i}" for i in range(-10, 0, 2)]
mapping["composition"]["rotate"]["pos"] = [f"rotate;{i}" for i in range(0, 11, 2) if i != 0]
mapping["composition"]["hcrop"] = dict()
mapping["composition"]["hcrop"]["neg"] = [f"hcrop;{i}" for i in range(-5, 0)]
mapping["composition"]["hcrop"]["pos"] = [f"hcrop;{i}" for i in range(0, 6) if i != 0]
mapping["composition"]["vcrop"] = dict()
mapping["composition"]["vcrop"]["neg"] = [f"vcrop;{i}" for i in range(-5, 0)]
mapping["composition"]["vcrop"]["pos"] = [f"vcrop;{i}" for i in range(0, 6) if i != 0]
mapping["composition"]["leftcornerscrop"] = dict()
mapping["composition"]["leftcornerscrop"]["neg"] = [f"leftcornerscrop;{i}" for i in range(-5, 0)]
mapping["composition"]["leftcornerscrop"]["pos"] = [f"leftcornerscrop;{i}" for i in range(0, 6) if i != 0]
mapping["composition"]["rightcornerscrop"] = dict()
mapping["composition"]["rightcornerscrop"]["neg"] = [f"rightcornerscrop;{i}" for i in range(-5, 0)]
mapping["composition"]["rightcornerscrop"]["pos"] = [f"rightcornerscrop;{i}" for i in range(0, 6) if i != 0]
mapping["composition"]["ratio"] = dict()
mapping["composition"]["ratio"]["neg"] = [f"ratio;{i}" for i in range(-5, 0)]
mapping["composition"]["ratio"]["pos"] = [f"ratio;{i}" for i in range(0, 6) if i != 0]

mapping["change_steps"] = dict()
for distortion in ["styles", "technical", "composition"]:
    mapping["change_steps"][distortion] = dict()
    for parameter in mapping[distortion]:
        mapping["change_steps"][distortion][parameter] = dict()
        for polarity in mapping[distortion][parameter]:
            if len(mapping[distortion][parameter][polarity]) > 0:
                mapping["change_steps"][distortion][parameter][polarity] = 1 / len(
                    mapping[distortion][parameter][polarity]
                )

mapping["all_changes"] = ["original"]

mapping["styles_changes"] = []
mapping["technical_changes"] = []
mapping["composition_changes"] = []

for _, v in mapping["styles"].items():
    for _, polarity in v.items():
        mapping["all_changes"].extend(polarity)
        mapping["styles_changes"].extend(polarity)

for _, v in mapping["technical"].items():
    for _, polarity in v.items():
        mapping["all_changes"].extend(polarity)
        mapping["technical_changes"].extend(polarity)

for _, v in mapping["composition"].items():
    for _, polarity in v.items():
        mapping["all_changes"].extend(polarity)
        mapping["composition_changes"].extend(polarity)


def filename2path(filename: str) -> str:
    threedirs = hashlib.sha256(filename.encode("utf-8")).hexdigest()[:3]
    return "/".join(list(threedirs) + [filename])


def rotatedRectWithMaxArea(w: int, h: int, angle: float):
    """
    Given a rectangle of size wxh that has been rotated by 'angle',
    computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    https://stackoverflow.com/a/16778797/6388328
    """
    angle = math.radians(angle)
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return wr, hr
