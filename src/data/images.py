"""Image decoding for the distillation dataloader.

Every training step decodes `2 * batch_size` JPEGs from disk and shrinks them to
the model's input resolution. At `--image_resolution 448` against MMEB's
originals that is throwing away most of the pixels the JPEG decoder just spent
its time producing.

Two things here:

* :func:`decode_image` gives libjpeg the target size *before* decoding, via
  ``Image.draft``. libjpeg can emit a DCT-scaled image at 1/2, 1/4 or 1/8 of the
  stored resolution for a fraction of the work, so a 1600x1200 source headed for
  448 is decoded at 400x300 and then resized from there. Same output within
  resampling error, a fraction of the decode. It is a no-op for formats that
  cannot do it (PNG), so it is always safe to ask.
* :func:`resize_to_budget` replaces a resize that ignored the aspect ratio.

Both are pure functions of the file and the resolution setting, which is what
lets `--teacher_embedding_cache` key on the dataset index. Anything added here
that is not deterministic invalidates that cache.
"""

from PIL import Image

# Below this the vision towers' patch embedding has nothing to work with, and
# some processors divide by the smaller side.
MIN_SIZE = 16

_NAMED_RESOLUTIONS = {"high": 1344, "mid": 672, "low": 448}
_DEFAULT_MAX_DIM = 1344


def resolve_target(resolution, max_dim=_DEFAULT_MAX_DIM):
    """Longest-side budget in pixels for a ``--image_resolution`` value.

    Accepts the three named presets and a bare pixel count -- the presets cannot
    express every setting the papers use (LLaVA-OneVision trains at 336, EM-KD
    with LLaVA-OneVision at 128).
    """
    if resolution in _NAMED_RESOLUTIONS:
        return _NAMED_RESOLUTIONS[resolution]
    try:
        return int(resolution)
    except (TypeError, ValueError):
        return max_dim


def resize_to_budget(image, target_max, keep_aspect=False):
    """Shrink ``image`` so its longest side is at most ``target_max``.

    Args:
        keep_aspect: if True, scale both sides by the same factor. If False
            (the default) squash to a ``target_max`` square, which is what this
            code has always done -- every checkpoint and every teacher embedding
            cache in the repo was produced that way, so flipping the default
            would silently change what an existing command line means. See
            ``--image_keep_aspect_ratio``.

    Images already inside the budget are returned untouched.
    """
    if image is None:
        return None
    width, height = image.size
    if max(width, height) <= target_max:
        return image
    if not keep_aspect:
        return image.resize((target_max, target_max))
    scale = target_max / max(width, height)
    return image.resize(
        (max(MIN_SIZE, round(width * scale)), max(MIN_SIZE, round(height * scale)))
    )


def pad_to_min_size(image):
    """Centre a too-small image on a black canvas of at least ``MIN_SIZE``."""
    width, height = image.size
    if width >= MIN_SIZE and height >= MIN_SIZE:
        return image
    new_width, new_height = max(width, MIN_SIZE), max(height, MIN_SIZE)
    canvas = Image.new(image.mode, (new_width, new_height), (0, 0, 0))
    canvas.paste(image, ((new_width - width) // 2, (new_height - height) // 2))
    return canvas


def decode_image(path, target_max=None):
    """Open one image file as RGB, decoding no larger than necessary.

    Args:
        path: file to open.
        target_max: longest side this image is headed for, or None if it will be
            used at full resolution. Only a hint -- ``draft`` picks a DCT scale
            at or above the request, so the result is never smaller than asked
            for and :func:`resize_to_budget` still does the exact resize.

    Returns:
        An RGB ``Image``, padded up to :data:`MIN_SIZE` if the source was tiny.
    """
    image = Image.open(path)
    if target_max:
        # Ask for a box rather than the exact target: draft only halves, so it
        # lands on the first scale that is still >= the request.
        image.draft("RGB", (target_max, target_max))
    image = image.convert("RGB")
    return pad_to_min_size(image)
