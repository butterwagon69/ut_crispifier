from io import BytesIO
from pathlib import Path
from hashlib import sha256

import numpy as np
from sqlalchemy import select
from sqlalchemy.sql import and_
from wand.image import Image as WandImage
from PIL import Image, ImageTransform, ImageFilter

from orm import Preference, Output, Input
from crispifier import get_image_array

class ImageNotFoundException(ValueError):
    pass


def get_upscaled_from_orm(image_hash, session, model_hash=None):
    if model_hash is None:
        stmt = select(
            Preference
        ).where(Preference.image_hash == image_hash)
        pref, = session.execute(stmt).one()
        out = pref.output
    else:
        stmt = select(
            Output
        ).where(
            and_(
                Output.image_hash == image_hash,
                Output.model_hash == model_hash,
            )
        )
        out, = session.execute(stmt).one()
    return Image.open(BytesIO(out.output_data))


def scale_mask(mask_array, factor, blur=2, grow=3, shrink=3):
    mask_image = Image.fromarray(mask_array * np.uint8(255))
    scaled = mask_image.resize(
        (int(mask_image.width * factor), int(mask_image.height * factor))
    )
    if blur:
        scaled = scaled.filter(ImageFilter.GaussianBlur(blur))
    if shrink:
        scaled = scaled.filter(ImageFilter.MaxFilter(shrink))
    if grow:
        scaled = scaled.filter(ImageFilter.MinFilter(grow))
    return scaled

def rescale_single(
    db_session,
    factor,
    input_image,
    output_dir,
    blur=1,
    grow=1,
    shrink=1,
    model_hash=None,
):

    output_dir = Path(output_dir)
    image = Image.open(BytesIO(input_image.input_data))
    upscaled = get_upscaled_from_orm(
        input_image.image_hash, db_session, model_hash
    )
    mask = ~np.all(np.array(image) == [255, 0, 255, 255], axis=-1)
    scaled_mask = scale_mask(
        mask, factor, blur=blur, grow=grow, shrink=shrink
    )
    upscaled.putalpha(scaled_mask)
    outfile = (
        output_dir
        / Path(input_image.source_file).stem
        / input_image.group_name
        / (input_image.image_name + ".dds")
    )
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with WandImage.from_array(np.array(upscaled)) as img:
        img.compression = "dxt1"
        img.save(filename=outfile)

def rescale_all(
    db_session,
    factor,
    output_dir,
    blur=1,
    grow=1,
    shrink=1,
    model_hash=None,
):
    output_dir = Path(output_dir)
    for input_image in db_session.query(Input).all():
        if input_image.image_name.startswith("FlatFXTex"):
            continue
        if not input_image.source_file or input_image.source_file == "MISSING":
            continue

        rescale_single(
            db_session,
            factor,
            input_image,
            output_dir,
            blur=blur,
            grow=grow ,
            shrink=shrink,
            model_hash=model_hash,
        )
