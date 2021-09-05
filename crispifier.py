import torch
import sys
import sqlite3
import glob
import os.path
from hashlib import sha256
from io import BytesIO

sys.path.append("./ESRGAN")
from ESRGAN import architecture as arch
from ut_parser import package
from ut_parser.module_types import idx
from ut_parser.ut_objects import texture

mipmap_subcon = texture.subcons[-1].subcon.subcon
import numpy as np
from PIL import Image, ImageTransform, ImageFilter


def hash_image(image):
    sha = sha256()
    sha.update(image.tobytes())
    return sha.digest().hex()


def get_image_array(obj):
    img = obj.object.img_data[0]
    width = img.width
    height = img.height
    return np.array(list(img.imgbytes)).reshape(height, width).astype(np.uint8)


def get_palette_array(obj):
    return np.array(
        [[color.R, color.G, color.B, color.A] for color in obj.object.colors]
    ).astype(np.uint8)


def get_model_model_hash(path):
    model = arch.RRDB_Net(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(path))
    model.eval()
    model = model.to(torch.device("cpu"))
    hash = sha256()
    with open(path, "rb") as f:
        hash.update(f.read())
    return model, hash.digest().hex()


def get_named_object(name, construct):
    return [obj for obj in construct.export_objects if obj.obj_name == name][0]


def get_named_property(object, name):
    return [prop for prop in object.object.property if prop.prop_name == name][0]


def get_palette(texture, construct):
    palette_prop = get_named_property(texture, "Palette")
    palette_obj = construct.export_objects[palette_prop.value.data.index - 1]
    palette_array = get_palette_array(palette_obj)
    return palette_array


def get_image(image_array, palette_array):
    return Image.fromarray(palette_array[image_array])


def get_processable(img, device):
    return (
        torch.from_numpy(
            np.transpose(np.array(img)[:, :, [2, 1, 0]], (2, 0, 1)) / 255.0
        )
        .float()
        .unsqueeze(0)
        .to(device)
    )


def get_output_array(output):
    return (
        (output.numpy()[[2, 1, 0], :, :].transpose((1, 2, 0)) * 255)
        .round()
        .astype(np.uint8)
    )


def get_upscaled_from_db(image, model_hash, cursor):
    image_hash = hash_image(image)
    results = cursor.execute(
            """SELECT
                output_data
            FROM
                outputs
            WHERE
                image_hash = ?
                AND model_hash = ?
            """,
            (image_hash, model_hash),
        ).fetchone()
    i = results[0]
    return Image.open(BytesIO(i))


def put_upscaled_in_db(image, upscaled, name, model_hash, cursor):
    image_hash = hash_image(image)
    stream = BytesIO()
    image.save(stream, "png")
    stream.seek(0)
    cursor.execute(
        """INSERT OR IGNORE INTO 
            inputs(image_hash, image_name, input_data)
        VALUES
            (:image_hash, :name, :image_file)
        """,
        dict(image_hash=image_hash, name=name, image_file=stream.read(),),
    )
    stream = BytesIO()
    upscaled.save(stream, "png")
    stream.seek(0)
    cursor.execute(
        """INSERT OR IGNORE INTO
            outputs(model_hash, image_hash, output_data)
        VALUES
            (:model_hash, :image_hash, :output_data)
        """,
        dict(model_hash=model_hash, image_hash=image_hash, output_data=stream.read(),),
    )


def upscale_image_gan(image, model, device):
    with torch.no_grad():
        input = get_processable(image, device)
        output = model(input).data.squeeze().float().cpu().clamp_(0, 1)
        output_array = get_output_array(output)
        return Image.fromarray(output_array)


def upscale_image(
    image, name, model, device, factor, db_filename=None, model_hash=None
):
    """
        image: a PIL Image
        name: the name of the image
        model: a torch GAN model for upscaling
        device: a torch device
        factor: the scale factor of the image
        db_filename: optional path for a cached sqlite database of upscaled images
    """
    if db_filename is None:
        upscaled = upscale_image_gan(image, model, device)
    else:
        with sqlite3.connect(db_filename) as conn:
            cursor = conn.cursor()
            try:
                upscaled = get_upscaled_from_db(image, model_hash, cursor)
            except (IndexError, TypeError):
                upscaled = upscale_image_gan(image, model, device)
                put_upscaled_in_db(image, upscaled, name, model_hash, cursor)
        conn.close()
    return upscaled.resize((image.width * factor, image.height * factor))


def palettize(image, palette_array):
    return (
        np.argmin(
            np.linalg.norm(
                np.array(image).reshape(-1, 3)[:, np.newaxis, :]
                - palette_array[np.newaxis, :, :3],
                axis=2,
            ),
            axis=1,
        )
        .reshape(image.width, image.height)
        .astype(np.uint8)
    )


def palettize(image, palette_array):
    palette_im = Image.new("P", (16, 16))
    palette_im.putpalette(palette_array[:, :3].ravel())
    return image.convert("RGB").quantize(palette=palette_im)


def get_image_dict(image, start_pos=0):
    bytes = image.tobytes()
    block_size = len(bytes)
    return dict(
        pos_after=start_pos + block_size + len(idx.build(block_size)),
        block_size=block_size,
        imgbytes=bytes,
        width=image.width,
        height=image.height,
        widthbits=image.width.bit_length() - 1,
        heightbits=image.height.bit_length() - 1,
    )


def generate_mipmaps(image, start_pos=0):
    count = max(image.width.bit_length(), image.height.bit_length())
    out = []
    for i in range(count):
        width = max(image.width >> i, 1)
        height = max(image.height >> i, 1)
        img_dict = get_image_dict(image.resize((width, height)), start_pos=start_pos)
        img_bytes = mipmap_subcon.build(img_dict)
        start_pos += len(img_bytes)
        out.append(img_dict)
    return out


def get_mipmap_length(mipmaps):
    return sum(len(mipmap_subcon.build(mipmap)) for mipmap in mipmaps)


def scale_image_properties(props, scale_factor):
    for property in props:
        name = property.prop_name
        if name in ("UBits", "VBits"):
            property.value.data += scale_factor.bit_length()
        elif name in ("USize", "VSize", "UClamp", "VClamp"):
            property.value.data *= scale_factor


def get_mask_index(palette_array):
    return np.argwhere(np.all(palette_array == [255, 0, 255, 255], axis=-1)).item()


def get_mask(palettized_image, mask_index):
    return np.array(palettized_image) == mask_index


def scale_mask(mask_array, factor, blur=2, grow=3, shrink=3, threshold=100):
    mask_image = Image.fromarray(mask_array * np.uint8(255))
    scaled = mask_image.resize((mask_image.width * factor, mask_image.height * factor))
    return (
        np.array(
            scaled.filter(ImageFilter.GaussianBlur(blur))
            .filter(ImageFilter.MaxFilter(shrink))
            .filter(ImageFilter.MinFilter(grow))
        )
        > threshold
    )


def mask_image(image, mask, mask_value):
    """
    image: a PIL Image with palletized colors
    mask: a 2d np boolean array of the size of the image
    mask_value: the color of the mask
    """
    image_array = np.array(image)
    image_array[mask] = mask_value
    image.putdata(image_array.ravel())


def get_upscaled_texture(
    package,
    obj,
    model,
    device,
    factor,
    db_filename,
    blur=4,
    grow=3,
    shrink=3,
    threshold=128,
    model_hash=None,
):
    palette_array = get_palette(obj, package)
    image_array = get_image_array(obj)
    image = get_image(image_array, palette_array)
    palettized_image = palettize(image, palette_array)
    upscaled = upscale_image(
        image,
        obj.obj_name,
        model,
        device,
        factor,
        db_filename=db_filename,
        model_hash=model_hash,
    )
    palettized_upscaled = palettize(upscaled, palette_array)
    try:
        mask_index = get_mask_index(palette_array)
    except:
        mask_index = -1
    mask = get_mask(image_array, mask_index)
    scaled_mask = scale_mask(mask, factor, blur=4, grow=3, shrink=3, threshold=128)
    mask_image(palettized_upscaled, scaled_mask, mask_index)
    return palettized_upscaled


def rescale_package(
    package,
    model,
    device,
    factor,
    model_hash,
    db_filename=None,
    blur=4,
    grow=3,
    shrink=3,
    threshold=128,
    
):
    offset = 0
    for obj, header in zip(package.export_objects, package.export_headers):
        header.serial_offset += offset
        obj.serial_offset += offset
        if obj.cls_name == "Texture":
            mipmap_shift = (
                obj.object.img_data[0].pos_after
                - obj.object.img_data[0].block_size
                - len(idx.build(obj.object.img_data[0].block_size))
                + offset
            )
            original_length = get_mipmap_length(obj.object.img_data)
            upscaled = get_upscaled_texture(
                package,
                obj,
                model,
                device,
                factor,
                db_filename,
                blur=blur,
                grow=grow,
                shrink=shrink,
                threshold=threshold,
                model_hash=model_hash,
            )
            mipmaps = generate_mipmaps(upscaled, start_pos=mipmap_shift)
            new_length = get_mipmap_length(mipmaps)
            delta = new_length - original_length
            offset += delta
            obj.serial_size += delta
            header.serial_size += delta
            obj.object.mip_map_count = len(mipmaps)
            obj.object.img_data = [
                mipmap_subcon.parse(mipmap_subcon.build(mipmap)) for mipmap in mipmaps
            ]
            scale_image_properties(obj.object.property, factor)
        elif obj.cls_name.lower() in {"lodmesh", "mesh"}:
            obj.object.pos0 += offset
            obj.object.pos1 += offset
            obj.object.pos2 += offset
            obj.object.pos3 += offset
    package.header.export_offset += offset
    package.header.import_offset += offset
