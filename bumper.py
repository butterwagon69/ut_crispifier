from pathlib import Path
from orm import Input, Bump
from sqlalchemy import select
from sqlalchemy.orm.exc import NoResultFound
from io import BytesIO
import numpy as np
from PIL import Image, ImageTransform, ImageFilter
from wand.image import Image as WandImage
import torch
import sys
sys.path.append("./ESRGAN")
from ESRGAN import architecture as arch


def load_model(model_path, device):
    state_dict = torch.load(model_path)
    model = arch.RRDB_Net(
        3,
        3,
        32,
        12,
        gc=32,
        upscale=1,
        norm_type=None,
        act_type='leakyrelu',
        mode='CNA',
        res_scale=1,
        upsample_mode='upconv'
    )
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    return model.to(device)

def process(img, model, device):
    img = img * 1. / np.iinfo(img.dtype).max
    img = img[:, :, [2, 1, 0]]
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    output = model(img_LR).data.squeeze(0).float().cpu().clamp_(0, 1).numpy()
    output = output[[2, 1, 0], :, :]
    output = np.transpose(output, (1, 2, 0))
    output = (output * 255.).round()
    return output

def image_to_bytes(image):
    stream = BytesIO()
    image.save(stream, "png")
    stream.seek(0)
    return stream.read()

def generate_bumps(
    db_session,
    model_paths,
    output_dir,
    device,
):
    output_dir = Path(output_dir)
    models = [
        load_model(model_path, device) for model_path in model_paths
    ]
    for input_image in db_session.query(Input).all():
        if input_image.image_name.startswith("FlatFXTex"):
            continue
        if not input_image.source_file or input_image.source_file == "MISSING":
            continue
        image = Image.open(BytesIO(input_image.input_data))
        stmt = select(
            Bump
        ).where(Bump.image_hash == input_image.image_hash)
        try:
            bump, = db_session.execute(stmt).one()
        except NoResultFound:
            rlts = [process(np.array(image), model, device) for model in models]
            normal = Image.fromarray(rlts[0].astype(np.uint8))
            roughness = Image.fromarray(rlts[1][:, :, 1].astype(np.uint8))
            displacement = Image.fromarray(rlts[1][:, :, 0].astype(np.uint8))
            bump = Bump(
                image_hash=input_image.image_hash,
                normal=image_to_bytes(normal),
                roughness=image_to_bytes(roughness),
                displacement=image_to_bytes(displacement),
            )
            db_session.add(bump)
            db_session.commit()
        normal = WandImage(file=BytesIO(bump.normal))
        roughness = WandImage(file=BytesIO(bump.roughness))
        displacement = WandImage(file=BytesIO(bump.displacement))
        outdir = (
            output_dir
            / Path(input_image.source_file).stem
            / input_image.group_name
        )
        normal.save(filename=outdir / (input_image.image_name + ".normal.dds"))
        displacement.save(filename=outdir / (input_image.image_name + ".height.dds"))
        break
