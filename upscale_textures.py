"Perform texture upscaling on each texture in the database."


import crispifier
from PIL import Image
import torch
import sqlite3
from itertools import chain
from pathlib import Path
from io import BytesIO


if __name__ == "__main__":

    device = torch.device("cpu")
    candidates = list(Path("./models/").glob("*.pth"))
    db_path = Path("textures.db")
    root = Path("../resources/")
    if not db_path.exists():
        print("Creating database!")
        with sqlite3.connect(db_path) as conn, open("schema.sql") as f:
            conn.cursor().executescript(f.read())
        conn.close()
    models_hashes = [crispifier.get_model_model_hash(path) for path in candidates]

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA FOREIGN_KEYS = ON")
        for path, (model, model_hash) in zip(candidates, models_hashes):
            print(path)
            print(type(model))
            cursor.execute(
                """INSERT OR IGNORE INTO
                models(model_hash, model_name)
                values(:model_hash, :model_name)""",
                dict(model_hash=model_hash, model_name=str(path)),
            )
        for resource_path in chain(root.glob("*.u"), root.glob("*.utx")):
            print(resource_path)
            try:
                package = crispifier.package.parse_file(resource_path)
            except BaseException as e:
                print(e)
                break
            for texture in (
                    obj for obj in package.export_objects if obj.cls_name == "Texture"
            ):
                try:
                    image = crispifier.get_image(
                        crispifier.get_image_array(texture),
                        crispifier.get_palette(texture, package),
                    )
                except BaseException as e:
                    print(e)
                    break
                try:
                    image_hash = crispifier.hash_image(image)
                except BaseException as e:
                    print(e)
                    break
                palette_array = crispifier.get_palette(texture, package)
                image_arrays = [
                    crispifier.get_image_array(texture, mipmap_index=i)
                    for i in range(texture.object.mip_map_count)
                ]

                stream = BytesIO()
                image.save(stream, "png")
                stream.seek(0)
                image_name = texture.obj_name
                cursor.execute(
                    """INSERT OR IGNORE INTO
                        inputs(
                            image_hash,
                            image_name,
                            input_data
                        )
                    VALUES
                        (
                            :image_hash,
                            :image_name,
                            :input_data
                        )
                    """,
                    dict(
                        image_hash=image_hash,
                        image_name=image_name,
                        input_data=stream.read(),
                    ),
                )
                crispifier.put_mipmap_palette_db(image_arrays, palette_array, cursor)
                for model, model_hash in models_hashes:
                    crispifier.upscale_image(
                        image,
                        image_name,
                        model,
                        device,
                        4,
                        cursor,
                        model_hash=model_hash,
                    )
                print(f"Upscaled {image_name} from {resource_path.name}")
