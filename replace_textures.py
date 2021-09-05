"""Replace textures in the resources folder using the textures database."""
import crispifier
from pathlib import Path
from itertools import chain

if __name__ == "__main__":
    root = Path("../resources/")
    dest = Path("../release/System/")
    db_path = Path("textures.db")

    exceptions = []

    for path in chain(root.glob("*.u"), root.glob("*.utx")):
        output_path = dest / path.relative_to(root)
        try:
            with open(path, "rb") as f:
                package = crispifier.package.parse(f.read())
                crispifier.rescale_package_db(
                    package,
                    4,
                    db_path,
                    model_hash=None,
                    blur=2,
                    grow=1,
                    shrink=1,
                    threshold=128,
                )
            with open(output_path, "wb") as f:
                f.write(crispifier.package.build(package))
            print(f"Processed {path.name} sucessfully")
        except BaseException as e:
            exceptions.append((path, e))
            raise e
            print(f"{path.name} failed to process sucessfully!")

    if exceptions:
        for path, e in exceptions:
            print(f"Error for {path.name}:\n")
            print(e)
