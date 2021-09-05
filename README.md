README
======


About
-----

This is a set of scripts that upscale textures in Unreal Engine
games using [ESRGAN](https://esrgan.readthedocs.io/en/latest/index.html).


Dependencies
------------

This uses [Pipenv](https://pipenv.pypa.io/en/latest/) to manage Python
dependencies. There is also a [Jupyter Notebook](https://jupyter.org/) 
to provide a simple UI.


Principle of Operation
----------------------

The user provides game resource files and ESRGAN models. The
[ut_parser](https://github.com/butterwagon69/ut_parser) library parses each file
and extracts textures and associated metadata to a Sqlite database. A script
uses each provided ESRGAN model to upscale each image and save the result in the
Sqlite database. The user then selects the best-looking model for each image 
and saves their selection in the Sqlite database. Finally, a script replaces
each texture in the game resource file with the upscaled version the user
selected.

Use
---

0. Install the Pipenv environment. See Pipenv docs.
1. Run `pipenv run python upscale_textures.py` to upscale all textures.
   This command can take a long time to run: Each image may take a few 
   minutes to scale for each model; multiply that by possibly thousands
   of images and a dozen models and you will see multiple-day runtimes.
2. Run the Jupyter Notebook `PicturePicker.ipynb` and select the best-looking
   model for each image. This is a manual process but could be automated
   if you know something about which model is a good choice for each 
   image.
3. Run `pipenv run python replace_textures.py` to replace textures with
   upscaled versions. Upscaled game files will be saved to the `release` folder.

Known Issues
------------

1. Texture replacement in maps isn't easy. Here's the process:
    1. Upscale all `.utx` files.
    2. For each map, iterate through each surface and change the texture
       scale and offset using `ut_parser`. This fits the new, high resolution
       textures to the level geometry, but screws up the lighting.
    3. Open the level editor, open each level, and rebuild it.
   This means if you want to rescale all level textures for a game, you have
   to release modified versions of each map. I'm not personally OK doing that.
2. Mipmaps in level textures don't work right now. I don't know how to fix this.
3. Transparencies get messed up. Transparency in old Unreal Engine games
   like Deus Ex is handled with a magenta mask. Your GAN model of choice will
   see this magenta color as part of the image and bleed it into the edges.
   Then, you'll have weird off-magenta artifacts in all your transparent textures
   in-game. This can be mitigated by re-masking each texture after GAN upscaling,
   but the boundaries lose fidelity.
4. You have to upscale everything to upscale anything. I'm not sure why right
   now, but the game won't start if only some images are upscaled.
