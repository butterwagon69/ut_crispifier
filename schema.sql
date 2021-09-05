CREATE TABLE inputs (
    image_hash string NOT NULL,
    image_name string NOT NULL,
    input_data blob NOT NULL,
    PRIMARY KEY (image_hash)
);

CREATE TABLE models (
    model_hash string NOT NULL,
    model_name string NOT NULL,
    PRIMARY KEY (model_hash)
);

CREATE TABLE outputs (
    model_hash string NOT NULL,
    image_hash string NOT NULL,
    output_data blob NOT NULL,
    PRIMARY KEY (model_hash, image_hash),
    FOREIGN KEY (image_hash) REFERENCES inputs(image_hash),
    FOREIGN KEY (model_hash) REFERENCES models(model_hash)
);

CREATE TABLE preferences (
    image_hash string NOT NULL,
    model_hash string NOT NULL,
    PRIMARY KEY (image_hash),
    FOREIGN KEY (image_hash, model_hash) REFERENCES outputs(image_hash, model_hash)
);

CREATE TABLE IF NOT EXISTS palettes (
    palette_hash string NOT NULL,
    palette_data blob NOT NULL,
    PRIMARY KEY (palette_hash)
);
CREATE TABLE IF NOT EXISTS mipmaps (
    image_hash string NOT NULL,
    mipmap_index integer NOT NULL,
    mipmap_data blob NOT NULL,
    PRIMARY KEY (image_hash, mipmap_index),
    FOREIGN KEY (image_hash) REFERENCES inputs(image_hash)
);
CREATE TABLE IF NOT EXISTS palette_uses(
    palette_hash string NOT NULL,
    image_hash string NOT NULL,
    PRIMARY KEY (palette_hash, image_hash),
    FOREIGN KEY (image_hash) REFERENCES inputs(image_hash),
    FOREIGN KEY (palette_hash) REFERENCES palettes(palette_hash)
);
