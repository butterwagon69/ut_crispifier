CREATE TABLE textures (
    image_hash string,
    image_name string NOT NULL,
    width integer,
    height integer,
    texture_file blob,
    PRIMARY KEY (image_hash)
)
