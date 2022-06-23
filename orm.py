from sqlalchemy import Column, Integer, String, LargeBinary, ForeignKey, ForeignKeyConstraint
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Input(Base):
    __tablename__ = 'inputs'
    image_hash = Column(String, primary_key=True)
    image_name = Column(String, nullable=False)
    input_data = Column(LargeBinary, nullable=False)
    source_file = Column(String, nullable=False)
    group_name = Column(String)

class Bump(Base):
    __tablename__ = "bumps"
    image_hash = Column(String, ForeignKey('inputs.image_hash'), primary_key=True)
    roughness = Column(LargeBinary)
    normal = Column(LargeBinary)
    displacement = Column(LargeBinary)

class MipMap(Base):
    __tablename__ = "mipmaps"
    image_hash = Column(String, ForeignKey('inputs.image_hash'), primary_key=True)
    mipmap_index = Column(Integer, primary_key=True)
    mipmap_data = Column(LargeBinary, nullable=False)


class Model(Base):
    __tablename__ = "models"
    model_hash = Column(String, primary_key=True)
    model_name = Column(String, nullable=False)
    model_data = Column(LargeBinary, nullable=False)


class Output(Base):
    __tablename__ = "outputs"
    model_hash = Column(String, ForeignKey("models.model_hash"), primary_key=True)
    image_hash = Column(String, ForeignKey("inputs.image_hash"), primary_key=True)
    output_data = Column(LargeBinary, nullable=False)

class PaletteUse(Base):
    __tablename__ = "palette_uses"
    palette_hash = Column(String, ForeignKey("palettes.palette_hash"), primary_key=True)
    image_hash = Column(String, ForeignKey("inputs.image_hash"), primary_key=True)


class Palette(Base):
    __tablename__ = "palettes"
    palette_hash = Column(String, primary_key=True)
    palette_data = Column(LargeBinary, nullable=False)


class Preference(Base):
    __tablename__  = "preferences"
    __table_args__ = (
        ForeignKeyConstraint(
            ['image_hash', 'model_hash'],
            ['outputs.image_hash', 'outputs.model_hash']
        ),
    )
    image_hash = Column(String, primary_key=True)
    model_hash = Column(String, nullable=False)

    output = relationship("Output", foreign_keys=[image_hash, model_hash])
