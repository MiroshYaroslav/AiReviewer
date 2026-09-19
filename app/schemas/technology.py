from pydantic import BaseModel, ConfigDict


class TechnologyBase(BaseModel):
    name: str


class TechnologyCreate(TechnologyBase):
    pass


class TechnologyResponse(TechnologyBase):
    id: int

    model_config = ConfigDict(from_attributes=True)
