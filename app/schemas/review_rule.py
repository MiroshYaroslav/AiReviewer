from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from .technology import TechnologyResponse


class ReviewRuleBase(BaseModel):
    name: str
    description: str
    example_bad: Optional[str] = None
    example_good: Optional[str] = None
    is_active: bool = True
    is_global: bool = False


class ReviewRuleCreate(ReviewRuleBase):
    technology_names: List[str] = []


class ReviewRuleResponse(ReviewRuleBase):
    id: int
    technologies: List[TechnologyResponse] = []

    model_config = ConfigDict(from_attributes=True)
