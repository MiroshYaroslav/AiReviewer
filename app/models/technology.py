from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from .base import Base


class Technology(Base):
    __tablename__ = "technologies"

    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)

    rules = relationship(
        "ReviewRule", secondary="rule_tech", back_populates="technologies"
    )
