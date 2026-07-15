from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, String, Text, Boolean, ForeignKey, Table
from sqlalchemy.orm import relationship
from database import Base

rule_tech = Table(
    'rule_tech', Base.metadata,
    Column('rule_id', Integer, ForeignKey('review_rules.id'), primary_key=True),
    Column('tech_id', Integer, ForeignKey('technologies.id'), primary_key=True)
)

class Technology(Base):
    __tablename__ = 'technologies'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    rules = relationship("ReviewRule", secondary=rule_tech, back_populates="technologies")

class ReviewRule(Base):
    __tablename__ = 'review_rules'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(Text, nullable=False)
    example_bad = Column(Text)
    example_good = Column(Text)
    is_active = Column(Boolean, default=True)
    is_global = Column(Boolean, default=False)
    embedding = Column(Vector(3072))
    technologies = relationship("Technology", secondary=rule_tech, back_populates="rules")