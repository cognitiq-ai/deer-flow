# pylint: disable=line-too-long

from typing import List, Literal, Type

from pydantic import BaseModel, Field, create_model

from src.kg.base_models import RelationshipType
from src.kg.utils import PydanticEnum


class InferredRelationship(BaseModel):
    """Schema for a single inferred relationship between a pair of concepts."""

    relationship_type: PydanticEnum(RelationshipType) = Field(
        description="Type of relationship between the concepts"
    )
    direction: int = Field(
        description="Direction of the relationship (1 for A -> B, -1 for B -> A)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Strength of the relationship (0.0-1.0)"
    )
    sources: List[str] = Field(
        default_factory=list,
        description="Source ids (`sources.id`) from where the relationship is inferred",
    )

    @classmethod
    def with_types(cls, types: List[RelationshipType]) -> Type[BaseModel]:
        """
        Dynamically creates a model inheriting original descriptions and constraints
        while updating the relationship_type to a restricted Literal.
        """
        # 1. Define the restricted literal
        allowed_set = set(types).union([RelationshipType.NO_RELATIONSHIP])
        literals = Literal.__getitem__(tuple(rt.code for rt in allowed_set))

        # 2. Extract original FieldInfo for all fields to keep metadata
        base_fields = {
            name: (field.annotation, field) for name, field in cls.model_fields.items()
        }

        # 3. Update the specific field with the new Literal constraint
        base_fields["relationship_type"] = (
            literals,
            cls.model_fields["relationship_type"],
        )

        # 4. Create model using __base__ to inherit any validators or methods
        return create_model(f"{cls.__name__}Constrained", __base__=cls, **base_fields)
