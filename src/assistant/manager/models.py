from typing import Literal, Any
from pydantic import BaseModel, Field


class ActionPlan(BaseModel):
    """Represents an action plan with action type and legacy content."""

    action: Literal["analyze", "transform"] = Field(
        description="Action to perform: 'analyze' to examine a pair of legacy metadata field and value against the target schema, 'transform' to apply generated patches for conversion."
    )
    legacy_field: str = Field(
        description="Field name from the legacy metadata that requires processing."
    )
    legacy_value: Any = Field(
        description="Actual value associated with the legacy field."
    )
