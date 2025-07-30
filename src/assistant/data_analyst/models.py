from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field


class SchemaField(BaseModel):
    """Represents a field in the target schema."""

    name: str = Field(description="Field name")
    description: str = Field(description="Field description")
    type: Literal["text", "categorical", "number"] = Field(
        description="Field type (text, categorical, number)"
    )
    required: bool = Field(description="Whether field is required")
    regex: Optional[str] = Field(default=None, description="Validation regex pattern")
    default_value: Optional[Union[str, int, float, bool]] = Field(
        default=None, description="Default value"
    )
    permissible_values: Optional[List[str]] = Field(
        default=None, description="List of allowed values"
    )


class TargetSchema(BaseModel):
    """Represents the complete target schema."""

    fields: List[SchemaField] = Field(description="List of target schema fields")


class MappingDigest(BaseModel):
    """Represents a recommended mapping."""

    target_field: str = Field(description="Target field name")
    target_value: Union[str, int, float, bool, None] = Field(
        description="Final transformed value"
    )
    confidence_score: float = Field(description="Confidence score (0-1)")


class PastMappingRecord(BaseModel):
    """Represents a past mapping analysis record for learning."""

    legacy_field: str = Field(description="Legacy field name")
    legacy_value: Union[str, int, float, bool, None] = Field(
        description="Legacy field value"
    )
    recommended_mappings: List[MappingDigest] = Field(
        description="Past recommended mappings"
    )
    reasoning: str = Field(
        description="Explanation of why this is the best mapping approach"
    )


class PastAnalysis(BaseModel):
    """Represents collection of past analysis records."""

    records: List[PastMappingRecord] = Field(
        default_factory=list, description="List of past mapping records"
    )


class MappingResult(BaseModel):
    """Represents a single mapping analysis result."""

    target_field: str = Field(description="Target field name")
    target_value: Union[str, int, float, bool, None] = Field(
        description="Transformed or original value"
    )
    field_analysis_notes: str = Field(
        description="Notes from conducting field name analysis"
    )
    value_analysis_notes: str = Field(
        description="Notes from conducting value compatibility analysis"
    )
    field_similarity_score: float = Field(description="Field similarity score (0-1)")
    value_compatibility_score: float = Field(
        description="Value compatibility score (0-1)"
    )
    confidence_score: float = Field(description="Overall confidence score (0-1)")
    warnings: List[str] = Field(
        default_factory=list, description="List of warnings if any"
    )


class AnalysisResult(BaseModel):
    """Complete analysis result from the analyst."""

    legacy_field: str = Field(description="The legacy field name analyzed")
    legacy_value: Union[str, int, float, bool, None] = Field(
        description="The legacy field value analyzed"
    )
    mapping_results: List[MappingResult] = Field(
        default_factory=list,
        description="All viable mappings with confidence >= 0.6, ordered by confidence",
    )
    recommended_mappings: List[MappingDigest] = Field(
        default_factory=list,
        description="Best mapping(s) selected from mapping_results",
    )
    overall_confidence: float = Field(
        description="Overall confidence in the mapping approach (0-1)"
    )
    mapping_strategy: Literal["one-to-one", "one-to-many"] = Field(
        description="Strategy used for mapping"
    )
    reasoning: str = Field(
        description="Explanation of why this is the best mapping approach"
    )
    alternative_mappings: List[MappingDigest] = Field(
        default_factory=list,
        description="Other viable options from mapping_results not in recommended_mappings",
    )
    no_mapping_reason: Optional[str] = Field(
        default=None, description="Explanation if no viable mapping found"
    )


class AnalysisOutput(BaseModel):
    """Top-level wrapper for analysis output."""

    analysis: AnalysisResult = Field(description="The complete analysis result")


class JsonPatch(BaseModel):
    """Represents a single RFC 6902 JSON Patch operation."""

    op: Literal["add", "remove", "replace", "move", "copy", "test"] = Field(
        description="The operation to be performed"
    )
    path: str = Field(description="JSON Pointer string indicating the target location")
    value: Optional[Union[str, int, float, bool, list, dict, None]] = Field(
        default=None,
        description="The value to be used for the operation (not used for remove operations)",
    )
    from_: Optional[str] = Field(
        default=None,
        alias="from",
        description="The source location for move and copy operations",
    )


class ImplementationOutput(BaseModel):
    """Top-level wrapper for implementation output containing JSON patches."""

    patches: List[JsonPatch] = Field(
        description="List of RFC 6902 JSON Patch operations to apply"
    )
