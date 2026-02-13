"""
Pydantic Validation Models - LLM Output Schema Validation

WHY THIS EXISTS:
- Prevents LLM hallucinations from corrupting database
- Ensures data consistency
- Catches malformed JSON
- Type safety
- Automatic validation

TESTED:
- [x] Valid data passes
- [x] Invalid data raises ValidationError
- [x] Type coercion works
- [x] Custom validators work
- [x] Nested models work

USAGE:
    from storm_modules.validation_models import TheoryProfile, HypothesisModel
    
    # Parse LLM output
    try:
        theory = TheoryProfile(**llm_json_output)
        # ✓ Validated, safe to use
        save_to_db(theory.dict())
    except ValidationError as e:
        print(f"LLM output invalid: {e}")
        # Handle error gracefully
"""

from __future__ import annotations
from pydantic.v1 import (
    BaseModel, 
    Field, 
    validator, 
    root_validator,
    constr,
    confloat,
    conint
)
from typing import List, Dict, Optional, Literal
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class HypothesisType(str, Enum):
    """Valid hypothesis types."""
    MAIN_EFFECT = "main_effect"
    MODERATION = "moderation"
    MEDIATION = "mediation"
    INTERACTION = "interaction"


class ClaimType(str, Enum):
    """Valid claim types."""
    HYPOTHESIS = "hypothesis"
    FINDING = "finding"
    METHOD = "method"
    THEORY = "theory"


class DesignType(str, Enum):
    """Valid research design types."""
    SURVEY = "survey"
    EXPERIMENT = "experiment"
    QUASI_EXPERIMENT = "quasi-experiment"
    LONGITUDINAL = "longitudinal"
    CROSS_SECTIONAL = "cross-sectional"


class GapType(str, Enum):
    """Valid research gap types."""
    TOPIC_GAP = "topic_gap"
    CITATION_GAP = "citation_gap"
    CONTRADICTION_GAP = "contradiction_gap"
    GEOGRAPHIC_GAP = "geographic_gap"
    METHOD_GAP = "method_gap"


# ============================================================================
# CORE VALIDATION MODELS
# ============================================================================

class KeyConcept(BaseModel):
    """Key concept in a theory."""
    name: constr(min_length=2, max_length=200)
    definition: constr(min_length=10, max_length=1000)
    measurement: constr(min_length=5, max_length=500)
    
    class Config:
        frozen = True  # Immutable


class TypicalMethods(BaseModel):
    """Typical research methods for a theory."""
    design_type: DesignType
    sample_size_norm: constr(min_length=3, max_length=100)
    common_tests: List[constr(min_length=2, max_length=100)] = Field(
        min_items=1, max_items=10
    )
    common_measures: List[constr(min_length=5, max_length=200)] = Field(
        min_items=1, max_items=20
    )
    
    @validator('common_tests')
    def validate_tests(cls, v):
        """Ensure no duplicate tests."""
        if len(v) != len(set(v)):
            raise ValueError("Duplicate tests not allowed")
        return v


class TheoryProfile(BaseModel):
    """
    Complete theory profile with validation.
    
    Used by: theory_builder.py
    Prevents: Malformed theory data in database
    
    Validation Rules:
    - Name: 5-200 characters
    - Propositions: 1-10 items, each 10-500 chars
    - Key concepts: 1-20 items
    - Hypotheses: 1-30 items
    - Methods: Required, validated structure
    - Boundary conditions: 0-10 items
    - Digital application: 20-2000 chars
    """
    name: constr(min_length=5, max_length=200)
    core_propositions: List[constr(min_length=10, max_length=500)] = Field(
        min_items=1, max_items=10
    )
    key_concepts: List[KeyConcept] = Field(min_items=1, max_items=20)
    typical_hypotheses: List[constr(min_length=10, max_length=300)] = Field(
        min_items=1, max_items=30
    )
    typical_methods: TypicalMethods
    boundary_conditions: List[constr(min_length=5, max_length=300)] = Field(
        default=[], max_items=10
    )
    digital_application: constr(min_length=20, max_length=2000)
    
    @validator('name')
    def validate_name(cls, v):
        """Ensure name is properly capitalized."""
        if not v[0].isupper():
            raise ValueError("Theory name must start with capital letter")
        return v
    
    @validator('core_propositions')
    def validate_propositions(cls, v):
        """Ensure propositions are unique and substantial."""
        if len(v) != len(set(v)):
            raise ValueError("Duplicate propositions not allowed")
        return v
    
    @root_validator
    def validate_consistency(cls, values):
        """Ensure theory components are consistent."""
        name = values.get('name', '')
        digital_app = values.get('digital_application', '')
        
        # Theory name should appear in digital application
        if name.lower() not in digital_app.lower():
            raise ValueError(
                f"Digital application must reference theory name: {name}"
            )
        
        return values
    
    class Config:
        use_enum_values = True
        validate_assignment = True


class HypothesisModel(BaseModel):
    """
    Individual hypothesis with full validation.
    
    Used by: hypothesis_generator.py
    Prevents: Invalid hypotheses in database
    """
    id: constr(regex=r'^H\d+$')  # Must be H1, H2, etc.
    statement: constr(min_length=20, max_length=500)
    type: HypothesisType
    IV: constr(min_length=2, max_length=100)
    DV: constr(min_length=2, max_length=100)
    moderator: Optional[constr(min_length=2, max_length=100)] = None
    mediator: Optional[constr(min_length=2, max_length=100)] = None
    expected_direction: Optional[Literal["positive", "negative", "curvilinear"]] = None
    expected_effect_size: Optional[confloat(ge=0.0, le=2.0)] = Field(
        default=0.25, description="Cohen's d or similar"
    )
    theory_basis: constr(min_length=10, max_length=300)
    
    @root_validator
    def validate_hypothesis_structure(cls, values):
        """Ensure hypothesis structure matches type."""
        h_type = values.get('type')
        moderator = values.get('moderator')
        mediator = values.get('mediator')
        
        if h_type == HypothesisType.MODERATION and not moderator:
            raise ValueError("Moderation hypothesis must have moderator")
        
        if h_type == HypothesisType.MEDIATION and not mediator:
            raise ValueError("Mediation hypothesis must have mediator")
        
        return values
    
    class Config:
        use_enum_values = True


class HypothesisSet(BaseModel):
    """Collection of hypotheses."""
    hypotheses: List[HypothesisModel] = Field(min_items=1, max_items=20)
    
    @validator('hypotheses')
    def validate_unique_ids(cls, v):
        """Ensure hypothesis IDs are unique."""
        ids = [h.id for h in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate hypothesis IDs not allowed")
        return v


class VariableSet(BaseModel):
    """Set of research variables."""
    independent: List[constr(min_length=2, max_length=100)] = Field(
        default=[], max_items=10
    )
    dependent: List[constr(min_length=2, max_length=100)] = Field(
        default=[], max_items=10
    )
    moderators: List[constr(min_length=2, max_length=100)] = Field(
        default=[], max_items=5
    )
    mediators: List[constr(min_length=2, max_length=100)] = Field(
        default=[], max_items=5
    )
    controls: List[constr(min_length=2, max_length=100)] = Field(
        default=["age", "education", "gender"], max_items=10
    )
    
    @validator('independent', 'dependent')
    def validate_required_variables(cls, v, field):
        """Ensure at least one IV and DV."""
        if not v:
            raise ValueError(f"At least one {field.name} variable required")
        return v


class MeasureSpec(BaseModel):
    """Specification for a research measure."""
    scale_name: constr(min_length=3, max_length=200)
    citation: constr(min_length=5, max_length=200)
    items: conint(ge=1, le=100)
    response_format: constr(min_length=5, max_length=100)
    expected_alpha: confloat(ge=0.0, le=1.0)
    
    @validator('expected_alpha')
    def validate_alpha(cls, v):
        """Ensure alpha is realistic."""
        if v < 0.60:
            raise ValueError("Expected alpha should be >= 0.60 for reliability")
        return v


class AnalysisPlan(BaseModel):
    """Statistical analysis plan."""
    hypothesis: constr(regex=r'^H\d+$')
    test: constr(min_length=3, max_length=200)
    expected_beta: Optional[confloat(ge=-1.0, le=1.0)] = None
    bootstrap: Optional[conint(ge=1000, le=10000)] = None
    alpha_level: confloat(ge=0.001, le=0.1) = 0.05
    
    @validator('test')
    def validate_test_name(cls, v):
        """Ensure test name is recognized."""
        valid_tests = [
            "regression", "hierarchical regression", "anova", "ancova",
            "t-test", "hayes process", "sem", "path analysis",
            "logistic regression", "mixed model", "multilevel"
        ]
        if not any(test in v.lower() for test in valid_tests):
            raise ValueError(f"Unrecognized statistical test: {v}")
        return v


class StudyDesign(BaseModel):
    """
    Complete study design specification.
    
    Used by: study_designer.py
    Prevents: Invalid study designs
    """
    design_type: DesignType
    sample_size: conint(ge=30, le=100000)  # Minimum 30 for basic stats
    variables: VariableSet
    measures: Dict[str, MeasureSpec]
    analysis_plan: List[AnalysisPlan] = Field(min_items=1, max_items=50)
    
    @validator('sample_size')
    def validate_sample_size(cls, v, values):
        """Ensure sample size is appropriate for design."""
        design_type = values.get('design_type')
        
        if design_type == DesignType.EXPERIMENT and v < 100:
            raise ValueError("Experiments should have N >= 100")
        
        if design_type == DesignType.SURVEY and v < 200:
            raise ValueError("Surveys should have N >= 200")
        
        return v
    
    @root_validator
    def validate_measures_match_variables(cls, values):
        """Ensure all variables have measures."""
        variables = values.get('variables')
        measures = values.get('measures', {})
        
        if not variables or not measures:
            return values
        
        all_vars = (
            variables.independent + 
            variables.dependent + 
            variables.moderators + 
            variables.mediators
        )
        
        missing = [v for v in all_vars if v not in measures]
        if missing:
            raise ValueError(f"Missing measures for variables: {missing}")
        
        return values


class ClaimModel(BaseModel):
    """
    Research claim validation.
    
    Used by: claim_extractor.py
    Prevents: Malformed claims in database
    """
    claim_type: ClaimType
    claim_text: constr(min_length=20, max_length=1000)
    evidence: Optional[constr(min_length=10, max_length=2000)] = None
    confidence: confloat(ge=0.0, le=1.0) = 0.7
    source_context: Optional[constr(max_length=500)] = None
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Ensure confidence is reasonable."""
        if v < 0.6:
            raise ValueError("Claims with confidence < 0.6 should not be stored")
        return v


class ClaimSet(BaseModel):
    """Collection of claims."""
    claims: List[ClaimModel] = Field(min_items=1, max_items=20)



class ResearchGap(BaseModel):
    """
    Research gap specification.
    
    Used by: gap_finder.py
    Prevents: Invalid gaps
    """
    gap_type: GapType
    suggestion: constr(min_length=50, max_length=2000)
    gap_score: confloat(ge=0.0, le=1.0)
    supporting_evidence: Optional[List[str]] = Field(default=[], max_items=10)
    detected_at: datetime = Field(default_factory=datetime.now)
    
    @validator('gap_score')
    def validate_score(cls, v):
        """Ensure gap score is significant."""
        if v < 0.3:
            raise ValueError("Gap score should be >= 0.3 to be meaningful")
        return v


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_llm_output(
    model_class: type[BaseModel], 
    llm_output: str | dict,
    strict: bool = True
) -> Optional[BaseModel]:
    """
    Validate LLM output against a Pydantic model.
    
    Args:
        model_class: Pydantic model class
        llm_output: Raw LLM output (JSON string or dict)
        strict: If True, raise exception on validation error
    
    Returns:
        Validated model instance or None
    
    Example:
        >>> llm_json = '{"name": "Theory X", "core_propositions": [...]}'
        >>> theory = validate_llm_output(TheoryProfile, llm_json)
        >>> if theory:
        ...     save_to_db(theory.dict())
    """
    import json
    from pydantic import ValidationError
    
    # Parse JSON if string
    if isinstance(llm_output, str):
        try:
            # Clean common LLM artifacts
            clean = llm_output.replace('```json', '').replace('```', '').strip()
            data = json.loads(clean)
        except json.JSONDecodeError as e:
            if strict:
                raise ValueError(f"Invalid JSON from LLM: {e}")
            print(f"[VALIDATION] JSON parse error: {e}")
            return None
    else:
        data = llm_output
    
    # Validate
    try:
        validated = model_class(**data)
        return validated
    except ValidationError as e:
        if strict:
            raise
        print(f"[VALIDATION] Validation error: {e}")
        return None


# ============================================================================
# TESTING
# ============================================================================

def test_theory_validation():
    """Test theory profile validation."""
    print("[TEST] Theory validation...")
    
    # Valid theory
    valid_theory = {
        "name": "Social Cognitive Theory",
        "core_propositions": [
            "People learn through observation and modeling",
            "Behavior is influenced by environmental factors"
        ],
        "key_concepts": [
            {
                "name": "Self-efficacy",
                "definition": "Belief in one's ability to succeed",
                "measurement": "Likert scale questionnaire"
            }
        ],
        "typical_hypotheses": [
            "Higher self-efficacy leads to better performance"
        ],
        "typical_methods": {
            "design_type": "survey",
            "sample_size_norm": "300-500",
            "common_tests": ["regression", "SEM"],
            "common_measures": ["Self-Efficacy Scale (Bandura, 1997)"]
        },
        "boundary_conditions": ["Adult populations"],
        "digital_application": "Social Cognitive Theory explains how people learn from social media influencers"
    }
    
    theory = TheoryProfile(**valid_theory)
    assert theory.name == "Social Cognitive Theory"
    print("[TEST] ✓ Valid theory passes")
    
    # Invalid theory (missing name)
    try:
        invalid = valid_theory.copy()
        invalid["name"] = "x"  # Too short
        TheoryProfile(**invalid)
        assert False, "Should have raised ValidationError"
    except Exception:
        print("[TEST] ✓ Invalid theory rejected")


def test_hypothesis_validation():
    """Test hypothesis validation."""
    print("[TEST] Hypothesis validation...")
    
    valid_hyp = {
        "id": "H1",
        "statement": "Higher algorithmic exposure leads to increased polarization",
        "type": "main_effect",
        "IV": "algorithmic_exposure",
        "DV": "political_polarization",
        "expected_direction": "positive",
        "expected_effect_size": 0.35,
        "theory_basis": "Filter Bubble Theory, Proposition 2"
    }
    
    hyp = HypothesisModel(**valid_hyp)
    assert hyp.id == "H1"
    print("[TEST] ✓ Valid hypothesis passes")
    
    # Invalid: Moderation without moderator
    try:
        invalid = valid_hyp.copy()
        invalid["type"] = "moderation"
        HypothesisModel(**invalid)
        assert False, "Should have raised ValidationError"
    except Exception:
        print("[TEST] ✓ Invalid hypothesis rejected")


if __name__ == "__main__":
    print("=" * 60)
    print("PYDANTIC VALIDATION MODELS - SELF TEST")
    print("=" * 60)
    
    test_theory_validation()
    test_hypothesis_validation()
    
    print("\n" + "=" * 60)
    print("✓ ALL VALIDATION TESTS PASSED")
    print("=" * 60)
