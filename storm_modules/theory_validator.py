"""
Theory Validator Module
Ensures the quality and completeness of generated theories before they are persisted.
Implements multi-stage validation: Schema -> Content -> Specificity -> Consistency.
"""

from typing import Dict, List, Tuple
from pydantic import ValidationError

class TheoryValidator:
    def __init__(self):
        # Quality thresholds defined in Implementation Plan
        self.quality_thresholds = {
            "min_propositions": 3,
            "min_concepts": 3,
            "min_hypotheses": 2,
            "min_application_length": 50,
            "min_prop_length": 10,  # words
            "specificity_penalty_words": ["various", "multiple", "certain", "things", "stuff", "etc"]
        }
    
    def validate_theory(
        self, 
        theory_data: Dict
    ) -> Tuple[bool, List[str], float]:
        """
        Validate theory quality.
        
        Returns:
            (is_valid, errors, quality_score)
        """
        errors = []
        quality_score = 0.0
        
        # 1. Schema validation (Base requirement)
        try:
            from storm_modules.validation_models import TheoryProfile
            theory = TheoryProfile(**theory_data)
            quality_score += 0.3 # Base score for valid schema
        except ValidationError as e:
            errors.append(f"Schema validation failed: {str(e)[:100]}...")
            return (False, errors, 0.0)
        except Exception as e:
            errors.append(f"Unexpected validation error: {e}")
            return (False, errors, 0.0)
        
        # 2. Content Completeness Checks
        
        # Propositions
        propositions = theory.core_propositions
        if len(propositions) >= self.quality_thresholds["min_propositions"]:
            quality_score += 0.15
        else:
            errors.append(f"Too few propositions: {len(propositions)} (Min: {self.quality_thresholds['min_propositions']})")
        
        # Proposition Depth (Avg words)
        avg_prop_words = sum(len(p.split()) for p in propositions) / max(len(propositions), 1)
        if avg_prop_words >= self.quality_thresholds["min_prop_length"]:
            quality_score += 0.1
        else:
            # Not a hard error, just no points
            pass

        # Concepts
        if len(theory.key_concepts) >= self.quality_thresholds["min_concepts"]:
            quality_score += 0.1
        else:
            errors.append(f"Too few key concepts: {len(theory.key_concepts)}")

        # Hypotheses
        if len(theory.typical_hypotheses) >= self.quality_thresholds["min_hypotheses"]:
            quality_score += 0.1
        
        # 3. Specificity Check
        # Penalize vague words in propositions
        vague_count = 0
        all_text = " ".join(propositions).lower()
        for word in self.quality_thresholds["specificity_penalty_words"]:
            if word in all_text:
                vague_count += 1
        
        if vague_count == 0:
            quality_score += 0.1
        elif vague_count > 3:
            quality_score -= 0.1 # Penalty
            errors.append(f"Theory contains vague language ({vague_count} instances)")

        # 4. Consistency & Application
        # Check if theory name is referenced in the application
        name_in_app = theory.name.lower() in theory.digital_application.lower()
        if name_in_app:
            quality_score += 0.05
        else:
            errors.append("Digital application does not reference theory name")
            
        # Check application length
        app_len = len(theory.digital_application.split())
        if app_len >= self.quality_thresholds["min_application_length"]:
            quality_score += 0.1
        else:
             errors.append(f"Digital application too short: {app_len} words")

        # 5. Methods Validation
        if theory.typical_methods:
            quality_score += 0.0
            # Check for non-empty lists
            if theory.typical_methods.common_tests:
                quality_score += 0.05
            if theory.typical_methods.common_measures:
                quality_score += 0.05

        # Final Scoring & Threshold
        # Max score is ~1.0. Pass threshold 0.7
        
        # Round score
        quality_score = round(quality_score, 2)
        
        # Determine validity
        # Must have NO critical errors (schema, counts) AND score >= 0.7
        is_valid = (len(errors) == 0) and (quality_score >= 0.7)
        
        # If score is low but no specific errors, add a generic error
        if not is_valid and len(errors) == 0:
            errors.append(f"Quality score too low: {quality_score} (Needs 0.7)")
            
        return (is_valid, errors, quality_score)
