import pytest
from storm_modules.theory_validator import TheoryValidator

class TestTheoryValidator:
    def test_valid_theory(self):
        """Test a perfect theory."""
        validator = TheoryValidator()
        valid_data = {
            "name": "Social Cognitive Theory",
            "core_propositions": [
                "People learn through observation and modeling of others behavior in social contexts and media environments.",
                "Behavior is influenced by environmental factors and personal cognition acting together in a reciprocal manner.", 
                "Reinforcement determines whether observed behavior is repeated over time and across different situations."
            ],
            "key_concepts": [
                {"name": "Self-efficacy", "definition": "Belief in one's own ability to succeed.", "measurement": "Self-Report Scale"},
                {"name": "Modeling", "definition": "Learning through observation of others.", "measurement": "Behavioral Observation"},
                {"name": "Reciprocal Determinism", "definition": "Interaction between person and env.", "measurement": "Statistical Model"}
            ],
            "typical_hypotheses": [
                "Higher self-efficacy leads to better performance", 
                "Observing rewards increases modeling likelihood"
            ],
            "typical_methods": {
                "design_type": "survey",
                "sample_size_norm": "300-500",
                "common_tests": ["regression"],
                "common_measures": ["Scale"]
            },
            "boundary_conditions": ["Adults"],
            "digital_application": "Social Cognitive Theory explains how users learn from influencers on social media platforms like Instagram and TikTok. Users observe the rewards (likes, comments) that influencers receive and model their own behavior to achieve similar outcomes. This demonstrates the power of observational learning in digital environments where direct reinforcement is not always present but vicarious reinforcement is abundant and highly visible to the audience."
        }
        
        is_valid, errors, score = validator.validate_theory(valid_data)
        assert is_valid, f"Validation failed! Score: {score}, Errors: {errors}"
        assert len(errors) == 0
        assert score >= 0.7
    def test_insufficient_propositions(self):
        """Test theory with too few propositions."""
        validator = TheoryValidator()
        # Schema valid, but Validator invalid (count check)
        bad_data = {
            "name": "Weak Theory",
            "core_propositions": ["Only one proposition here that is sufficiently long for schema validation purposes."],
            "key_concepts": [{"name":"C1", "definition":"Definition is long enough.", "measurement":"Measure1"}],
            "typical_hypotheses": ["Hypothesis 1 is definitely long enough for schema validation."],
            "typical_methods": {"design_type":"survey", "sample_size_norm":"100", "common_tests":["t-test"], "common_measures":["Measurement Scale"]},
            "boundary_conditions": [],
            "digital_application": "Weak Theory application test must be long enough and reference Weak Theory."
        }
        
        is_valid, errors, score = validator.validate_theory(bad_data)
        assert not is_valid, f"Should be invalid due to prop count. Score: {score}, Errors: {errors}"
        # detailed schema validation might pass now, so we expect validator error
        assert any("Too few propositions" in e for e in errors), f"Errors: {errors}"
    
    def test_vague_language_penalty(self):
        """Test specificty penalty."""
        validator = TheoryValidator()
        vague_data = {
            "name": "Vague Theory",
            # 3 propositions to pass count check
            "core_propositions": [
                "Various things happen in certain situations etc and this is long enough.",
                "Multiple stuff occurs when things change significantly enough for schema.",
                "Etc etc etc various multiple repetition to ensure penalty triggers."
            ],
            "key_concepts": [{"name":"C1", "definition":"Definition is long enough.", "measurement":"Measure1"}, {"name":"C2", "definition":"Definition is long enough.", "measurement":"Measure2"}, {"name":"C3", "definition":"Definition is long enough.", "measurement":"Measure3"}],
            "typical_hypotheses": ["Hypothesis 1 is long enough.", "Hypothesis 2 is also long enough."],
            "typical_methods": {"design_type":"survey", "sample_size_norm":"100", "common_tests":["t-test"], "common_measures":["Measurement Scale"]},
            "boundary_conditions": [],
            "digital_application": "Vague Theory application test must be sufficiently long and mention Vague Theory. We need at least fifty words here to ensure that the application length check passes successfully without adding an error to the list. This text is filler just to reach the required word count threshold defined in the validator configuration."
        }
        
        is_valid, errors, score = validator.validate_theory(vague_data)
        
        assert any("vague language" in e for e in errors), f"Errors: {errors}"
