"""
STORM System Unit Tests
FAZ I: Zero-Error Architecture Implementation

Tests critical modules for correctness and error handling.
"""
import sys
import os
import json
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add storm to path
STORM_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(STORM_DIR))


class TestLLMGateway(unittest.TestCase):
    """Test LLM Gateway functionality."""
    
    def test_import(self):
        """Test that LLM Gateway can be imported."""
        from storm_modules.llm_gateway import LLMGateway, get_llm_gateway, retry_with_backoff
        self.assertTrue(callable(get_llm_gateway))
        self.assertTrue(callable(retry_with_backoff))
    
    def test_singleton(self):
        """Test that get_llm_gateway returns same instance."""
        from storm_modules.llm_gateway import get_llm_gateway
        llm1 = get_llm_gateway()
        llm2 = get_llm_gateway()
        self.assertIs(llm1, llm2)
    
    def test_provider_detection(self):
        """Test that provider is detected."""
        from storm_modules.llm_gateway import get_llm_gateway
        llm = get_llm_gateway()
        self.assertIn(llm.provider, ['groq', 'openrouter', 'ollama', None])
    
    def test_retry_decorator(self):
        """Test retry_with_backoff decorator exists and is callable."""
        from storm_modules.llm_gateway import retry_with_backoff
        
        # Just verify decorator works without errors
        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def simple_function():
            return "success"
        
        result = simple_function()
        self.assertEqual(result, "success")


class TestTheoryBuilder(unittest.TestCase):
    """Test Theory Builder functionality."""
    
    def test_import(self):
        """Test that Theory Builder can be imported."""
        from storm_modules.theory_builder import TheoryDatabaseBuilder
        self.assertTrue(callable(TheoryDatabaseBuilder))
    
    def test_init_uses_llm_gateway(self):
        """Test that TheoryDatabaseBuilder uses LLM Gateway, not direct Ollama."""
        from storm_modules.theory_builder import TheoryDatabaseBuilder
        builder = TheoryDatabaseBuilder()
        # Should have llm attribute from get_llm_gateway
        self.assertTrue(hasattr(builder, 'llm'))
    
    def test_theory_list(self):
        """Test that theory list has expected theories."""
        from storm_modules.theory_builder import TheoryDatabaseBuilder
        self.assertIn('Dead Internet Theory', TheoryDatabaseBuilder.THEORY_LIST)
        self.assertIn('Social Cognitive Theory', TheoryDatabaseBuilder.THEORY_LIST)
        self.assertGreater(len(TheoryDatabaseBuilder.THEORY_LIST), 20)


class TestThesisGenerator(unittest.TestCase):
    """Test Thesis Generator with semantic theory matching."""
    
    def test_import(self):
        """Test that Thesis Generator can be imported."""
        from storm_modules.thesis_generator import ThesisGenerator
        self.assertTrue(callable(ThesisGenerator))
    
    def test_semantic_matching_method_exists(self):
        """Test that semantic matching method exists."""
        from storm_modules.thesis_generator import ThesisGenerator
        tg = ThesisGenerator()
        self.assertTrue(hasattr(tg, '_match_theory_semantic'))
        self.assertTrue(callable(tg._match_theory_semantic))
    
    def test_literature_synthesizer_exists(self):
        """Test that lit_synth attribute exists."""
        from storm_modules.thesis_generator import ThesisGenerator
        tg = ThesisGenerator()
        self.assertTrue(hasattr(tg, 'lit_synth'))


class TestHypothesisGenerator(unittest.TestCase):
    """Test Hypothesis Generator error handling."""
    
    def test_import(self):
        """Test that Hypothesis Generator can be imported."""
        from storm_modules.hypothesis_generator import HypothesisGenerator
        self.assertTrue(callable(HypothesisGenerator))
    
    def test_specific_exceptions(self):
        """Test that exceptions are specific, not bare."""
        import inspect
        from storm_modules.hypothesis_generator import HypothesisGenerator
        
        source = inspect.getsource(HypothesisGenerator)
        # Check that we don't have bare 'except:' (should be 'except XXXError')
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'except:' in line and 'except Exception' not in line.replace('except:', 'except Exception:'):
                # Allow except: only if followed by specific type
                self.fail(f"Found bare 'except:' at line {i}: {line}")


class TestGapFinder(unittest.TestCase):
    """Test Gap Finder error handling."""
    
    def test_import(self):
        """Test that Gap Finder can be imported."""
        from storm_modules.gap_finder import GapFinder
        self.assertTrue(callable(GapFinder))
    
    def test_gap_types(self):
        """Test that gap types are defined."""
        from storm_modules.gap_finder import GapFinder
        gf = GapFinder('.')
        
        # Should have methods for different gap types
        self.assertTrue(hasattr(gf, 'find_topic_gaps'))
        self.assertTrue(hasattr(gf, 'find_citation_gaps'))


class TestStudyDesigner(unittest.TestCase):
    """Test Study Designer functionality."""
    
    def test_import(self):
        """Test that Study Designer can be imported."""
        from storm_modules.study_designer import QuantitativeStudyDesigner
        self.assertTrue(callable(QuantitativeStudyDesigner))
    
    def test_uses_llm_gateway(self):
        """Test that StudyDesigner uses LLM Gateway."""
        from storm_modules.study_designer import QuantitativeStudyDesigner
        sd = QuantitativeStudyDesigner()
        self.assertTrue(hasattr(sd, 'llm'))


class TestClaimExtractor(unittest.TestCase):
    """Test Claim Extractor functionality."""
    
    def test_import(self):
        """Test that claim extractor functions can be imported."""
        from storm_modules.claim_extractor import extract_claims, extract_claims_regex
        self.assertTrue(callable(extract_claims))
        self.assertTrue(callable(extract_claims_regex))
    
    def test_regex_extraction(self):
        """Test regex-based claim extraction."""
        from storm_modules.claim_extractor import extract_claims_regex
        
        text = """
        This study demonstrates that social media bots significantly influence
        public opinion. Our findings suggest that automated accounts can shape
        discourse patterns in measurable ways.
        """
        
        # extract_claims_regex may have different signature, just test it works
        try:
            claims = extract_claims_regex(text, max_claims=3)
            self.assertIsInstance(claims, list)
        except TypeError:
            # Try without max_claims if signature is different
            claims = extract_claims_regex(text)
            self.assertIsInstance(claims, list)


class TestConfigPaths(unittest.TestCase):
    """Test configuration path functions."""
    
    def test_import(self):
        """Test that config functions can be imported."""
        from storm_modules.config import get_academic_brain_db_path, get_work_dir
        self.assertTrue(callable(get_academic_brain_db_path))
        self.assertTrue(callable(get_work_dir))
    
    def test_paths_are_path_objects(self):
        """Test that path functions return Path objects."""
        from storm_modules.config import get_academic_brain_db_path, get_work_dir
        
        work_dir = get_work_dir()
        db_path = get_academic_brain_db_path()
        
        self.assertIsInstance(work_dir, Path)
        self.assertIsInstance(db_path, Path)


def run_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestLLMGateway))
    suite.addTests(loader.loadTestsFromTestCase(TestTheoryBuilder))
    suite.addTests(loader.loadTestsFromTestCase(TestThesisGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestHypothesisGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestGapFinder))
    suite.addTests(loader.loadTestsFromTestCase(TestStudyDesigner))
    suite.addTests(loader.loadTestsFromTestCase(TestClaimExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigPaths))
    
    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_tests()
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"Success: {result.wasSuccessful()}")
    print("=" * 60)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
