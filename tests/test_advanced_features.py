"""
Test Suite for Advanced Features (Phase 3)
- Theory Versioning
- Error Handling
"""

import pytest
import tempfile
import os
import json
import logging
from storm_modules.theory_manager import TheoryVersionManager
from storm_modules.error_handler import GlobalExceptionHandler

class TestAdvancedFeatures:
    
    def test_theory_versioning_flow(self):
        """Test create, update, and rollback of theories."""
        # Use simple temp DB setup (mock config or just use direct path if possible)
        # Since TheoryVersionManager uses config.get_academic_brain_db_path, we need to mock it or use a real temp file.
        # But we can override self.db_path in the instance.
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            db_path = f.name
        
        try:
            # Init Manager
            manager = TheoryVersionManager()
            manager.db_path = db_path # Override for test
            
            # Setup Schema
            from storm_modules.schema import apply_schema
            apply_schema(db_path)
            
            # 1. Create Theory (v1)
            data_v1 = {"core_propositions": ["A"], "digital_application": "App v1"}
            manager.create_or_update_theory("TestTheory", data_v1, "Initial")
            
            hist = manager.get_history("TestTheory")
            assert len(hist) == 1
            assert hist[0]["version"] == 1
            
            # 2. Update Theory (v2)
            data_v2 = {"core_propositions": ["A", "B"], "digital_application": "App v2"}
            manager.create_or_update_theory("TestTheory", data_v2, "Added B")
            
            hist = manager.get_history("TestTheory")
            assert len(hist) == 2
            assert hist[0]["version"] == 2
            
            # Check DB HEAD
            from storm_modules.db_safety import get_db_connection
            with get_db_connection(db_path) as conn:
                row = conn.execute("SELECT digital_application FROM theories").fetchone()
                assert row[0] == "App v2"
            
            # 3. Rollback to v1
            manager.rollback("TestTheory", 1)
            
            # Rollback creates a NEW version (v3) that is a copy of v1
            hist = manager.get_history("TestTheory")
            assert len(hist) == 3
            assert hist[0]["version"] == 3
            assert "Rollback to v1" in hist[0]["description"]
            
            # Check DB HEAD matches v1 content
            with get_db_connection(db_path) as conn:
                row = conn.execute("SELECT digital_application FROM theories").fetchone()
                assert row[0] == "App v1"
                
        finally:
            try:
                os.unlink(db_path)
            except:
                pass

    def test_error_handling(self):
        """Test global error handler."""
        log_file = "test_error.log"
        handler = GlobalExceptionHandler(log_file=log_file)
        
        # Test wrapper
        @handler.safe_execute(context="TestFunc")
        def fail_func():
            raise ValueError("Intentional Fail")
            
        result = fail_func()
        assert result is None
        
        # Verify log
        assert os.path.exists(log_file)
        with open(log_file, "r") as f:
            content = f.read()
            assert "Intentional Fail" in content
            assert "[TestFunc]" in content
            
        # Cleanup
        logging.shutdown()
        try:
            os.unlink(log_file)
        except:
            pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
