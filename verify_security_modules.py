import sys
import os
import json
import time
import tempfile
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SecurityVerifier")

def test_imports():
    logger.info("Testing imports...")
    try:
        from storm_modules.db_safety import get_db_connection, safe_db_operation
        from storm_modules.rate_limiter import get_rate_limiter, rate_limited, RateLimiter
        from storm_modules.validation_models import TheoryProfile, validate_llm_output
        logger.info("✓ Imports successful")
        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def test_db_safety():
    logger.info("Testing DB Safety...")
    from storm_modules.db_safety import get_db_connection, DatabaseError
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        db_path = tmp.name
        
    try:
        # Test 1: Basic Transaction
        with get_db_connection(db_path) as conn:
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, val TEXT)")
            conn.execute("INSERT INTO test (val) VALUES (?)", ("safe",))
            
        # Verify persistence
        with get_db_connection(db_path) as conn:
            cursor = conn.execute("SELECT val FROM test")
            result = cursor.fetchone()[0]
            if result == "safe":
                logger.info("✓ DB Transaction persisted")
            else:
                logger.error("✗ DB Persistence failed")
                return False
                
        # Test 2: Rollback on Error
        try:
            with get_db_connection(db_path) as conn:
                conn.execute("INSERT INTO test (val) VALUES (?)", ("unsafe",))
                raise RuntimeError("Force Rollback")
        except (RuntimeError, DatabaseError):
            pass
            
        with get_db_connection(db_path) as conn:
            cursor = conn.execute("SELECT count(*) FROM test WHERE val='unsafe'")
            count = cursor.fetchone()[0]
            if count == 0:
                logger.info("✓ DB Rollback successful")
            else:
                logger.error("✗ DB Rollback failed")
                return False
                
        return True
    except Exception as e:
        logger.error(f"✗ DB Test Exception: {e}")
        return False
    finally:
        if os.path.exists(db_path):
            try:
                os.close(tmp.file.fileno()) # Ensure closed
                os.unlink(db_path)
            except:
                pass

def test_rate_limiter():
    logger.info("Testing Rate Limiter...")
    from storm_modules.rate_limiter import RateLimiter, get_rate_limiter
    
    # 1. Direct Instantiation Test
    limiter = RateLimiter(calls_per_minute=2, burst_size=1)
    
    # First call should pass
    if limiter.wait(timeout=1.0):
        logger.info("✓ First token acquired (Direct)")
    else:
        logger.error("✗ First token failed (Direct)")
        return False
        
    # 2. Registry Test
    registry_limiter = get_rate_limiter("test_service_registry")
    if registry_limiter.wait(timeout=1.0):
        logger.info("✓ Registry token acquired")
    else:
         logger.error("✗ Registry token failed")
         return False

    return True

def test_validation():
    logger.info("Testing Data Validation...")
    from storm_modules.validation_models import TheoryProfile, validate_llm_output
    
    valid_json = json.dumps({
        "name": "Simulated Reality Theory",
        "core_propositions": ["Reality is a construct", "Perception is a filter"],
        "key_concepts": [
             {"name": "Simulation", "definition": "A constructed reality", "measurement": "Observation"},
             {"name": "Filter", "definition": "Perceptual limit", "measurement": "Psychophysics"}
        ],
        "typical_hypotheses": ["If simulation, then glitch"],
        "typical_methods": {
            "design_type": "experiment",
            "sample_size_norm": "100-200",
            "common_tests": ["t-test"],
            "common_measures": ["Reality Scale"]
        },
        "boundary_conditions": [],
        "digital_application": "This Simulated Reality Theory applies to digital twins and metaverse environments extensively."
    })
    
    # Test Valid
    model = validate_llm_output(TheoryProfile, valid_json)
    if model and model.name == "Simulated Reality Theory":
        logger.info("✓ Valid JSON parsed correctly")
    else:
        logger.error("✗ Valid JSON rejected")
        return False
        
    # Test Invalid (Schema violation)
    invalid_json = json.dumps({
        "name": "Bad",
        "core_propositions": ["Only one prop"] # Min is 2
    })
    
    # validate_llm_output returns None on failure in non-strict mode or raises error in strict
    # The default is strict=True which raises ValueError
    try:
        validate_llm_output(TheoryProfile, invalid_json, strict=True)
        logger.error("✗ Invalid schema accepted (should have raised error)")
        return False
    except Exception:
        logger.info("✓ Invalid schema rejected")
        
    return True

if __name__ == "__main__":
    tests = [test_imports, test_db_safety, test_rate_limiter, test_validation]
    failed = False
    
    print("="*60)
    print("STORM SECURITY MODULE VERIFICATION (OFFICIAL API)")
    print("="*60)
    
    for test in tests:
        if not test():
            failed = True
            
    print("="*60)
    if failed:
        print("❌ VERIFICATION FAILED")
        sys.exit(1)
    else:
        print("✅ VERIFICATION PASSED")
        sys.exit(0)
