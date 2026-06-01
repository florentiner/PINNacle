#!/usr/bin/env python3
"""Quick test to verify Optuna fixes work without actual training"""

import os
import sys
os.environ["DDEBACKEND"] = "pytorch"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import optuna
import numpy as np

# Test 1: Check if conditional parameter structure works
def test_conditional_params():
    print("=== Test 1: Conditional Parameter Structure ===")
    
    def objective(trial):
        # This mimics the actual conditional parameter structure
        chain = []
        for i in range(5):  # CHAIN_STEPS = 5
            opt_type = trial.suggest_categorical(f"step_{i}_type", ["Adam", "LBFGS", "PSO"])
            
            # Different optimizers have different lr ranges
            if opt_type == "Adam":
                lr = trial.suggest_categorical(f"step_{i}_Adam_lr", [0.01, 0.001, 0.0001])
                epochs = trial.suggest_categorical(f"step_{i}_Adam_epochs", [100, 1000, 2500])
            elif opt_type == "LBFGS":
                lr = trial.suggest_categorical(f"step_{i}_LBFGS_lr", [1.0, 0.5, 0.1])
                epochs = trial.suggest_categorical(f"step_{i}_LBFGS_epochs", [100, 500, 1000])
            else:  # PSO
                lr = trial.suggest_categorical(f"step_{i}_PSO_lr", [0.0, 0.001, 0.0001])
                epochs = trial.suggest_categorical(f"step_{i}_PSO_epochs", [100, 200, 300])
            
            chain.append({"optimizer": opt_type, "lr": lr, "epochs": epochs})
        
        # Simulate a loss value
        loss = sum(lr for _, lr, _ in [(c["optimizer"], c["lr"], c["epochs"]) for c in chain])
        return float(loss)  # Ensure Python float, not numpy
    
    # Create study with proper sampler settings
    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        n_startup_trials=3,
        warn_independent_sampling=False,
        seed=42
    )
    
    study = optuna.create_study(
        study_name="test_conditional",
        storage="sqlite:///test_conditional.db",
        sampler=sampler,
        load_if_exists=False
    )
    
    # Run a few trials
    errors = []
    for trial_num in range(5):
        try:
            study.optimize(objective, n_trials=1)
            print(f"  Trial {trial_num+1}: Success")
        except Exception as e:
            errors.append(str(e))
            print(f"  Trial {trial_num+1}: FAILED - {e}")
    
    if errors:
        print(f"✗ Test 1 FAILED with {len(errors)} errors")
        for err in errors[:3]:
            print(f"  Error: {err}")
        return False
    else:
        print("✓ Test 1 PASSED: Conditional parameters work")
        return True

# Test 2: Check float32 to float conversion
def test_float_conversion():
    print("\n=== Test 2: Float Conversion for JSON Serialization ===")
    
    def objective(trial):
        # Return numpy float32 values (like the actual code)
        import numpy as np
        rmse = np.float32(0.1234)
        brmse = np.float32(0.5678)
        
        # Test setting attributes with numpy floats
        trial.set_user_attr("test_rmse", float(rmse))  # Convert to Python float
        trial.set_user_attr("test_brmse", float(brmse))
        
        # Return numpy float
        return float(rmse + brmse)  # Must convert to Python float
    
    study = optuna.create_study(
        study_name="test_float",
        storage="sqlite:///test_float.db",
        load_if_exists=False
    )
    
    try:
        study.optimize(objective, n_trials=1)
        print("✓ Test 2 PASSED: Float conversion works")
        return True
    except Exception as e:
        print(f"✗ Test 2 FAILED: {e}")
        return False

# Test 3: Check parameter consistency in database
def test_parameter_consistency():
    print("\n=== Test 3: Parameter Consistency ===")
    
    import sqlite3
    import json
    
    db_path = "test_conditional.db"
    if not os.path.exists(db_path):
        print("  No database to check")
        return True
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check that we have the expected parameter names
    cursor.execute('''
        SELECT DISTINCT param_name 
        FROM trial_params 
        ORDER BY param_name
    ''')
    params = cursor.fetchall()
    
    print(f"  Found {len(params)} unique parameter names")
    
    # Check for any parameter with inconsistent distributions
    cursor.execute('''
        SELECT param_name, distribution_json
        FROM trial_params
        ORDER BY param_name, trial_id
    ''')
    all_params = cursor.fetchall()
    
    param_distributions = {}
    inconsistencies = []
    for param_name, dist_json in all_params:
        dist = json.loads(dist_json)
        if param_name not in param_distributions:
            param_distributions[param_name] = dist
        elif json.dumps(param_distributions[param_name]) != json.dumps(dist):
            inconsistencies.append(param_name)
    
    if inconsistencies:
        print(f"✗ Test 3 FAILED: {len(inconsistencies)} parameters have inconsistent distributions")
        for param in set(inconsistencies[:3]):
            print(f"  Inconsistent: {param}")
        return False
    else:
        print("✓ Test 3 PASSED: All parameters have consistent distributions")
        return True

def main():
    print("Running comprehensive Optuna fix tests...")
    print("=" * 60)
    
    results = []
    results.append(test_conditional_params())
    results.append(test_float_conversion())
    results.append(test_parameter_consistency())
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  Tests passed: {sum(results)}/{len(results)}")
    
    # Clean up test databases
    for db in ["test_conditional.db", "test_float.db"]:
        if os.path.exists(db):
            os.remove(db)
    
    if all(results):
        print("✓ ALL TESTS PASSED - Ready for full run!")
        return 0
    else:
        print("✗ SOME TESTS FAILED - Need to fix issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())