import subprocess
import time
import sys
import os
import psutil
from datetime import datetime

# Redirect output to file for analysis
log_file = "audit_session.log"
sys.stdout = open(log_file, "w", encoding="utf-8")

print(f"[{datetime.now()}] STARTING LIVE AUDIT SESSION...")
print(f"[{datetime.now()}] Target: storm_commander.py")
print("-" * 60)

# Start process
process = subprocess.Popen(
    [sys.executable, "storm_commander.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    cwd=os.getcwd()
)

print(f"[{datetime.now()}] Process Started. PID: {process.pid}")

# Monitor loop
start_time = time.time()
duration = 45  # Run for 45 seconds to capture startup + 1-2 cycles
cpu_samples = []
mem_samples = []

try:
    while time.time() - start_time < duration:
        if process.poll() is not None:
            print(f"[{datetime.now()}] Process exited prematurely with code {process.returncode}")
            break
            
        # Resource Monitor
        try:
            p = psutil.Process(process.pid)
            cpu = p.cpu_percent(interval=0.1)
            mem = p.memory_info().rss / 1024 / 1024  # MB
            cpu_samples.append(cpu)
            mem_samples.append(mem)
        except:
            pass
            
        # Read output non-blocking (simplified by reading chunks or lines if available)
        # For simplicity in this script, we'll let it buffer and read at end or rely on file logging if configured
        # But we need to ensure the process actually runs.
        
        time.sleep(1)

    print(f"[{datetime.now()}] Audit Duration Reached. Terminating...")
    process.terminate()
    try:
        process.wait(timeout=5)
    except:
        process.kill()
        
    print(f"[{datetime.now()}] Process Terminated.")

except Exception as e:
    print(f"ERROR: {e}")
    if process: process.kill()

# Analyze captured output
print("-" * 60)
print("CAPTURED STDOUT SNAPSHOT:")
print("-" * 60)
stdout, _ = process.communicate()
if stdout:
    print(stdout)
else:
    print("(No stdout captured or buffering issue)")

# Resource Stats
if mem_samples:
    print("-" * 60)
    print(f"RESOURCE USAGE:")
    print(f"Message: Peak RAM: {max(mem_samples):.2f} MB")
    print(f"Message: Avg CPU: {sum(cpu_samples)/len(cpu_samples):.2f}%")
print("-" * 60)
print("AUDIT COMPLETE")
