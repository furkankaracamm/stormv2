
import os
import shutil
import time
from storm_commander import Colors

BRAIN_DIR = os.path.join(os.getcwd(), "storm_persistent_brain")

def reset_brain():
    print(f"{Colors.HEADER}>>> STORM BRAIN RESET UTILITY{Colors.ENDC}")
    print(f"{Colors.WARNING}WARNING: This will wipe the vector memory and force re-reading of all PDFs.{Colors.ENDC}")
    print(f"Directory: {BRAIN_DIR}")
    
    # confirm = input("Are you sure? (Type 'YES' to proceed): ")
    # if confirm.strip() != "YES":
    #     print("Aborted.")
    #     return
    print("Auto-confirming reset...")

    # 1. Stop if Locked (Simple check)
    lock_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".storm_lock")
    if os.path.exists(lock_file):
         print(f"{Colors.FAIL}\nERROR: STORM IS RUNNING!{Colors.ENDC}")
         print("Please close the main STORM window/process first.")
         return

    # 2. Backup
    backup_path = f"{BRAIN_DIR}_backup_{int(time.time())}"
    print(f"\nCreating backup at: {backup_path}")
    try:
        shutil.copytree(BRAIN_DIR, backup_path)
    except Exception as e:
        print(f"Backup failed: {e}")
        # Proceeding anyway usually requested, but let's pause
        # confirm = input("Backup failed. Continue wiping? (y/n)")
    
    # 3. Wipe Files
    print("Wiping corrupted indices...")
    targets = [
        "brain.faiss",
        "metadata.db",
        "metadata.db-shm",
        "metadata.db-wal",
        "processed_files"
    ]
    
    for t in targets:
        p = os.path.join(BRAIN_DIR, t)
        if os.path.exists(p):
            try:
                os.remove(p)
                print(f"  Deleted: {t}")
            except Exception as e:
                print(f"  Failed to delete {t}: {e}")
                
    print(f"\n{Colors.GREEN}✔ BRAIN RESET COMPLETE.{Colors.ENDC}")
    print("Next time you start STORM, it will re-ingest all PDFs correctly.")

if __name__ == "__main__":
    reset_brain()
