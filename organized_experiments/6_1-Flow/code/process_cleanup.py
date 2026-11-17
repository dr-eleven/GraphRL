"""Release orphaned Ray actors belonging to this experiment only."""
import os
import psutil


def stop_owned_ray_workers(run_dir, process_group):
    if not run_dir:
        return
    owned = []
    for process in psutil.process_iter(['pid', 'name']):
        try:
            if not (process.info['name'] or '').startswith('ray::'):
                continue
            if os.getpgid(process.pid) != process_group:
                continue
            if process.environ().get('FLOW_RUN_DIR') != run_dir:
                continue
            process.terminate()
            owned.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
            continue
    _, remaining = psutil.wait_procs(owned, timeout=5)
    for process in remaining:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
