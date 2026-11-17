import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from process_cleanup import stop_owned_ray_workers


class CleanupTests(unittest.TestCase):
    def test_only_same_run_and_process_group_ray_actors_are_stopped(self):
        processes = []
        for pid, name, run in [(1, 'ray::WorkerDict', '/run/ours'),
                               (2, 'ray::WorkerDict', '/run/other'),
                               (3, 'python', '/run/ours'),
                               (4, 'ray::WorkerDict', '/run/ours')]:
            process = Mock(pid=pid, info={'pid':pid, 'name':name})
            process.environ.return_value = {'FLOW_RUN_DIR':run}
            processes.append(process)
        with patch('process_cleanup.psutil.process_iter', return_value=processes), \
             patch('process_cleanup.os.getpgid', side_effect=lambda pid:99 if pid==4 else 42), \
             patch('process_cleanup.psutil.wait_procs', return_value=([], [processes[0]])) as wait:
            stop_owned_ray_workers('/run/ours',42)
        processes[0].terminate.assert_called_once()
        processes[0].kill.assert_called_once()
        for process in processes[1:]:
            process.terminate.assert_not_called()
            process.kill.assert_not_called()
        wait.assert_called_once_with([processes[0]],timeout=5)
