import bootstrap
from agentflow.verl.entrypoint import run_ppo
import hydra
@hydra.main(config_path='../configs',config_name='graph_flow',version_base=None)
def main(config):
    import ray, os
    from process_cleanup import stop_owned_ray_workers
    try:run_ppo(config)
    finally:
        try:ray.shutdown()
        finally:stop_owned_ray_workers(os.environ.get('FLOW_RUN_DIR'),os.getpgrp())
if __name__=='__main__':main()
