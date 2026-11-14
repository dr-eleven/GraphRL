"""Two CPU/Gloo workers: an infrastructure failure invalidates a cross-rank G group."""
import json
from pathlib import Path
import tempfile
import torch
import torch.distributed as dist
from test_rollout_runtime import RuntimeTests
from runtime import InfrastructureError


def main():
    dist.init_process_group('gloo')
    rank=dist.get_rank()
    RuntimeTests.setUpClass()
    test=RuntimeTests()
    failed=[False]
    def script(h,i):
        if rank==0 and i==0 and not failed[0]:
            failed[0]=True
            raise InfrastructureError('cross-rank injected failure')
        return '{"analysis":"Reason","prediction":"library"}'
    def gather(tensor):
        result=[torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(result,tensor)
        return torch.cat(result)
    with tempfile.TemporaryDirectory() as tmp:
        harness=test.harness(tmp,script,sources=(7,8))
        harness.accelerator.gather=gather
        harness.accelerator.process_index=rank
        result=harness._generate([[],[]])
        retries=[e._infra_group_retries for e in harness.environments]
        counts=[len(harness.histories[i]) for i in range(2)]
        assert retries==[1,0], (rank,retries)
        assert counts==[2,1], (rank,counts)
        assert [e._score(c) for e,c in zip(harness.environments,result[3])]==[1,1]
        rows=[None]*dist.get_world_size()
        dist.all_gather_object(rows,dict(rank=rank,retries=retries,generations=counts))
        if rank==0:
            report=dict(backend='gloo',cpu_workers=2,scripted_generation=True,
                        cross_rank_group_retry_verified=True,healthy_group_not_resampled=True,workers=rows)
            (Path(__file__).resolve().parents[1]/'verification/distributed_groups.json').write_text(json.dumps(report,indent=2)+'\n')
    dist.destroy_process_group()


if __name__=='__main__':main()
