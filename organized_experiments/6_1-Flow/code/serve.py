"""OpenAI-compatible fixed/evaluation model serving with exact token instrumentation."""
import bootstrap
if __name__=='__main__':
    from agentflow.instrumentation.vllm import instrument_vllm
    instrument_vllm()
    import runpy
    runpy.run_module('vllm.entrypoints.openai.api_server',run_name='__main__')
