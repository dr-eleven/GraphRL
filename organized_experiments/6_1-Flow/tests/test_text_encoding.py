import os,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from text_encoding import repair_surrogates
from engines import Engine

class EncodingTests(unittest.TestCase):
    def test_valid_text_unchanged_and_invalid_units_visible(self):
        text='中文 👩🏽‍💻 é quotation “hello”'
        self.assertEqual(repair_surrogates(text),(text,[]))
        repaired,events=repair_surrogates('a\ud800b\udc00c')
        self.assertEqual(repaired,r'a\ud800b\udc00c')
        self.assertEqual(len(events),2);repaired.encode('utf-8')
        self.assertEqual(repair_surrogates('\ud83d\ude00')[0],'😀')
    def test_sent_prompt_trace_and_diagnostics_match(self):
        class Response:
            status_code=200
            def json(self):return dict(choices=[dict(message=dict(content='ok'),finish_reason='stop')],prompt_token_ids=[1,2],response_token_ids=[3])
        class HTTP:
            def post(self,url,json,timeout):self.payload=json;return Response()
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,FLOW_RUN_DIR=tmp):
            trace=[];engine=Engine('http://local','fixture',role='planner',trace=trace);engine.http=HTTP()
            self.assertEqual(engine('bad \ud800'),'ok')
            self.assertEqual(engine.http.payload['messages'],trace[0]['messages'])
            self.assertEqual(trace[0]['messages'][1]['content'],r'bad \ud800')
            self.assertEqual(trace[0]['response_ids'],[3])
            events=list((Path(tmp)/'request_diagnostics').glob('unicode_repair*'))
            self.assertEqual(len(events),1)
            audit=json.loads(events[0].read_text());self.assertEqual(audit['original_user_text'],'bad \ud800')
    def test_http_error_saves_exact_failed_request(self):
        class Response:status_code=400;text='TextEncodeInput error'
        class HTTP:
            def post(self,**kwargs):return Response()
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,FLOW_RUN_DIR=tmp):
            engine=Engine('http://local','fixture')
            with patch.object(engine.http,'post',return_value=Response()):
                with self.assertRaises(RuntimeError):engine('input')
            files=list((Path(tmp)/'request_diagnostics').glob('http_failure*'))
            self.assertEqual(json.loads(files[0].read_text())['payload']['messages'][1]['content'],'input')
