"""Local engine implementing AgentFlow's existing callable interface."""
import json, re, requests, os, uuid
from pathlib import Path
from text_encoding import repair_surrogates
class CapacityError(RuntimeError):pass
class Engine:
    def diagnostic(self,kind,**data):
        root=os.environ.get('FLOW_RUN_DIR')
        if not root:return
        folder=Path(root)/'request_diagnostics';folder.mkdir(parents=True,exist_ok=True)
        (folder/(kind+'_'+uuid.uuid4().hex+'.json')).write_text(json.dumps(dict(kind=kind,role=self.role,url=self.url,**data),ensure_ascii=True,indent=2)+'\n')
    def __init__(self,url,model,temperature=0.,max_tokens=8192,max_prompt=18432,trace=None,role='fixed',seed=42):
        self.url=url.rstrip('/');self.model=model;self.temperature=temperature;self.max_tokens=max_tokens;self.max_prompt=max_prompt;self.trace=trace;self.role=role;self.seed=seed;self.calls=[];self.http=requests.Session();self.http.trust_env=False
    def __call__(self,content,response_format=None,**kwargs):
        text=content if isinstance(content,str) else '\n'.join(content)
        # Pass the upstream response schema as a request for JSON, not a new role prompt.
        if response_format is not None:
            text+='\nReturn exactly one JSON object matching this schema: '+json.dumps(response_format.model_json_schema(),ensure_ascii=False)
        original_text=text
        text,unicode_repairs=repair_surrogates(text)
        if unicode_repairs:self.diagnostic('unicode_repair',original_user_text=original_text,sent_user_text=text,repairs=unicode_repairs)
        messages=[dict(role='system',content='You are a helpful, creative, and smart assistant.'),dict(role='user',content=text)]
        payload=dict(model=self.model,messages=messages,temperature=self.temperature,max_tokens=self.max_tokens,top_p=.99,frequency_penalty=1.2,presence_penalty=0,seed=self.seed)
        if response_format is not None and self.role=='fixed':payload['response_format']={'type':'json_object'}
        try:response=self.http.post(self.url+'/chat/completions',json=payload,timeout=600)
        except requests.RequestException as exc:
            self.diagnostic('transport_failure',payload=payload,error=repr(exc))
            raise
        if response.status_code!=200:
            self.diagnostic('http_failure',payload=payload,status=response.status_code,response=response.text)
            if response.status_code==400 and any(x in response.text.lower() for x in ['context','maximum','max_tokens']):raise CapacityError(response.text)
            raise RuntimeError(f'{self.role} server failed: {response.status_code} {response.text[:1000]}')
        data=response.json();choice=data['choices'][0];raw=choice['message']['content'] or ''
        entry=dict(role=self.role,messages=messages,raw_output=raw,finish_reason=choice['finish_reason'],usage=data.get('usage',{}))
        if unicode_repairs:entry['input_unicode_repairs']=unicode_repairs
        self.calls.append(entry)
        if self.trace is not None:
            p=data.get('prompt_token_ids');r=data.get('response_token_ids')
            if r and isinstance(r[0],list):r=r[0]
            if not p or not r:raise RuntimeError('Trainable Planner endpoint did not return exact prompt/response token IDs.')
            entry.update(prompt_ids=p,response_ids=r);self.trace.append(entry)
            if len(p)>self.max_prompt:raise CapacityError('Planner prompt exceeds configured capacity')
        if choice['finish_reason']=='length':raise CapacityError('Module output hit max_tokens; no fabricated answer')
        if response_format is not None:
            candidate=raw.strip()
            fenced=re.fullmatch(r'```(?:json)?\s*(.*?)\s*```',candidate,re.S)
            if fenced:candidate=fenced.group(1)
            try:return response_format.model_validate_json(candidate)
            except ValueError:return raw
        return raw
