import json,sys,os
from pathlib import Path
from datetime import datetime
p=Path(sys.argv[1]);temp=p.with_suffix('.tmp')
stage=json.loads(p.read_text())['stage'] if sys.argv[2]=='failed' and p.exists() else sys.argv[3]
temp.write_text(json.dumps(dict(status=sys.argv[2],stage=stage,exit_code=int(sys.argv[4]) if len(sys.argv)>4 else None,time=datetime.now().astimezone().isoformat(),pipeline_pid=os.getppid()),indent=2)+'\n');temp.replace(p)
