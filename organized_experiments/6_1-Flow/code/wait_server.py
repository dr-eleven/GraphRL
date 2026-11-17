import os,sys,time,requests
url,pid=sys.argv[1],int(sys.argv[2]);deadline=time.monotonic()+600
http=requests.Session();http.trust_env=False
while time.monotonic()<deadline:
    try:
        os.kill(pid,0)
        if http.get(url,timeout=2).status_code==200:sys.exit(0)
    except ProcessLookupError:raise SystemExit('Model server exited before readiness; inspect server log')
    except requests.RequestException:pass
    time.sleep(2)
raise SystemExit('Timed out waiting for model server')
