import argparse,base64,json,os,re,sys,time,urllib.request,urllib.error
from pathlib import Path
LOG='/var/log/omniserve/access.log'
def bad5xx():
    n=0
    try:
        with open(LOG,'rb') as f:
            f.seek(max(0,os.path.getsize(LOG)-4_000_000))
            for l in f:
                m=re.search(rb'status=(5\d\d) dur_ms=([\d.]+)',l)
                if m and not (m.group(1)==b'503' and float(m.group(2))>=19900): n+=1
    except OSError: pass
    return n
ap=argparse.ArgumentParser()
ap.add_argument('--base',required=True);ap.add_argument('--out',required=True)
ap.add_argument('--variants',required=True);ap.add_argument('--prompts',default='prompts.jsonl')
ap.add_argument('--secret',default='');ap.add_argument('--only',default='');ap.add_argument('--budget-s',type=float,default=1140)
ap.add_argument('--seed',type=int,default=1234);ap.add_argument('--steps',type=int,default=0)
a=ap.parse_args()
variants=json.load(open(a.variants))
AUTO=json.load(open('auto_select.json'))
REG={e['id']:e for e in json.load(open('/nvme0n1-disk/code/cutedsl-site/inference/lora_registry.json'))}
prompts=[json.loads(l) for l in open(a.prompts)]
if a.only: prompts=[p for p in prompts if re.search(a.only,p['id'])]
t_start=time.monotonic();base5=bad5xx()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
log=open(out/'runs.jsonl','a')
for v in variants:
    d=out/v['name'];d.mkdir(exist_ok=True)
    for p in prompts:
        if v.get('cats') and p['category'] not in v['cats']: continue
        png=d/f"{p['id']}.png"
        if png.exists(): continue
        if time.monotonic()-t_start>a.budget_s: print('budget reached');sys.exit(3)
        if bad5xx()-base5>=3: print('ABORT: new 5xx in access log');sys.exit(4)
        body={'prompt':p['prompt'],'width':1024,'height':1024,'seed':a.seed,'output_format':'png','response_format':'b64_json'}
        if a.steps: body['steps']=a.steps
        body.update(v.get('payload',{}))
        lor=v.get('loras_by_cat',{}).get(p['category'],v.get('loras'))
        if lor: body['loras']=lor
        tpl=v.get('template_by_cat',{}).get(p['category'],v.get('template'))
        if tpl: body['prompt']=tpl.replace('{prompt}',p['prompt'])
        if v.get('auto'):
            sel=AUTO[p['id']]
            body['prompt']=sel['prompt']
            if sel['lora']: body['loras']=[{'path':REG[sel['lora']]['path'],'scale':sel['scale'] or 1.0}]
        h={'Content-Type':'application/json','X-Omniserve-Tier':'background'}
        if a.secret: h['Authorization']='Bearer '+a.secret
        t0=time.monotonic()
        try:
            r=urllib.request.urlopen(urllib.request.Request(a.base+'/v1/images/generations',json.dumps(body).encode(),h),timeout=600)
            doc=json.loads(r.read()); row=doc['data'][0]
            png.write_bytes(base64.b64decode(row.get('b64_json') or row['image_base64']))
            rec={'variant':v['name'],'id':p['id'],'wall_s':round(time.monotonic()-t0,3),'inference_ms':row.get('inference_time_ms'),'loras':lor,'prompt_sent':body['prompt'],'resp':{k:row[k] for k in row if k not in('b64_json','image_base64')}}
        except urllib.error.HTTPError as e:
            rec={'variant':v['name'],'id':p['id'],'error':e.code,'detail':e.read(500).decode('utf-8','replace')}
        except Exception as e:
            rec={'variant':v['name'],'id':p['id'],'error':str(e)[:300]}
        log.write(json.dumps(rec)+'\n');log.flush()
        print(v['name'],p['id'],rec.get('wall_s',rec.get('error')),flush=True)
