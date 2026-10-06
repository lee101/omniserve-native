import json,base64,io,urllib.request,numpy as np,ctypes
from PIL import Image
from notch import notch
def gen(extra,fmt="png"):
    b=dict(prompt="a red fox in snow, photograph",width=512,height=512,steps=6,seed=3,guidance_scale=1.0,output_format=fmt,**extra)
    r=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8792/v1/images/generations",json.dumps(b).encode(),{"Content-Type":"application/json","X-API-Key":"localdev"}),timeout=300))
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(r["data"][0]["b64_json"]))).convert("RGB")),r["data"][0].get("format") if isinstance(r["data"][0],dict) else None,r.get("format")
d=lambda x,y:np.abs(x.astype(int)-y.astype(int))
for extra in ({"cache_threshold":0},{"cache_threshold":0.1},{}):
    a,_,f1=gen(extra); n,_,f2=gen({**extra,"notch":True})
    print(extra,"fmt",f1,f2,"n vs notch(a) mean",d(notch(a),n).mean(),"max",d(notch(a),n).max(),"| a vs n",d(a,n).mean())
