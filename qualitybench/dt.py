import json,base64,io,urllib.request,numpy as np
from PIL import Image
from notch import notch
def gen(extra):
    b=dict(prompt="a red fox in snow, photograph",width=512,height=512,steps=6,seed=3,guidance_scale=1.0,output_format="png",cache_threshold=0,**extra)
    r=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8792/v1/images/generations",json.dumps(b).encode(),{"Content-Type":"application/json","X-API-Key":"localdev"}),timeout=300))
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(r["data"][0]["b64_json"]))).convert("RGB"))
a=gen({});a2=gen({});n=gen({"notch":True})
d=lambda x,y:np.abs(x.astype(int)-y.astype(int))
print("determinism maxdiff",d(a,a2).max(),"| C notch vs py notch: max",d(notch(a),n).max(),"mean",d(notch(a),n).mean())
print("n vs a",d(n,a).mean(),"n vs notch(notch(a))",d(notch(notch(a)),n).mean(),"notch(a) vs a",d(notch(a),a).mean())
a3=gen({}); n2=gen({"notch":True}); f=gen({"notch":False})
print("a3 vs a",d(a3,a).max(),"n2 vs n",d(n2,n).max(),"notch:false vs a",d(f,a).max())
print("n vs notch(a3)",d(notch(a3),n).mean())
e=notch(a3).astype(int)-n.astype(int)
print("interior mean abs",np.abs(e[8:-8,8:-8]).mean(),"edge",np.abs(e[:4]).mean(),"signed mean",e.mean(),"per-channel",e.mean((0,1)))
Image.fromarray(np.clip(np.abs(e)*20,0,255).astype(np.uint8)).save("out/notch_diff.png")
g=a3[...,1].astype(float); t=n[...,1].astype(float)
H,W=g.shape; cols=[]
for dy in range(-3,4):
    for dx in range(-3,4):
        cols.append(np.roll(np.roll(g,dy,0),dx,1)[8:-8,8:-8].ravel())
A=np.stack(cols,1); k=np.linalg.lstsq(A,t[8:-8,8:-8].ravel(),rcond=None)[0].reshape(7,7)
np.set_printoptions(precision=3,suppress=True,linewidth=200)
print(k*4096/1); print("expected center",44*44)
import ctypes
l=ctypes.CDLL("/tmp/claude-1000/-vfast-data-code-omniserve-native/ced77a2b-8826-48b7-9932-00fa2e767070/scratchpad/n.so")
c=a3.copy(); l.notch_rgb(c.ctypes.data_as(ctypes.c_void_p),512,512,3)
print("standalone C vs server n",d(c,n).max(),"standalone C vs numpy",d(c,notch(a3)).max())
