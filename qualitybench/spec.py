import sys, numpy as np
from PIL import Image
box=(200,250,712,762)
for p in sys.argv[1:]:
    g=np.asarray(Image.open(p).convert("L").crop(box),float)
    g=g-g.mean(); w=np.hanning(512)[:,None]*np.hanning(512)[None]
    F=np.abs(np.fft.fftshift(np.fft.fft2(g*w)))**2
    n=512; c=n//2; y,x=np.mgrid[:n,:n]; fy=(y-c)/n; fx=(x-c)/n; r=np.maximum(abs(fx),abs(fy))
    tot=F.sum(); nyq=F[r>0.45].sum()/tot; mid=F[(r>0.2)&(r<=0.3)].sum()/tot
    m=F.copy(); m[r<0.08]=0; i=np.unravel_index(m.argmax(),m.shape)
    pk=m.max()/np.median(F[r>0.15])
    print(p.split("/")[-2] if "out/" in p else "user", f"nyq={nyq:.2e} mid={mid:.2e} peak/median={pk:.0f} f=({fy[i[0],0]:.3f},{fx[0,i[1]]:.3f})")
print("top peaks (png_dense20_notch / png_dense20):")
for p in [sys.argv[2], sys.argv[1]]:
    g=np.asarray(Image.open(p).convert("L").crop(box),float); g=g-g.mean()
    F=np.abs(np.fft.fftshift(np.fft.fft2(g*w)))**2
    m=F.copy(); m[r<0.04]=0; med=np.median(F[r>0.15]); out=[]
    for _ in range(8):
        i=np.unravel_index(m.argmax(),m.shape); out.append((round(float(fy[i[0],0]),3),round(float(fx[0,i[1]]),3),int(m[i]/med))); m[max(0,i[0]-4):i[0]+5,max(0,i[1]-4):i[1]+5]=0
    print(p.split("/")[-2],out)
