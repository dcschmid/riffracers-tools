"""Camera data for source-backed preproduction; no Blender code."""
import numpy as np


def cameras(lay):
    s=lay['source']; c=np.array([[p['x'],-p['z'],p.get('y',0)] for p in s['points']]); n=len(c)
    lo=c.min(axis=0);hi=c.max(axis=0);center=(lo+hi)/2; cams={}
    def add(key,loc,target,lens=35,**kw):
        cams[key]=dict(loc=list(map(float,loc)),target=list(map(float,target)),lens=lens,res=[3840,2160],file='iron-chorus-'+key,rel_out='iron-chorus-'+key+'.png',**kw)
    def track(key,node,back=30,up=7,side=0,lens=32,**kw):
        i=node-1;p=c[i];t=c[(i+1)%n,:2]-p[:2];t/=np.linalg.norm(t);nr=np.array([-t[1],t[0]])
        loc=p+np.r_[-t*back+nr*side,up];target=c[(i+2)%n]+[0,0,1.3]
        add(key,loc,target,lens,**kw)
    add('hero',center+[570,660,600],center+[0,0,6],43)
    add('isometric',center+[-600,680,570],center+[0,0,7],43)
    span=hi-lo
    add('birdseye',[center[0],center[1],1200],[center[0],center[1],0],50,ortho=max(span[0],span[1]*16/9)*1.16)
    add('stage',[220,50,140],[0,-246,72],34)
    add('grid-hero',[-70,80,48],[0,-50,3],38)
    add('pit-lane-high',[80,110,90],[-26,8,2],38)
    add('crowd',[20,-73,7],[31,-73,3],32)
    add('faces',[19,-73,5.2],[28,-74,3.0],58)
    track('guitar-bridge',174,back=75,up=46,side=58,lens=35)
    track('loading-spiral',143,back=40,up=65,side=58,lens=33)
    add('amp-canyon',c[98]+[0,0,10],c[105]+[0,0,2],30)
    track('amp-cut',88,back=19,up=5,side=2)
    for phase,key in enumerate(('a','b','c')): track('canyon-lift-'+key,80,back=34,up=6,side=0,phase=phase)
    track('forge',236,back=70,up=30,side=-35)
    for phase,key in enumerate(('a','b','c')): track('forge-press-'+key,228,back=38,up=6,side=0,phase=phase)
    track('backstage',280,back=58,up=27,side=-45)
    track('backstage-cut',268,back=18,up=5,side=1)
    track('leap-front',329,back=40,up=7,side=0)
    track('leap-side',329,back=6,up=55,side=48,lens=34)
    p=c[328];t=c[329]-p;t[2]=0;t/=np.linalg.norm(t);nr=np.array([-t[1],t[0],0])
    add('leap-landing',p+t*70-nr*50+[0,0,45],p+t*30+[0,0,1.5],36)
    add('leap-ghost',p+nr*75-t*10+[0,0,60],p+t*30+[0,0,2],34,ghost=True)
    track('finale',354,back=38,up=12,side=4)
    track('pyro-active',38,back=32,up=6,side=0,phase=1)
    cams['topdown']=dict(cx=float(center[0]),cy=float(center[1]),scale=float(max(span[0],span[1]*1.5)*1.16),res=[2400,1600])
    return cams
