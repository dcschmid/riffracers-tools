"""Readable schematic trap states and ramp choices for existing-source preproduction."""
import json
import math
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from world_create import get_project
import sys

FONT='/usr/share/fonts/google-noto/NotoSans-'
def font(size,bold=False):return ImageFont.truetype(FONT+('Bold' if bold else 'Regular')+'.ttf',size)


def traps(s,out):
    W,H=2800,1800;im=Image.new('RGB',(W,H),'#f4eee4');d=ImageDraw.Draw(im)
    d.text((70,40),'IRON CHORUS · FALLEN LESEN',font=font(54,True),fill='#232b38')
    d.text((70,122),'Repräsentative Zustände der Schmiedepressen; kein zeitlich maßstäblicher Animationsablauf.',font=font(28),fill='#354252')
    h=next(x for x in s['timed_hazards'] if x['id']=='forge_press_a1');half=s['points'][h['node']-1]['width']/2+s['shoulder']-.25;reach=h['reach'];free=2*(half-reach)
    for k,(title,sub) in enumerate([('A · VORWARNUNG','Köpfe oben · betroffene Seite gelb'),('B · EINE SEITE AKTIV','Roter Kopf unten · Mitte frei'),('C · GEGENPHASE','Gegenseite unten · Mitte frei')]):
        ox=70+k*910;d.rounded_rectangle((ox,220,ox+850,1430),radius=24,fill='#ffffff')
        d.text((ox+30,250),title,font=font(34,True),fill='#263243');d.text((ox+30,307),sub,font=font(24),fill='#526171')
        left=ox+85;right=ox+765;top=420;bottom=1270;scale=(right-left)/(2*half)
        d.rectangle((left,top,right,bottom),fill='#454b56');d.line((left,top,left,bottom),fill='#e85737',width=12);d.line((right,top,right,bottom),fill='#e85737',width=12)
        active_side=1 if k==1 else -1 if k==2 else 0
        for sd in (-1,1):
            x0=left if sd==-1 else right-reach*scale;x1=x0+reach*scale
            col='#e85536' if sd==active_side else '#eeb84e' if k==0 and sd==1 else '#74817e'
            d.rectangle((x0,680,x1,920),fill=col,outline='#202835',width=5)
            d.text((x0+15,730),'UNTEN' if sd==active_side else 'OBEN',font=font(26,True),fill='white')
        mid=(left+right)/2
        for sd in (-1,1):d.line((mid+sd*2.2*scale,top+20,mid+sd*2.2*scale,bottom-20),fill='#50d9e5',width=6)
        d.line((mid,bottom-70,mid,top+95),fill='#81edc0',width=14);d.polygon([(mid,top+65),(mid-25,top+110),(mid+25,top+110)],fill='#81edc0')
        d.text((ox+210,1320),f'Mitte durchgehend frei: {free:.1f} m',font=font(28,True),fill='#27745c')
    d.text((70,1490),'Cyan: 4,4 m markierte Leitspur. Restabstand einer 1,7-m-Kart-Hülle inklusive 0,3 m Reserve: 3,10 m je Seite.',font=font(28),fill='#263243')
    d.text((70,1550),'Neuer Plan: Pressen 1,2 s Warnung (alte Quelle 0,9 s); Canyon-Lifts / Pyro 1,216 s. Noch nicht ins Spiel integriert.',font=font(28),fill='#a54c2f')
    d.text((70,1620),'Statisches Fußabdruckmodell. Sichtbarkeit, Kollision, Verkehr und tatsächliche Fahrbarkeit sind im Spiel zu testen.',font=font(28),fill='#526171')
    im.save(out/'trap-phases.png')


def jumps(s,out):
    im=Image.new('RGB',(2800,1800),'#f4eee4');d=ImageDraw.Draw(im);d.text((70,40),'IRON CHORUS · SPEAKER-RAMPEN',font=font(54,True),fill='#263243')
    d.text((70,130),'Zwei vorhandene Rampen auf durchgehender Straße. Kein verpflichtender Sprung über eine Lücke.',font=font(30),fill='#526171')
    half=s['points'][328]['width']/2+s['shoulder']-.25;scale=78;cx=1400;ground=590
    d.rectangle((cx-half*scale,ground,cx+half*scale,ground+38),fill='#454b56')
    for r,col in zip(s['ramps'],['#466f8e','#ce4b35']):
        x=cx+r['offset']*scale;w=r['width']*scale
        d.rectangle((x-w/2,ground-r['height']*scale,x+w/2,ground),fill=col)
        d.text((x-w/2,ground-r['height']*scale-65),r['variant'].upper()+f" · {r['width']} m",font=font(30,True),fill=col)
    gap=(s['ramps'][1]['offset']-s['ramps'][1]['width']/2)-(s['ramps'][0]['offset']+s['ramps'][0]['width']/2)
    d.text((740,680),f'Frontschema · freier Streifen zwischen den Rampen: {gap:.2f} m',font=font(32,True),fill='#27745c')
    for k,r in enumerate(s['ramps']):
        ox=110+k*1360;oy=1250;sc=13
        d.line((ox,oy,ox+1150,oy),fill='#454b56',width=15)
        d.polygon([(ox,oy),(ox+r['length']*sc,oy),(ox+r['length']*sc,oy-r['height']*sc)],fill='#ce4b35' if k else '#466f8e')
        for v,col in [(18,'#b99543'),(25,'#37a783'),(38,'#8272ba')]:
            t=(r['impulse']+math.sqrt(r['impulse']**2+2*16.1865*r['height']))/16.1865;tt=np.linspace(0,t,90)
            pts=[(ox+(r['length']+v*x)*sc,oy-(r['height']+r['impulse']*x-16.1865*x*x/2)*sc) for x in tt]
            d.line(pts,fill=col,width=6)
        d.text((ox,1370),r['id'],font=font(32,True),fill='#263243')
        d.text((ox,1420),f"Höhe {r['height']} m · Impuls {r['impulse']} m/s",font=font(28),fill='#526171')
    d.text((70,1530),'Seitenschema: ocker 18 · grün 25 · violett 38 m/s; hier ebener Boden. Exakte Quellstraßen-Landungen im Audit.',font=font(29),fill='#263243')
    d.text((70,1600),'Rampenmitte = Quellknoten; Lippe bei Länge/2. Bonuszonen werden bei zu hohem Tempo übersprungen.',font=font(29),fill='#a54c2f')
    d.text((70,1670),'Modell, kein Fahrtest. Geradliniger Flug, g = 16,1865 m/s², keine Lenkung, kein Verkehr, keine Beschleunigung in der Luft.',font=font(27),fill='#526171')
    im.save(out/'jump-sections.png')


if __name__=='__main__':
    p=get_project();track=p.track_dir(sys.argv[1]);s=json.loads((track/'planning/source-track.json').read_text());traps(s,track/'blueprints');jumps(s,track/'blueprints')
    from prepare_source import diagram
    lay=json.loads((track/'planning/track-layout.json').read_text())
    for mode,file in [('clean','topdown-clean.png'),('annotated','topdown-annotated.png'),('pit','pitlane-detail.png'),('options','shortcuts-jumps.png'),('hazards','trap-positions.png')]:diagram(lay,track/'blueprints'/file,mode)
