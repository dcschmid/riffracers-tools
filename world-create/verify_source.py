"""Verify external source-track geometry, decoration reservations and original files."""
import hashlib
import json
import sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from shapely.geometry import Polygon,Point
from shapely.ops import unary_union
from world_create import get_project


def road_polygons(lay):
    polys=[]
    for edge in lay['edges']:
        c=np.array(edge['centerline']);t=np.gradient(c[:,:2],axis=0);t/=np.linalg.norm(t,axis=1)[:,None];n=np.c_[-t[:,1],t[:,0]]
        half=(np.array(edge['widths'])+2*edge.get('shoulder',0)-(.5 if edge['kind']=='main' else 0))/2
        a=c[:,:2]-n*half[:,None];b=c[:,:2]+n*half[:,None]
        for i in range(len(c)-1):polys.append(Polygon([a[i],a[i+1],b[i+1],b[i]]).buffer(0))
    return polys


def verify(p,mod,geometry_only=False):
    track=p.track_dir(mod);m=p.import_track(mod);pack=p.cache_dir/m.ID/'pack';lay=json.loads((pack/'layout.json').read_text());cams=json.loads((pack/'cams.json').read_text())
    polys=road_polygons(lay);roads=unary_union(polys);td=cams['topdown'];W,H=td['res'];scale=W/td['scale']
    def pixel(pt):return (W/2+(pt[0]-td['cx'])*scale,H/2-(pt[1]-td['cy'])*scale)
    expected=Image.new('L',(W,H));d=ImageDraw.Draw(expected)
    for poly in polys:
        if poly.is_empty:continue
        parts=[poly] if poly.geom_type=='Polygon' else poly.geoms
        for part in parts:d.polygon([pixel(v) for v in part.exterior.coords],fill=255)
    expected.save(track/'renders/expected-roadmask.png')
    with Image.open(track/'renders/roadmask.png') as image:actual=np.array(image.convert('L'))>127
    wanted=np.array(expected)>127;iou=float((actual&wanted).sum()/(actual|wanted).sum())
    overlay=np.zeros((H,W,3),np.uint8);overlay[actual&wanted]=[210,220,230];overlay[wanted&~actual]=[240,60,160];overlay[actual&~wanted]=[40,220,230]
    image=Image.fromarray(overlay);d=ImageDraw.Draw(image);d.text((35,28),f'Road projection IoU {iou:.4f} | white: common | pink: plan only | cyan: render only',fill='white',font=ImageFont.truetype('/usr/share/fonts/google-noto/NotoSans-Regular.ttf',25));image.save(track/'planning/geometry-overlay.png')
    decor=json.loads((track/'planning/new-decoration-footprints.json').read_text());clearances=[]
    for prop in decor:
        gap=Point(prop['x'],prop['y']).distance(roads)-prop['radius'];clearances.append(dict(type=prop['type'],x=prop['x'],y=prop['y'],clearance_m=round(gap,3)))
    bad=[d for d in clearances if d['clearance_m']<-.01]
    decoration=dict(scope='conservative 2D circles around new solid scenery and spectator reservations, against every rendered road strip; not a physics collision mesh audit',count=len(decor),overlaps=bad,minimum_clearance_m=min(d['clearance_m'] for d in clearances),passed=not bad)
    (track/'planning/decoration-validation.json').write_text(json.dumps(decoration,indent=2))
    baseline=json.loads((track/'planning/preproduction-source-baseline.json').read_text());hashes=[]
    for group in ('game_sources','references','original_photos'):
        for item in baseline[group]:
            path=item.get('path') or track/item['file'];digest=hashlib.sha256(__import__('pathlib').Path(path).read_bytes()).hexdigest();hashes.append(dict(file=str(path),unchanged=digest==item['sha256']))
    files=[]
    if not geometry_only:
        for name,cd in cams.items():
            if name=='topdown':continue
            file=track/(cd['file']+'.png');ok=file.is_file() and not file.is_symlink()
            content={}
            if ok:
                with Image.open(file) as im:
                    ok=im.size==tuple(cd['res']) and im.format=='PNG';size=list(im.size);thumb=im.convert('L').resize((256,144));a=np.array(thumb,float)
                    content=dict(mean=round(float(a.mean()),2),std=round(float(a.std()),2),entropy=round(float(thumb.entropy()),3));ok=ok and 10<a.mean()<245 and a.std()>10 and thumb.entropy()>3
            else:size=[]
            files.append(dict(file=file.name,passed=bool(ok),size=size,content=content))
    report=dict(mask_iou=iou,mask_pass=iou>.93,decoration_pass=not bad,source_hashes=hashes,all_originals_unchanged=all(x['unchanged'] for x in hashes),images=files,all_hq_images_valid=all(x['passed'] for x in files) if files else None,human_driving_test=False,engine_simulation=False)
    (track/'planning/render-verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_hashes','images')},indent=2));print('New decor:',len(decor),'overlaps:',len(bad),'minimum gap:',decoration['minimum_clearance_m'])
    if bad:print(bad[:10])
    if not report['mask_pass'] or bad or not report['all_originals_unchanged'] or (files and not report['all_hq_images_valid']):raise SystemExit(2)
    return report


if __name__=='__main__':
    p=get_project();verify(p,sys.argv[1],geometry_only='--geometry-only' in sys.argv)
