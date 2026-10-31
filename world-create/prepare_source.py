"""Read-only import of an existing game track into an external preproduction world.

No Blender modeling or game writes here: source snapshots, explicit source-profile
checks, diagrams and a data pack consumed by the existing preview renderer.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from world_create import get_project, defaults
from w2lib import geom as G


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def xyz(points):
    return np.array([[p['x'], -p['z'], p.get('y', 0.0)] for p in points], float)


def lengths(c, closed=False):
    d = np.diff(np.vstack([c, c[:1]]) if closed else c, axis=0)
    return np.linalg.norm(d[:, :2], axis=1)


def source_audit(s, expected):
    """Source-profile checks. Values intentionally differ from the new World-2 rules."""
    c = xyz(s['points']); ds = lengths(c, True)
    n = len(c); rows = []
    def add(key, ok, detail, model=False):
        rows.append(dict(id=key, status='PASS' if ok else 'FAIL', detail=detail, model=model))
    add('finite', np.isfinite(c).all(), f'{n} finite metre-based points')
    add('node_count', n == expected['points'], f'{n} source points; expected {expected["points"]}')
    add('nonzero_segments', float(ds.min()) > .05, f'shortest segment {ds.min():.3f} m')
    add('closure', ds[-1] < 16, f'explicit cyclic source; closing segment {ds[-1]:.3f} m')
    add('length', 1500 < ds.sum() < 2600, f'main ring {ds.sum():.2f} m')
    widths = np.array([p.get('width', s['width']) for p in s['points']])
    add('main_width', widths.min() >= expected.get('minimum_main_width', 12), f'core widths {widths.min():.2f}–{widths.max():.2f} m; shoulders {s["shoulder"]} m each')
    grade = np.abs(np.diff(np.r_[c[:, 2], c[0, 2]]) / ds)
    add('grade', grade.max() <= .13, f'maximum main grade {grade.max()*100:.2f}%')
    a = c - np.roll(c, 1, axis=0); b = np.roll(c, -1, axis=0) - c
    cross = np.abs(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0])
    radius = np.linalg.norm(a[:, :2],axis=1)*np.linalg.norm(b[:, :2],axis=1)*np.linalg.norm((a+b)[:, :2],axis=1) / np.maximum(2*cross, 1e-10)
    add('radius', radius.min() + expected.get('radius_tolerance', 0) >= expected.get('minimum_radius', 20), f'minimum sampled circumradius {radius.min():.2f} m')
    gates = s['checkpoints']
    add('gates', [g['order'] for g in gates] == list(range(1,len(gates)+1)) and all(1<=g['node']<=n for g in gates), f'{len(gates)} ordered gates; finish {s["finish"]}')
    add('grid', len(s['spawns']) == expected['grid'] and all(1<=g['node']<=n for g in s['spawns']), f'{len(s["spawns"])} grid slots')
    sc = s['shortcuts']; connected = True
    for route in sc:
        r = xyz(route['points'])
        connected &= np.linalg.norm(r[0]-c[route['node']-1]) < .1
        connected &= np.linalg.norm(r[-1]-c[route['exit_node']-1]) < .1
    add('shortcut_graph', len(sc) == expected['shortcuts'] and connected, f'{len(sc)} distinct source branches; fork and merge endpoints on the directed ring')
    add('shortcut_width', all(r['width'] >= 3.6 for r in sc), '; '.join(f'{r["id"]}: {r["width"]} m' for r in sc))
    pit = s['pitlane']; bays = pit['bays']
    add('pit_boxes', len(bays) == expected['boxes'] and len({b['id'] for b in bays}) == len(bays), f'{len(bays)} service bays; source profile explicitly expects {expected["boxes"]}')
    pit_join_dist = np.linalg.norm(xyz(pit['points'])[[0,-1],:2]-c[[pit['entry_node']-1,pit['exit_node']-1],:2],axis=1)
    pit_ok = all(distance <= s['points'][node-1]['width']/2 for distance,node in zip(pit_join_dist,(pit['entry_node'],pit['exit_node'])))
    add('pit_graph', pit_ok, f'one-way source lane node {pit["entry_node"]} → {pit["exit_node"]}; {pit["length"]:.1f} m; endpoint offsets {pit_join_dist.round(2).tolist()} m within the main road, not forcibly snapped to its centre')
    add('ramp_choices', {r['variant'] for r in s['ramps']} == {'normal','risk'}, 'wide low and narrow high ramp; main road stays continuous')
    hazards = s['timed_hazards']; free=[]
    for h in hazards:
        p=s['points'][h['node']-1]
        same=[q for q in hazards if q['node']==h['node']]
        # Conservative: even opposing heads down simultaneously; include the source shoulders.
        reach_by_side={side:max([q['reach'] for q in same if q['side']==side] or [0]) for side in (-1,1)}
        free.append(p['width']+2*s['shoulder']-.5-sum(reach_by_side.values()))
    add('hazard_free_corridor', min(free)>=4.4, f'conservative source corridor >= {min(free):.2f} m, including source shoulders', True)
    add('hazard_warning', min(h['telegraph'] for h in hazards)>=.85, f'source warnings {min(h["telegraph"] for h in hazards):.3f}–{max(h["telegraph"] for h in hazards):.3f} s; not silently changed to World-2 timings', True)
    add('hazard_duty', max((h['strike']+h['extended'])/h['period'] for h in hazards)<=.55, f'maximum strike+active duty {max((h["strike"]+h["extended"])/h["period"] for h in hazards):.3f}', True)
    # Check only spatially remote samples; exclude nearby portions of the same carriageway.
    arc=np.r_[0,np.cumsum(ds[:-1])]; delta=np.abs(arc[:,None]-arc[None,:]); delta=np.minimum(delta,ds.sum()-delta)
    dist=np.linalg.norm(c[:,None,:2]-c[None,:,:2],axis=2)
    sep=np.abs(c[:,None,2]-c[None,:,2])
    overlap=(dist < (widths[:,None]+widths[None,:])/2-1)&(delta>100)&np.triu(np.ones((n,n),bool),1)
    declared=np.zeros((n,n),bool)
    for x in s.get('crossings',[]):
        u=np.arange(x['upper'][0]-1,x['upper'][1]); l=np.arange(x['lower'][0]-1,x['lower'][1])
        declared[np.ix_(u,l)]=True; declared[np.ix_(l,u)]=True
    bad=overlap&((sep<6)|~declared)
    add('overpass', not bad.any(), f'{len(s.get("crossings",[]))} declared crossing; remote overlapping centreline samples checked for >=6 m elevation separation')
    return dict(profile='existing-game-source-v1',checks=rows,total=len(rows),fails=sum(r['status']=='FAIL' for r in rows),critical=sum(r['status']=='FAIL' for r in rows),warnings=0,geometry_validated=not any(r['status']=='FAIL' for r in rows),note='Source data audit and sampled geometric checks, not a new engine test or a collision audit of new decoration.')


def normalize(s, m):
    c=xyz(s['points']); n=len(c); ds=lengths(c,True); arc=np.r_[0,np.cumsum(ds[:-1])]
    joints={0:'n_sf'}
    for r in s['shortcuts']:
        joints[r['node']-1]='n_'+r['id']+'_fork'; joints[r['exit_node']-1]='n_'+r['id']+'_merge'
    p=s['pitlane']; joints[p['entry_node']-1]='n_pit_entry'; joints[p['exit_node']-1]='n_pit_exit'
    nodes=[dict(id=name,xy=c[i,:2].tolist(),z=float(c[i,2]),zone_radius=30) for i,name in sorted(joints.items())]
    edges=[]; main=[]
    keys=sorted(joints)
    for k,i in enumerate(keys):
        end=keys[(k+1)%len(keys)]; idx=np.arange(i,end+1 if end>i else end+n+1)%n
        name=f'main_{k+1}'; main.append(name)
        edges.append(dict(id=name,kind='main',centerline=c[idx].tolist(),widths=[s['points'][j]['width'] for j in idx],shoulder=s['shoulder'],**{'from':joints[i],'to':joints[end]},length=float(lengths(c[idx]).sum()),s0=float(arc[i]),s1=float(arc[end] if end>i else ds.sum()),source_indices=idx.tolist()))
    for r in s['shortcuts']:
        a=xyz(r['points'])
        edges.append(dict(id=r['id'],kind='shortcut',centerline=a.tolist(),widths=[r['width']]*len(a),shoulder=.25,**{'from':joints[r['node']-1],'to':joints[r['exit_node']-1]},length=float(lengths(a).sum())))
    a=xyz(p['points'])
    edges.append(dict(id='pit_lane',kind='pit_lane',centerline=a.tolist(),widths=[q.get('left',p['width']/2)+q.get('right',p['width']/2) for q in p['points']],shoulder=0,**{'from':'n_pit_entry','to':'n_pit_exit'},length=float(lengths(a).sum())))
    grid=[]
    for g in s['spawns']:
        i=g['node']-1; t=c[(i+1)%n,:2]-c[i,:2]; t/=np.linalg.norm(t)
        normal=np.array([-t[1],t[0]])
        grid.append(dict(id=g['id'],xy=(c[i,:2]+normal*g['lane']).tolist(),heading_deg=float(np.degrees(np.arctan2(t[1],t[0])))))
    area=np.sum(c[:,0]*np.roll(c[:,1],-1)-np.roll(c[:,0],-1)*c[:,1])/2
    return dict(schema='riffracers.source-preproduction/1',id=m.ID,name=m.NAME,genre=m.GENRE,bpm=m.BPM,difficulty=1,direction='CCW' if area>0 else 'CW',units='m',length_m=float(ds.sum()),nodes=nodes,edges=edges,main_loop=main,start_finish=dict(xy=c[0,:2].tolist(),grid=grid),pit=dict(boxes=p['bays']),stands=[],sections=s['art_zones'],shortcuts=s['shortcuts'],jumps=s['ramps'],traps=[dict(h,name=h['id'],visual='readable warning and open driving corridor') for h in s['timed_hazards']],terrain=dict(style='festival'),source=s,source_transform='game (x,y,z) → Blender (x,-z,y); source arrays never rewritten',notes=['Imported road geometry; approved new overview and section concepts are the visual references.'])


def diagram(lay, path, mode='clean'):
    W,H=2800,1800
    im=Image.new('RGB',(W,H),'#f5f1e9'); d=ImageDraw.Draw(im)
    try: font=ImageFont.truetype('/usr/share/fonts/google-noto/NotoSans-Regular.ttf',24); big=ImageFont.truetype('/usr/share/fonts/google-noto/NotoSans-Bold.ttf',43)
    except OSError: font=ImageFont.load_default(); big=font
    s=lay['source']; c=xyz(s['points']); a=np.vstack([np.array(e['centerline']) for e in lay['edges']])
    if mode=='pit': a=xyz(s['pitlane']['points'])
    lo=a[:,:2].min(axis=0)-45; hi=a[:,:2].max(axis=0)+45
    scale=min((W-420)/(hi[0]-lo[0]),(H-230)/(hi[1]-lo[1])); center=(lo+hi)/2
    def p(v): return ((float(v[0])-center[0])*scale+W*.53,H*.55-(float(v[1])-center[1])*scale)
    for edge in sorted(lay['edges'],key=lambda e:np.mean(np.array(e['centerline'])[:,2])):
        a=np.array(edge['centerline']); w=np.array(edge['widths'])
        col='#586273' if edge['kind']=='main' else '#259f85' if edge['kind']=='shortcut' else '#ca9641'
        t=np.gradient(a[:,:2],axis=0);t/=np.linalg.norm(t,axis=1)[:,None];normal=np.c_[-t[:,1],t[:,0]];half=w/2+edge.get('shoulder',0)
        left=a[:,:2]-normal*half[:,None];right=a[:,:2]+normal*half[:,None]
        for i in range(len(a)-1):
            if i in edge.get('air_segments', []): continue
            d.polygon([p(left[i]),p(left[i+1]),p(right[i+1]),p(right[i])],fill=col)
        d.line([p(q) for q in a],fill='#e5e9ed',width=2)
    for i in range(0,len(c),24):
        u=np.array(p(c[i])); v=np.array(p(c[(i+2)%len(c)])); t=v-u;t/=max(np.linalg.norm(t),1)
        q=u+t*20; n=np.array([-t[1],t[0]])
        d.polygon([tuple(q),tuple(u+n*5),tuple(u-n*5)],fill='#ffffff')
    if mode!='clean':
        for zone in s['art_zones']:
            q=p(c[zone['nodes'][0]-1]);d.text(q,zone['name'],font=font,fill='#17202d',stroke_width=1,stroke_fill='#ffffff')
        for r in s['shortcuts']:
            d.text(p(xyz(r['points'])[len(r['points'])//2]),r['name'],font=font,fill='#087765',stroke_width=1,stroke_fill='#ffffff')
        for h in s['timed_hazards']:
            q=p(c[h['node']-1]);d.ellipse([q[0]-8,q[1]-8,q[0]+8,q[1]+8],fill='#d54325')
        for bay in s['pitlane']['bays']:
            q=p([bay['x'],-bay['z']]);d.rectangle([q[0]-6,q[1]-6,q[0]+6,q[1]+6],fill='#f4c13b')
    d.text((50,32),lay['name'].upper()+' – '+mode.upper(),font=big,fill='#202939')
    d.text((50,102),f'{lay["length_m"]:.1f} m · {lay["direction"]} · source geometry · {len(s["pitlane"]["bays"])} service bays · {len(s["shortcuts"])} shortcuts · {len(s["ramps"])} optional ramps',font=font,fill='#394256')
    d.text((50,H-55),'External preproduction · design/model data, no new driving test · game files read-only',font=font,fill='#394256')
    im.save(path)


def make_pack(lay, module, project, target):
    import source_views
    target.mkdir(parents=True,exist_ok=True)
    raw=lay['source']; points=xyz(raw['points'])
    x0,y0=points[:,:2].min(axis=0)-220; x1,y1=points[:,:2].max(axis=0)+220
    xs=np.linspace(x0,x1,100); ys=np.linspace(y0,y1,120); xx,yy=np.meshgrid(xs,ys)
    v=np.c_[xx.ravel(),yy.ravel(),np.full(xx.size,-.18)]; faces=[]
    for j in range(len(ys)-1):
        for i in range(len(xs)-1):
            k=j*len(xs)+i;faces += [(k,k+1,k+len(xs)),(k+1,k+len(xs)+1,k+len(xs))]
    rng=np.random.default_rng(34); cols=np.tile(np.array([193,123,75])/255,(len(faces),1))*rng.uniform(.98,1.025,(len(faces),1))
    np.savez_compressed(target/'terrain.npz',V=v.astype('f4'),F=np.array(faces,'i4'),C=cols.astype('f4'),wV=np.empty((0,3),'f4'),wF=np.empty((0,3),'i4'),wC=np.empty((0,3),'f4'),H=np.full((120,100),-.18),bbox=[x0,y0,x1,y1],cell=(x1-x0)/99)
    payload=dict(lay);payload['game_root']=str(Path(project.data.get('game_root') or defaults()['game_root']).expanduser())
    from source_scenery import overrides,route_witnesses
    payload['prop_overrides']=overrides(raw,Path(payload['game_root']),project.track_dir(module.__name__.split('.')[-1])/'planning',getattr(module,'SOURCE_CORRECTIONS',()))
    route_witnesses(raw,payload['prop_overrides'],project.track_dir(module.__name__.split('.')[-1])/'planning')
    payload['master_dir']=str(project.track_dir(module.__name__.split('.')[-1])/'blender')
    sightlines=project.track_dir(module.__name__.split('.')[-1])/'planning/scenery-sightlines.json'
    if not sightlines.exists():sightlines.write_text(json.dumps([v for k,v in source_views.cameras(lay).items() if 'loc' in v and k not in ('hero','isometric','birdseye')],indent=2))
    payload['scenery_sightlines']=json.loads(sightlines.read_text())
    (target/'layout.json').write_text(json.dumps(payload))
    (target/'masks.json').write_text('{}')
    (target/'props.json').write_text('{}')
    cams=source_views.cameras(lay)
    (target/'cams.json').write_text(json.dumps(cams,indent=2))
    return cams


def main():
    project=get_project();name=sys.argv[1];m=project.import_track(name)
    folder=project.track_dir(name); planning=folder/'planning';planning.mkdir(parents=True,exist_ok=True)
    # game_root is a default for the shared project selector; avoid inventing a second repository setting.
    game_root=Path(project.data.get('game_root') or defaults()['game_root']).expanduser()
    paths=[game_root/p for p in (m.SOURCE_TRACK,m.SOURCE_LIFE,m.SOURCE_OVERLAY)]
    raw=json.loads(paths[0].read_text()); val=source_audit(raw,m.EXPECTED);lay=normalize(raw,m)
    refs=sorted(folder.glob('iron-chorus-*v1.png'))
    base=dict(game_sources=[dict(path=str(p),sha256=digest(p)) for p in paths],references=[dict(file=p.name,sha256=digest(p)) for p in refs],original_photos=[dict(file=str(p.relative_to(folder)),sha256=digest(p)) for sub in ('01-Overview','02-Isometric','03-Highlights') for p in sorted((folder/sub).glob('*.png'))])
    baseline=planning/'preproduction-source-baseline.json'
    if not baseline.exists(): baseline.write_text(json.dumps(base,indent=2))
    (planning/'source-track.json').write_text(json.dumps(raw,indent=1))
    (planning/'track-layout.json').write_text(json.dumps(lay,indent=1))
    (planning/'validation.json').write_text(json.dumps(val,indent=2))
    (planning/'hazards.json').write_text(json.dumps(dict(traps=lay['traps']),indent=2))
    report=['# Iron Chorus – Quellgeometrie-Prüfung','',val['note'],'',f"{val['total']} checks; {val['fails']} failures.",'']
    report += [f"- **{r['status']} {r['id']}**: {r['detail']}" for r in val['checks']]
    report += ['', 'Die Quellwarnzeiten der Schmiedepressen betragen 0,9 s und sind kürzer als der konservative 1,2-s-Wert von Coastal. Sie wurden nicht still geändert. Ein Fahrtest und eine gesonderte Prüfung neuer Dekoration stehen aus.']
    (planning/'validation-report.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(dict(track=lay['id'],length_m=lay['length_m'],checks=val['total'],fails=val['fails'])))
    if val['fails']: raise SystemExit(2)
    import source_drivability
    drive=source_drivability.write(raw,planning)
    print('Theory: all hazard centreline witnesses clear =',drive['all_hazard_centerlines_clear'],'; sampled ramp landings on source road =',drive['all_sampled_landings_on_road'])
    bp=folder/'blueprints';bp.mkdir(exist_ok=True)
    for mode,file in [('clean','topdown-clean.png'),('annotated','topdown-annotated.png'),('pit','pitlane-detail.png'),('options','shortcuts-jumps.png'),('hazards','trap-positions.png')]: diagram(lay,bp/file,mode)
    cams=make_pack(lay,m,project,project.cache_dir/m.ID/'pack')
    (planning/'status.json').write_text(json.dumps(dict(status='GEOMETRY_VALIDATED',scope='source layout only',decoration_validated=False,integration_authorized=False),indent=2))
    print(f'Prepared pack, {len(cams)-1} cameras and five technical diagrams in {folder}')


if __name__=='__main__': main()
