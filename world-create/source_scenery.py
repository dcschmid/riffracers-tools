"""CPU-only, external corrections for legacy scenery intruding into source routes."""
import json
import sys
import numpy as np
from shapely.geometry import LineString, MultiPoint, Polygon
from shapely.ops import unary_union


def overrides(s,game_root,planning,corrections=()):
    # Reuse the game's read-only glTF sampler, without creating __pycache__ there.
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(game_root/'tools/asset_import'))
    import ghost_colliders
    source_roads=[]
    p=s['points'];c=np.array([[q['x'],-q['z']] for q in p])
    source_roads.append(LineString(np.vstack([c,c[:1]])).buffer(max(q['width'] for q in p)/2+s['shoulder']))
    for r in s['shortcuts']:
        source_roads.append(LineString([[q['x'],-q['z']] for q in r['points']]).buffer(r['width']/2))
    pit=s['pitlane'];source_roads.append(LineString([[q['x'],-q['z']] for q in pit['points']]).buffer(pit['width']/2))
    road=unary_union(source_roads);out={};report=[]
    for ident,route_id in corrections:
        item=next(l for l in s['landmarks'] if l['id']==ident);route=next(r for r in s['shortcuts'] if r['id']==route_id)
        v=ghost_colliders.vertices(item['asset']);scale=item.get('scale',1);scale=[scale]*3 if isinstance(scale,(int,float)) else scale
        v=v*np.array(scale);yaw=item.get('yaw',0);co,si=np.cos(yaw),np.sin(yaw)
        xy=np.c_[co*v[:,0]+si*v[:,2],si*v[:,0]-co*v[:,2]]
        pos=item['position'];base=np.array([pos['x'],-pos['z']]);points=np.array([[q['x'],-q['z']] for q in route['points']])
        nearest=points[np.linalg.norm(points-base,axis=1).argmin()];away=base-nearest;away/=np.linalg.norm(away)
        found=False
        for step in np.arange(0,80,.5):
            for angle in np.linspace(0,2*np.pi,16,endpoint=False):
                co,si=np.cos(angle),np.sin(angle);direction=np.array([co*away[0]-si*away[1],si*away[0]+co*away[1]])
                p=base+direction*step;foot=MultiPoint(xy+p).convex_hull;clearance=foot.distance(road)
                if clearance>=1.4:found=True;break
            if found:break
        else:raise ValueError('No free external location for '+ident)
        out[ident]=[float(p[0]),float(p[1]),pos.get('y',0)]
        report.append(dict(id=ident,reason='legacy source mesh intersects '+route_id,source_blender_xy=base.tolist(),external_blender_xy=p.tolist(),translation_m=round(float(step),2),conservative_all_roads_clearance_m=round(clearance,3)))
    (planning/'scenery-overrides.json').write_text(json.dumps(dict(scope='external Blender preview only; game source unchanged',overrides=report),indent=2))
    return out


def route_witnesses(s,overrides,planning):
    from shapely import contains_xy
    witnesses=[]
    for route in s['shortcuts']:
        c=np.array([[p['x'],-p['z'],p['y']] for p in route['points']]);length=np.linalg.norm(np.diff(c[:,:2],axis=0),axis=1);arc=np.r_[0,np.cumsum(length)]
        stations=np.linspace(0,arc[-1],int(np.ceil(arc[-1]/.5))+1);center=np.array([np.interp(stations,arc,c[:,j]) for j in range(3)]).T
        tangent=np.gradient(center[:,:2],axis=0);tangent/=np.linalg.norm(tangent,axis=1)[:,None];normals=np.c_[-tangent[:,1],tangent[:,0]]
        limit=route['width']/2-1.15;lateral=np.linspace(-limit,limit,2*int(np.floor(limit/.1))+1)
        xy=center[:,None,:2]+normals[:,None,:]*lateral[None,:,None]
        line=LineString(center[:,:2]);obstacles=[];names=[]
        for item in s['landmarks']:
            pp=item['position'];position=overrides.get(item['id'],[pp['x'],-pp['z'],pp['y']]);co,si=np.cos(item.get('yaw',0)),np.sin(item.get('yaw',0));rotation=np.array([[co,si],[si,-co]])
            for box in item.get('collision_boxes',[]):
                b=box['position'];w,h,d=box['size'];zlo=position[2]+b['y']-h/2;zhi=zlo+h
                if zlo>2 or zhi<.15:continue
                corners=np.array([[-w/2,-d/2],[w/2,-d/2],[w/2,d/2],[-w/2,d/2]])+[b['x'],b['z']]
                poly=Polygon(corners@rotation.T+position[:2]).buffer(1.15)
                if poly.distance(line)<limit:obstacles.append(poly);names.append(item['id']+'::'+box['id'])
        blocked=contains_xy(unary_union(obstacles),xy[:,:,0],xy[:,:,1]) if obstacles else np.zeros(xy.shape[:2],bool)
        N=len(lateral);previous=np.full((len(stations),N),-1,int);cost=np.where(np.abs(lateral)<.2,lateral*lateral,np.inf);cost[blocked[0]]=np.inf
        # One 0.1 m state per ~0.5 m forward step: <=4.5 m/s lateral at 22 m/s forward.
        max_shift=6/22*(stations[1]-stations[0]);indices=np.arange(N)
        for k in range(1,len(stations)):
            new=np.full(N,np.inf)
            for j in range(N):
                allowed=(np.abs(lateral-lateral[j])<=max_shift+1e-8)&np.isfinite(cost)
                if blocked[k,j] or not allowed.any():continue
                candidates=np.where(allowed)[0];best=int(candidates[np.argmin(cost[allowed]+(lateral[allowed]-lateral[j])**2*5)])
                new[j]=cost[best]+lateral[j]**2*.01+(lateral[best]-lateral[j])**2*5;previous[k,j]=best
            cost=new
        cost[np.abs(lateral)>.4]=np.inf;passed=bool(np.isfinite(cost).any());path=[]
        if passed:
            j=int(cost.argmin())
            for k in range(len(stations)-1,-1,-1):
                path.append(dict(station_m=round(float(stations[k]),3),lateral_m=round(float(lateral[j]),3),xyz=np.r_[xy[k,j],center[k,2]].round(3).tolist()));j=int(previous[k,j]) if k else 0
            path.reverse()
        witnesses.append(dict(id=route['id'],passed=passed,width_m=route['width'],length_m=float(arc[-1]),nearby_collision_boxes=names,witness=path))
    report=dict(scope='external preview plus source authored collision boxes; static kinematic witness, no engine simulation',kart_radius_m=.85,margin_m=.30,forward_speed_mps=22,lateral_speed_limit_mps=6,station_step_max_m=.5,all_routes_have_witness=all(r['passed'] for r in witnesses),routes=witnesses)
    (planning/'shortcut-witnesses.json').write_text(json.dumps(report,indent=1))
    if not report['all_routes_have_witness']:raise ValueError('No theoretical shortcut corridor witness')
    return report
