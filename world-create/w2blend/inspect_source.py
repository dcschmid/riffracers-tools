"""Read-only inspection of the saved external .blend, no rendering or game writes."""
import json
from pathlib import Path
import sys
import bpy
from mathutils import Vector

args=sys.argv[sys.argv.index('--')+1:];pack=Path(args[0]);planning=Path(args[1]);lay=json.loads((pack/'layout.json').read_text());cams=json.loads((pack/'cams.json').read_text());sc=bpy.context.scene
checks=[]
def add(name,ok,detail):checks.append(dict(id=name,passed=bool(ok),detail=detail))
views=[o for o in sc.objects if o.name.startswith('VIEW::') and o.type=='CAMERA']
add('camera_count',len(views)==len(cams)-1,f'{len(views)} named cameras; {len(cams)-1} expected')
for key,cd in cams.items():
    if key=='topdown':continue
    ob=sc.objects.get('VIEW::'+key)
    dynamic='--dynamic-cameras' in args and (key in ('festival-market','canyon-promenade') or cd.get('derived_from_scene',False))
    if dynamic:add('camera_'+key,ob is not None and bool(sc.get('lookdev_scope')),'camera derived from populated festival bounds; actual transform recorded separately')
    else:add('camera_'+key,ob is not None and (ob.location-Vector(cd['loc'])).length<.01,'saved position matches final camera pack')
add('canvas',(sc.render.resolution_x,sc.render.resolution_y)==(3840,2160),'3840 × 2160')
add('hero_active',sc.camera is not None and sc.camera.name=='VIEW::hero','hero selected')
add('colour_transform',sc.view_settings.view_transform=='AgX','AgX')
add('external_only','External visual prototype' in sc.get('preproduction_note',''),'explicit scene scope')
source=[o for o in sc.objects if 'source_asset' in o]
empty_platforms={'iron-pit-kart-stand','iron-pit-crew-platform'}  # empty workshop/platform scenery, no vehicle or person geometry
forbidden=[o['source_asset'] for o in source if Path(o['source_asset']).stem not in empty_platforms and any(s in Path(o['source_asset']).stem.lower() for s in ('kart','driver','racer','pit-crew'))]
add('no_kart_driver_assets',not forbidden,forbidden)
for name,pos in lay['prop_overrides'].items():
    ob=sc.objects.get('SRC::'+name);add('override_'+name,ob is not None and (ob.location-Vector(pos)).length<.01,'external correction saved')
ghost=sc.objects.get('ghost');add('flight_path_hidden',ghost is not None and ghost.hide_render,'model arc hidden in master')
fans=sum(o.name.startswith('FAN::') for o in sc.objects);add('faced_preview_crowd',fans>4000,f'{fans} preview spectators')
add('new_decor_recorded',len(json.loads(sc['new_decor_json']))>1000,'footprint reservations stored in master')
add('fonts_embedded',all(f.filepath=='<builtin>' or f.packed_file is not None for f in bpy.data.fonts),'font data embedded')
report=dict(file=bpy.data.filepath,checks=checks,passed=all(c['passed'] for c in checks),scene_objects=len(sc.objects),source_instances=len(source),source_asset_files=sorted({o['source_asset'] for o in source}),fan_instances=fans,no_game_write=True)
(planning/'blend-inspection.json').write_text(json.dumps(report,indent=2))
(planning/'cameras-resolved.json').write_text(json.dumps({o.name:dict(location=list(o.location),rotation=list(o.rotation_euler),lens=o.data.lens,file=o.get('output_file')) for o in views},indent=2))
print(json.dumps(report,indent=2))
if not report['passed']:raise RuntimeError('Saved Blender inspection failed')
