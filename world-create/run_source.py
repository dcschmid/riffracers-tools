"""Render a validated existing-source track using the shared external Blender pipeline.

Examples: python run_source.py --world ../World-01-Metal t01_iron_chorus quick hero,stage
          python run_source.py --world ../World-01-Metal t01_iron_chorus hq
The shared GPU lock covers every Blender process. Quick renders stay in the cache.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
from PIL import Image
from world_create import get_project,gpu_job


def main():
    p=get_project();mod=p.import_track(sys.argv[1]);mode=sys.argv[2]
    track=p.track_dir(sys.argv[1]);base=p.cache_dir/mod.ID;pack=base/'pack'
    validation=json.loads((track/'planning/validation.json').read_text())
    if validation['fails']:raise SystemExit('Geometry validation failed')
    cams=json.loads((pack/'cams.json').read_text());shots=sys.argv[3] if len(sys.argv)>3 else ','.join(k for k in cams if k!='topdown')
    if mode not in ('quick','hq','mask','grade'):raise SystemExit('Choose quick, hq, mask or grade')
    out=track/'blender/raw-renders' if mode in ('hq','grade') else track/'renders' if mode=='mask' else base/'preview-final'
    out.mkdir(parents=True,exist_ok=True);env=dict(os.environ,RIFF_WORLD_ROOT=str(p.root))
    if mode!='hq':env['W2_NO_SAVE']='1'
    if mode=='quick':env['W2_QUICK']='1'
    samples='24' if mode=='quick' else '96'
    command=[p.tool('blender','blender'),'-b','--python-exit-code','2','-P','w2blend/scene.py','--',str(pack),mod.ENV,str(out),'mask' if mode=='mask' else 'render',shots,samples]
    log=base/(mode+'-render.log')
    print(f'{mode}: {shots} → {out}',flush=True)
    if mode!='grade':
        with gpu_job() as lock_fd,log.open('w') as f:
            result=subprocess.run(command,cwd=Path(__file__).parent,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock_fd,))
        print(log.read_text()[-1600:],flush=True)
        if result.returncode:raise SystemExit(result.returncode)
    if mode=='mask':
        if not (out/'roadmask.png').is_file():raise RuntimeError('Missing road mask')
        return
    for key in shots.split(','):
        if key=='topdown':continue
        path=out/(cams[key]['file']+'.png')
        with Image.open(path) as im:
            expected=(1920,1080) if mode=='quick' else tuple(cams[key]['res'])
            if im.size!=expected:raise RuntimeError(f'Unexpected size: {path}: {im.size}')
        if mode in ('hq','grade'):
            from grade3 import grade
            grade(path,track/path.name,sat=1.16,contrast=.08,bloom=.15,vig=.10,warm=.02)
    print('Verified render files'+(' and graded into track root' if mode in ('hq','grade') else ''),flush=True)


if __name__=='__main__':main()
