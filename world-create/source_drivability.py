"""Conservative source-track witnesses. Not a Defold run or a vehicle simulation."""
import json
import math
import numpy as np


def audit(s):
    c=np.array([[p['x'],-p['z'],p.get('y',0)] for p in s['points']],float)
    q=np.roll(c,-1,axis=0);delta=q[:,:2]-c[:,:2];ll=np.maximum((delta*delta).sum(axis=1),1e-12)
    widths=np.array([p['width']+2*s['shoulder']-.5 for p in s['points']])
    witnesses=[]
    for h in s['timed_hazards']:
        half=widths[h['node']-1]/2;nearest=half-h['reach'];margin=nearest-.85-.30
        witnesses.append(dict(id=h['id'],core_width=s['points'][h['node']-1]['width'],source_shoulder=s['shoulder'],nearest_active_edge_to_center_m=round(nearest,3),kart_and_margin_m=1.15,centerline_clearance_m=round(margin,3),clear_for_all_phases=bool(margin>0),warning_s=h['telegraph']))
    jumps=[]
    for r in s['ramps']:
        i=r['node']-1;t=delta[i]/np.linalg.norm(delta[i]);nr=np.array([-t[1],t[0]])
        lip=c[i].copy();lip[:2]+=nr*r['offset']+t*r['length']/2;lip[2]+=r['height']
        trials=[]
        for speed in [18,22,25,30,34,38]:
            landing=None;g=9.81*1.65
            for time in np.arange(1/240,4,1/240):
                p=lip.copy();p[:2]+=t*speed*time;p[2]+=r['impulse']*time-.5*g*time*time
                u=np.clip(((p[:2]-c[:,:2])*delta).sum(axis=1)/ll,0,1)
                foot=c[:,:2]+delta*u[:,None];dist=np.linalg.norm(foot-p[:2],axis=1)
                half=(widths+(np.roll(widths,-1)-widths)*u)/2
                ground=c[:,2]+(q[:,2]-c[:,2])*u
                hit=(dist<=half-.85)&(p[2]<=ground+.01)
                if time>r['impulse']/g and hit.any():
                    j=int(np.where(hit)[0][np.argmin(dist[hit])]);forward=r['length']/2+speed*time
                    zone=next(z for z in s['landing_zones'] if z['ramp']==r['id'])
                    landing=dict(on_road=True,flight_s=round(float(time),3),forward_from_lip_m=round(speed*time,2),forward_from_ramp_origin_m=round(forward,2),in_bonus_zone=bool(zone['start']<=forward<=zone['finish'] and abs(float((p[:2]-c[i,:2])@nr)-zone['offset'])<=zone['width']/2),road_segment=j+1,xy=p[:2].round(3).tolist(),ground_z=round(float(ground[j]),3));break
                if p[2]<-.1:
                    landing=dict(on_road=False,flight_s=round(float(time),3),xy=p[:2].round(3).tolist());break
            trials.append(dict(speed_mps=speed,**(landing or dict(on_road=False,reason='no landing within 4 s'))))
        jumps.append(dict(id=r['id'],vertical_impulse_mps=r['impulse'],ramp_height_m=r['height'],trials=trials,all_samples_on_road=all(x['on_road'] for x in trials)))
    return dict(model='straight-flight source witness / 1',assumptions=dict(gravity_mps2=16.1865,kart_radius_m=.85,lateral_margin_m=.30,dt_s=1/240,speed_samples_mps=[18,22,25,30,34,38],horizontal_motion='constant source-tangent speed; no steering, drift, traffic or boost acceleration in flight',vertical_motion='source impulse applied at ramp lip; no slope-added impulse',surface='linear source road segments plus source shoulders, not new Blender collision meshes'),hazards=witnesses,jumps=jumps,all_hazard_centerlines_clear=all(x['clear_for_all_phases'] for x in witnesses),all_sampled_landings_on_road=all(j['all_samples_on_road'] for j in jumps),human_driving_test=False,engine_simulation=False)


def write(s,folder):
    report=audit(s)
    plan=[]
    for h in s['timed_hazards']:
        warning=max(1.2,h['telegraph']);idle=h['period']-warning-h['strike']-h['extended']-h['retract']
        plan.append(dict(id=h['id'],source_warning_s=h['telegraph'],planned_warning_s=warning,period_s=h['period'],strike_s=h['strike'],active_s=h['extended'],retract_s=h['retract'],remaining_idle_s=round(idle,3),scope='external timing proposal; not applied to game'))
    report['timing_proposal']=plan
    report['timing_proposal_passed']=all(p['planned_warning_s']>=1.2 and p['remaining_idle_s']>=0 for p in plan)
    (folder/'hazard-timing-proposal.json').write_text(json.dumps(dict(minimum_warning_s=1.2,passed=report['timing_proposal_passed'],hazards=plan,game_applied=False),indent=2))
    (folder/'drivability-audit.json').write_text(json.dumps(report,indent=2))
    rows=['# Iron Chorus – theoretischer Fahrbarkeitsnachweis','',
          'Modellwerte, kein Defold- oder menschlicher Fahrtest. Bestehende Straßen und Schultern, konstante Fluggeschwindigkeit, vertikaler Quellimpuls, keine Lenkung oder Fahrzeugdynamik in der Luft. Die Rampen sind um den Quellknoten zentriert; die Lippe liegt bei Länge/2 (wie in game/main/runtime.lua), nicht bei der vollen Länge.','',
          '## Freie Korridore','',
          'Eine 1,7 m breite Kart-Hülle plus 0,3 m Seitenreserve bleibt auf der Mittellinie in jeder Fallenphase außerhalb der Quell-Fußabdrücke. Das ist für diese bestehende Strecke ein statischer Zeuge; es behauptet keine getesteten Ausweichmanöver.','']
    rows += [f"- {h['id']}: {h['centerline_clearance_m']:.2f} m Restabstand; Vorwarnzeit {h['warning_s']:.3f} s." for h in report['hazards']]
    rows += ['', 'Für den neuen Vorproduktionsplan werden die fünf Pressenwarnungen auf 1,2 s angehoben. Die Periode bleibt 3,24 s; die Ruhephase verkürzt sich auf 0,86 s. Die übrigen Warnungen bleiben 1,216 s. Das ist eine dokumentierte Planung, keine Änderung am Spiel. Alle geplanten Phasendauern bleiben positiv.']
    rows += ['', '## Speaker-Rampen','', 'Die Straße unter und hinter den Rampen ist durchgehend. Es wurde keine zusätzliche Lücke erfunden. Eine Landung auf der Straße ist von der engeren Bonus-Landezone zu unterscheiden.','']
    for j in report['jumps']:
        rows += [f"### {j['id']}",'','| Tempo | Modell-Landung ab Rampenmitte | Auf Quellstraße | Bonuszone |','|---|---:|---|---|']
        rows += [f"| {x['speed_mps']} m/s | {x.get('forward_from_ramp_origin_m','–')} m | {'ja' if x['on_road'] else 'NEIN'} | {'ja' if x.get('in_bonus_zone') else 'nein'} |" for x in j['trials']]
    rows += ['', 'Vor einer Integration sind Rampenauslösung, Kamera, Kollisionen, Landefläche, Warnlesbarkeit, nasse/verschmutzte Beläge und Verkehr im tatsächlichen Spiel zu prüfen. Die alte Spielquelle hat weiterhin 0,9 s Pressenwarnung; die geplanten 1,2 s sind erst nach Freigabe zu integrieren und zu testen.']
    (folder/'drivability-audit.md').write_text('\n'.join(rows)+'\n')
    return report
