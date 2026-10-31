import sys, json, importlib, time
from world_create import get_project
PROJECT = get_project()
from w2lib.track import Track
from w2lib import validate as V
from w2lib import mapkit

def run(mod, out=None, preview=True, verbose=True):
    out = str(PROJECT.cache_dir) if out is None else out
    __import__("pathlib").Path(out).mkdir(parents=True, exist_ok=True)
    m = importlib.import_module(f"tracks.{mod}")
    T = m.make()
    t0 = time.time()
    lay = T.build()
    res = V.validate(lay)
    if verbose:
        print(f"{lay['name']}: {lay['length_m']} m, laps {json.dumps({k: v['time_s'] for k, v in lay['lap_times'].items()})}")
        for c in res["checks"]:
            print(f"  [{c['status']:4}]{'*' if c['critical'] else ' '} {c['id']:2} {c['name'][:70]} :: {c['detail'][:260]}")
        print(f"  critical={res['critical']} warnings={res['warnings']} fails={res['fails']}  ({time.time()-t0:.1f}s)")
    if preview:
        mapkit.quick_preview(lay, f"{out}/{lay['id']}-preview.png")
    return T, lay, res

if __name__ == "__main__":
    run(sys.argv[1])
