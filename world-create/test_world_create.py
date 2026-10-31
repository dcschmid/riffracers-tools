"""Project routing/containment regressions. No GPU or game writes.

Two projects deliberately use the same module/id, exercising independent imports
and outputs through the actual CLI. Temporary fixtures are inspected and trashed
with gio to respect the project's reversible-cleanup rule.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from world_create import Project, WORLD_CREATE_ROOT, contained

TRACK = '''
from w2lib.track import Track
LABEL = {label!r}
OUT_FOLDER = "wrong-module-default"
def make():
    t=Track("same-id",LABEL,"Test",2,"CCW","coastal",half=230)
    p=t.P
    p.line(230,"a").arc(80,90,"b").line(200,"c").arc(80,90,"d").line(460,"e").arc(80,90,"f").line(200,"g").arc(80,90,"h").line(230,"i")
    t.w_st=[("a",0,20),("i",1,20)]
    t.sections=[("intro",("a",0)),("verse",("b",0)),("build_up",("c",0)),("drop",("d",0)),("breakdown",("e",0)),("finale",("g",0))]
    return t
'''


class ProjectTests(unittest.TestCase):
    def setUp(self):
        # /tmp is an internal mount here and cannot use gio trash. Keep fixtures
        # on the user's filesystem so cleanup remains reversible.
        parent = WORLD_CREATE_ROOT / ".test-fixtures"
        parent.mkdir(exist_ok=True)
        self.temp = Path(tempfile.mkdtemp(prefix="riff-pipeline-test-", dir=parent))

    def tearDown(self):
        files = sorted(str(p.relative_to(self.temp)) for p in self.temp.rglob("*") if p.is_file())
        print("Inspected temporary fixture before gio trash:", self.temp, files)
        subprocess.run(["gio", "trash", "--", str(self.temp)], check=True)

    def world(self, name):
        root = self.temp/name
        (root/"_project/tracks").mkdir(parents=True)
        (root/"_project/tracks/__init__.py").write_text("")
        (root/"_project/tracks/demo.py").write_text(TRACK.format(label=name))
        data=dict(schema="riff-racers.world-project/1",id=name,name=name,code_dir="_project",cache_dir="_cache",
                  tracks=[dict(module="demo",id="same-id",folder="track-output")])
        (root/"riff-world.json").write_text(json.dumps(data))
        return root

    def cli(self, world, script, *args, env=None):
        result=subprocess.run([sys.executable,str(WORLD_CREATE_ROOT/script),"--world",str(world),*args],cwd=WORLD_CREATE_ROOT,
                              env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        return result.stdout

    def test_two_worlds_import_and_write_independently(self):
        alpha,beta=self.world("alpha"),self.world("beta")
        self.cli(alpha,"run_stage.py","demo","layout")
        first=(alpha/"track-output/planning/track-layout.json").read_bytes()
        self.cli(beta,"run_stage.py","demo","layout")
        a=json.loads(first);b=json.loads((beta/"track-output/planning/track-layout.json").read_text())
        self.assertEqual(a["id"],b["id"])
        self.assertEqual(a["name"],"alpha");self.assertEqual(b["name"],"beta")
        self.assertEqual((alpha/"track-output/planning/track-layout.json").read_bytes(),first)
        self.assertFalse((alpha/"wrong-module-default").exists())
        self.assertFalse((beta/"wrong-module-default").exists())

    def test_explicit_world_wins_over_environment(self):
        alpha,beta=self.world("alpha"),self.world("beta")
        env=dict(os.environ,RIFF_WORLD_ROOT=str(beta))
        info=json.loads(self.cli(alpha,"world_create.py","info",env=env))
        self.assertEqual(info["root"],str(alpha))

    def test_environment_selects_world(self):
        alpha=self.world("alpha")
        result=subprocess.run([sys.executable,str(WORLD_CREATE_ROOT/"world_create.py"),"root"],cwd=WORLD_CREATE_ROOT,
                              env=dict(os.environ,RIFF_WORLD_ROOT=str(alpha)),capture_output=True,text=True,check=True)
        self.assertEqual(result.stdout.strip(),str(alpha))

    def test_manifest_file_argument(self):
        alpha=self.world("alpha")
        self.assertEqual(self.cli(alpha/"riff-world.json","world_create.py","root").strip(),str(alpha))

    def test_unconfigured_world_does_not_fall_back_to_another_world(self):
        world=self.temp/"World-99-Unconfigured";world.mkdir()
        env=dict(os.environ,PWD=str(world));env.pop("RIFF_WORLD_ROOT",None)
        result=subprocess.run([sys.executable,str(WORLD_CREATE_ROOT/"world_create.py"),"root"],cwd=world,
                              env=env,capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn("has no riff-world.json",result.stderr)

    def test_parent_and_absolute_output_paths_are_rejected(self):
        alpha=self.world("alpha")
        for invalid in ("../escape",str(self.temp/"outside")):
            data=json.loads((alpha/"riff-world.json").read_text());data["tracks"][0]["folder"]=invalid
            (alpha/"riff-world.json").write_text(json.dumps(data))
            with self.assertRaises(ValueError):Project.load(alpha)

    def test_symlink_escape_is_rejected(self):
        alpha=self.world("alpha");outside=self.temp/"outside";outside.mkdir()
        (alpha/"linked").symlink_to(outside,target_is_directory=True)
        with self.assertRaises(ValueError):contained(alpha,"linked/output")

    def test_cache_escape_is_rejected(self):
        alpha=self.world("alpha");data=json.loads((alpha/"riff-world.json").read_text());data["cache_dir"]="../cache"
        (alpha/"riff-world.json").write_text(json.dumps(data))
        with self.assertRaises(ValueError):Project.load(alpha)

    def test_moss_references_are_required_and_stay_in_the_track(self):
        alpha=self.world("alpha")
        definition=alpha/"_project/tracks/demo.py"
        for sources, message in ((None, "own HQ_SOURCES"),
                                 ({key:"concepts/../../../outside.png" for key in ("environment","pitlane","trap","crowd")}, "Path must stay inside")):
            with self.subTest(sources=sources):
                definition.write_text(TRACK.format(label="alpha") + (f"\nHQ_SOURCES={sources!r}\n" if sources else ""))
                result=subprocess.run([sys.executable,str(WORLD_CREATE_ROOT/"w2lib/concepts_layout.py"),"--world",str(alpha),"demo"],
                                      cwd=WORLD_CREATE_ROOT,capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0)
                self.assertIn(message,result.stderr)
                self.assertFalse((alpha/"track-output").exists())

    def test_flat_concepts_publish_real_images_in_the_track_root(self):
        from PIL import Image
        alpha=self.world("alpha")
        definition=alpha/"_project/tracks/demo.py"
        definition.write_text(TRACK.format(label="alpha") + '''
HQ_SOURCES={key:"reference.png" for key in ("environment","pitlane","trap","crowd")}
CONCEPT_DIR="."
FIDELITY_FILE="planning/layout-fidelity.json"
CONCEPTS=dict(environment="test",pitlane="test",tiles={})
PALETTE={"test":"#33aacc"}
TITLE="test"
SUBTITLE="test"
''')
        output=alpha/"track-output"
        (output/"planning").mkdir(parents=True)
        (output/"planning/hazards.json").write_text(json.dumps(dict(traps=[dict(name="test",visual="test")])) )
        Image.new("RGB",(32,18),"#33aacc").save(output/"reference.png")
        source=f'''import sys, shutil
sys.path.insert(0,{str(WORLD_CREATE_ROOT/'w2lib')!r})
import concepts_layout as c
def cpu_repaint(init,prompt,out,seed):
    shutil.copy2(init,out)
    return (0.38,0.9)
c.repaint=cpu_repaint
c.run("demo","track-output")
'''
        result=subprocess.run([sys.executable,"-c",source,"--world",str(alpha)],cwd=WORLD_CREATE_ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        for name in ("environment-concept.png","pitlane-concept.png","signature-props.png","crowd-concept.png","moodboard.png"):
            self.assertTrue((output/name).is_file(),name)
            self.assertFalse((output/name).is_symlink(),name)
        self.assertTrue((output/"planning/layout-fidelity.json").is_file())
        self.assertFalse((output/"concepts").exists())
        self.assertFalse((output/"_layout-tiles").exists())
        self.assertFalse((output/"_v1-text-only").exists())

    def test_world_specific_blender_namespace(self):
        alpha=self.world("alpha");(alpha/"_project/w2blend").mkdir()
        (alpha/"_project/w2blend/local_theme.py").write_text("MARKER='alpha'\n")
        source="from world_create import get_project; get_project(); from w2blend import local_theme; import importlib.util; assert local_theme.MARKER=='alpha'; assert importlib.util.find_spec('w2blend.core') is not None; print('namespace OK')"
        result=subprocess.run([sys.executable,"-c",source,"--world",str(alpha)],cwd=WORLD_CREATE_ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn("namespace OK",result.stdout)


if __name__=="__main__":unittest.main()
