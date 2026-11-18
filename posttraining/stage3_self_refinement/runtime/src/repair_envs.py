"""Create new isolated wrappers over preserved packages, no installs or shared edits."""
import argparse
import os
import shutil
from pathlib import Path
import subprocess
from contract import require, sha, write_new

# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
ROOT=Path(__file__).resolve().parents[1]
# Python 3.12.14 interpreter used to create the wrappers (version checked below).
PY=Path(os.environ.get('RETRACE_PYTHON312', 'python3.12'))
OLD=Path(RETRACE_SHARE + '/retrace-continual')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    out=a.out.resolve(); require(out.is_relative_to(ROOT) and not out.exists(),'new runtime env root required')
    out.mkdir(parents=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',
             TRANSFORMERS_OFFLINE='1',XDG_CACHE_HOME=str(out/'cache'),HF_HOME=str(out/'hf'),
             TRITON_CACHE_DIR=str(out/'triton'),NUMBA_CACHE_DIR=str(out/'numba'),MPLCONFIGDIR=str(out/'mpl'))
    require(PY.is_file() or shutil.which(str(PY)), 'set RETRACE_PYTHON312 to a Python 3.12.14 interpreter')
    version=subprocess.check_output([str(PY),'-I','-B','--version'],text=True).strip()
    require(version=='Python 3.12.14','wrong interpreter version')
    receipts={}
    for name,old in [('eval','venv'),('serving','venv-vllm')]:
        dest=out/name
        subprocess.run([str(PY),'-I','-B','-m','venv','--without-pip',str(dest)],check=True,env=env)
        site=OLD/old/'lib/python3.12/site-packages'
        pth=dest/'lib/python3.12/site-packages/zz_preserved.pth'
        pth.write_text('import site; site.addsitedir('+repr(str(site))+')\n')
        probe=("import json,sys,importlib.metadata as m; import torch, transformers; "
               + ('import rdkit; ' if name=='eval' else 'import vllm; ')
               + "print(json.dumps({'python':sys.version,'prefix':sys.prefix,'cuda_initialized':torch.cuda.is_initialized(),"
               + "'versions':{k:m.version(k) for k in "
               + repr(['torch','transformers','rdkit'] if name=='eval' else ['torch','transformers','vllm']) + "}}))")
        result=subprocess.run([str(dest/'bin/python'),'-B','-c',probe],env=env,text=True,capture_output=True)
        (out/(name+'-imports.log')).write_text(result.stdout+result.stderr)
        receipts[name]={'returncode':result.returncode,'pth_sha256':sha(pth),'preserved_site_packages':str(site)}
        require(result.returncode==0,'isolated import failed: '+name)
        if name=='serving':
            result=subprocess.run([str(dest/'bin/python'),'-B',str(OLD/old/'bin/vllm'),'serve','--help'],
                                  env=env,text=True,capture_output=True)
            (out/'serving-help.log').write_text(result.stdout+result.stderr)
            receipts[name]['help_returncode']=result.returncode
            require(result.returncode==0,'serving entrypoint help failed')
    write_new(out/'ENV_REPORT.json',dict(python=version,interpreter_sha256=sha(PY),environments=receipts,
        installs=0,shared_mutations=0,weights_loaded=False,model_calls=0,gpu_kernel_validation=False))

if __name__=='__main__': main()
