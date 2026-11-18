"""Default prints the exact native benign query; owner must opt in to model-backed tools."""
import argparse
import json
from pathlib import Path
import sys
from contract import require, sha, write_new

def main():
    p=argparse.ArgumentParser();p.add_argument('--harness',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--execute-owner',action='store_true');a=p.parse_args()
    url='http://localhost:8010/retroplanner/api/single_step'
    rec=dict(url=url,smiles='CCOC(=O)c1ccccc1',model_name='GraphFP (USPTO-Full)',topk=1,
             source_sha256=sha(a.harness/'tools/_retroplanner_http.py'))
    if not a.execute_owner:print(json.dumps(rec,indent=2));return
    require(a.out.resolve().is_relative_to(Path(__file__).resolve().parents[1]) and not a.out.exists(),'new runtime receipt required')
    sys.path.insert(0,str(a.harness))
    from tools._retroplanner_http import RetroplannerHTTPClient
    rows=RetroplannerHTTPClient(url=url,timeout_s=180,retries=0).predict(smiles=rec['smiles'],model_name=rec['model_name'],topk=1)
    require(rows and all(x.get('reactants') for x in rows),'native relay query failed')
    write_new(a.out,dict(rec,native_query_passed=True,predictions=rows))
if __name__=='__main__':main()
