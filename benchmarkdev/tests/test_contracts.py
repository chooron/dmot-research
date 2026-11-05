"""GMD verification: registry/forward/parameter mapping/data contract/provenance.
Tolerance is fixed in config/config.json before running, never relaxed here.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
DEV=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(DEV/'src'))
from benchmarkdev.core import *
PARITY_ROWS=[]
@pytest.fixture(scope='session',autouse=True)
def export_parity():
    protect_sources()
    yield
    write_csv('data/B_parity.csv',PARITY_ROWS)
    register_claims()

@pytest.mark.parametrize('name',MODELS)
def test_registry_and_minimal_forward(name):
    """Model description / verification of callable public interface."""
    m=build(name)
    raw=torch.full((1,len(PARAM_INFO[name])),.5,dtype=torch.float64)
    x,_=forcing(all_ids()[:1],start='1989-01-01',end='1989-01-03')
    with torch.no_grad():q=m({'x_phy':x},(None,raw))['streamflow']
    assert q.shape==(3,1)
    assert torch.isfinite(q).all()

@pytest.mark.parametrize('name',MODELS)
def test_original_adapter_parity(name):
    """Asset migration: unchanged original build_model and new adapter same inputs."""
    from src.model_registry import build_model
    old=build_model(name,'cpu',warm_up=0,backend='eager',dtype=torch.float64)
    new=build(name)
    x,_=forcing(all_ids()[:1],start='1989-01-01',end='1989-01-03')
    raw=torch.full((1,len(PARAM_INFO[name])),.5,dtype=torch.float64)
    with torch.no_grad():a=old({'x_phy':x},(None,raw))['streamflow'];b=new({'x_phy':x},(None,raw))['streamflow']
    exact=bool(torch.equal(a.reshape(-1).contiguous().view(torch.uint8),b.reshape(-1).contiguous().view(torch.uint8)))
    PARITY_ROWS.append({'model':name,'max_absolute_difference':float((a-b).abs().max()),'bit_exact':exact,'status':'TRUE' if exact else 'FALSE','original_source':'project/benchmark/src/model_registry.py','new_source':'src/benchmarkdev/core.py'})
    assert exact

def test_subset_partition():
    """Verification sampling is based on regions, never on performance."""
    f=subset();assert len(f)==CFG['subset_size'];assert f.basin_id.nunique()==len(f)
    assert set(f.region)=={f'group_{i}' for i in range(11,18)}
    pd.testing.assert_frame_equal(f,subset())

def test_loss_public_identity():
    """Verification KGE wrapper calls public loss, not a copied formula."""
    x=torch.tensor([[1.,2.],[2.,3.],[3.,4.]],dtype=torch.float64)
    y=x+0.1
    assert torch.equal(kge(x,y),1-KgeLoss()(x,y,reduction='none'))

def test_claim_registry_matches_files():
    """Manuscript reproducibility: every registered cell has a unique source."""
    import csv
    seen=set();commits=set();current_source=None;data=None;digest=None
    with (MAN/'claims_registry.csv').open(newline='') as handle:
        for r in csv.DictReader(handle):
            assert r['claim_id'] not in seen;seen.add(r['claim_id']);commits.add(r['dmotpy_commit'])
            source=r['source_file']
            if source!=current_source:
                p=DEV/source
                digest=hashlib.sha256(p.read_bytes()).hexdigest()
                with p.open(newline='') as f:data=list(csv.DictReader(f))
                current_source=source
            assert digest==r['source_sha256']
            assert data[int(r['source_row'])][r['source_column']]==r['value']
    assert commits=={SHA}

@pytest.mark.parametrize('stage',['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9'])
def test_analysis_coverage(stage):
    """Paper-specific coverage: skipped results are not accepted as completed analyses."""
    state=json.loads((DEV/'cache/status.json').read_text()).get(stage)
    if not state or state['status'] in {'PARTIAL','NOT_AVAILABLE','NOT_APPLICABLE'}:
        pytest.skip(str(state))
    if state['status']=='FAIL':pytest.fail(state['reason'])
    assert state['status']=='DONE'
