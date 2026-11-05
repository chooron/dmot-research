"""Read-only adapters. Sources: benchmark/src/{model_registry,data_selection}.py,
benchmark/dpl/attributes.py; hydrological equations are imported, never copied.
"""
from __future__ import annotations
import ast, csv, hashlib, importlib.util, inspect, json, os, sys, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch
DEV = Path(__file__).resolve().parents[2]
CFG = json.loads((DEV / 'config/config.json').read_text())
UPSTREAM_ROOT = Path(CFG['root'])
ROOT = DEV.parent if (DEV.parent / 'dmotpy').exists() else UPSTREAM_ROOT
MAN = DEV / 'manuscript'
param_dir = DEV.parent / 'benchmarkparam' if (DEV.parent / 'benchmarkparam').exists() else ROOT / 'project/benchmark'
sys.path[:0] = [str(ROOT), str(param_dir), str(param_dir / 'src')]
from dmotpy.models.registry import PARAM_INFO, STATE_INFO, STFN_INFO, INIT_INFO
from dmotpy.models.hydrology_model import HydrologyModel
from dmotpy.models.endpoint_uh_model import ENDPOINT_UH_SCHEMES
from dmotpy.models.intermediate_uh_model import INTERMEDIATE_UH_CONFIG
from dmotpy.losses import KgeLoss
from src.model_registry import model_config
from src.data_selection import frozen_parameters
from dpl.attributes import CatchmentAttributeBuilder, CAMELS_35_ATTRIBUTES
MODELS = sorted(PARAM_INFO)
try:
    SHA = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
except Exception:
    SHA = subprocess.check_output(['git', '-C', str(UPSTREAM_ROOT), 'rev-parse', 'HEAD'], text=True).strip()
BUNDLE = None

def protect_sources():
    protected = [(ROOT / p).resolve() for p in CFG['read_only']]
    def denied(path):
        if not isinstance(path, (str, bytes, os.PathLike)): return False
        p = Path(os.fsdecode(path)).resolve()
        return any(p == r or r in p.parents for r in protected)
    def audit(event, args):
        if event == 'open':
            path, mode, flags = args
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (flags is not None and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            if writing and denied(path): raise PermissionError('READ_ONLY_WRITE_ATTEMPT: ' + str(path))
        elif event in {'os.remove','os.rmdir','os.mkdir','os.chmod','os.utime','os.rename','os.link','os.symlink'}:
            for p in args[:2]:
                if denied(p): raise PermissionError('READ_ONLY_WRITE_ATTEMPT: ' + str(p))
    sys.addaudithook(audit)

def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def write_csv(relative, rows, columns=None):
    p = MAN / relative
    if isinstance(rows, pd.DataFrame): frame = rows
    else: frame = pd.DataFrame(rows, columns=columns)
    p.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(p, index=False, float_format='%.17g')
    producer_path=DEV/'cache/producers.json'
    producers=json.loads(producer_path.read_text()) if producer_path.exists() else {}
    caller=Path(inspect.currentframe().f_back.f_code.co_filename).resolve()
    producers[str(p.relative_to(MAN))]=str(caller.relative_to(DEV)) if DEV in caller.parents else str(caller)
    producer_path.write_text(json.dumps(producers,indent=2))
    return p

def all_ids():
    return np.asarray(ast.literal_eval((ROOT/'data/531sub_id.txt').read_text()), dtype=np.int64)

def subset(count=None):
    ids = all_ids(); count = count or CFG['subset_size']
    groups = [np.intersect1d(ids, np.load(ROOT/f'data/basin_groups/group_{i}.npy').astype(np.int64).ravel()) for i in range(11,18)]
    merged = np.concatenate(groups)
    if len(np.unique(merged)) != len(merged) or set(merged) != set(ids):
        raise ValueError('Seven-region groups do not partition the frozen basin list')
    quotas = np.array([len(g) for g in groups])*count/len(ids)
    allocated = np.floor(quotas).astype(int)
    order = np.argsort(-(quotas - allocated), kind='stable')
    allocated[order[:count-int(allocated.sum())]] += 1
    rng = np.random.default_rng(CFG['seed']); rows = []
    for i,g,n in zip(range(11,18),groups,allocated):
        for basin in sorted(rng.choice(g,int(n),replace=False)):
            rows.append({'basin_id':str(int(basin)).zfill(8),'region':f'group_{i}','seed':CFG['seed'],'region_population':len(g),'region_selected':int(n)})
    return pd.DataFrame(rows)

def bundle():
    global BUNDLE
    if BUNDLE is None:
        import pickle
        with (ROOT/'data/camels_dataset').open('rb') as f: BUNDLE = pickle.load(f)
    return BUNDLE

def forcing(ids, start=None, end=None, prefix=0, dtype=torch.float64, device='cpu'):
    f,q,a = bundle()
    reference = np.load(ROOT/'data/gage_id.npy').astype(np.int64)
    idx = np.asarray([np.where(reference == int(b))[0][0] for b in ids])
    dates = pd.date_range('1980-10-01', '2014-09-30')
    left = dates.get_loc(pd.Timestamp(start or CFG['evaluation_start']))
    right = dates.get_loc(pd.Timestamp(end or CFG['evaluation_end']))+1
    if left < prefix: raise ValueError('insufficient warmup dates')
    x = f[idx,left-prefix:right,:3]
    doy = np.broadcast_to(dates[left-prefix:right].dayofyear.to_numpy()[None,:,None],(*x.shape[:2],1))
    x = np.concatenate([x,doy],axis=-1).transpose(1,0,2).copy()
    y = q[idx,left:right,0].astype(np.float64).copy()
    y *= (0.0283168*86400*1000/(a[idx,11]*1e6))[:,None]
    return torch.as_tensor(x,dtype=dtype,device=device), torch.as_tensor(y.T,dtype=dtype,device=device)

def build(name,warm=0,mapping='linear',dtype=torch.float64,device='cpu',backend='eager'):
    cfg = model_config(name,warm_up=warm,backend=backend,parameter_mapping=mapping)
    m = HydrologyModel(cfg,device=torch.device(device),backend=backend)
    m.compute_dtype = dtype
    return m.to(device=device,dtype=dtype)

def ic(name, ids=None):
    d = ROOT / CFG['ic_checkpoint_root'] / name
    if not d.exists(): d = UPSTREAM_ROOT / CFG['ic_checkpoint_root'] / name
    gen = 280 if name == 'simhyd' else 300
    files = sorted(d.glob(f'chunk_*_gen_{gen}.pt'))
    if not files: raise FileNotFoundError(f'No frozen generation {gen} checkpoint: {d}')
    # Verify starts from stored shape rather than assuming a number.
    sample = torch.load(files[0],map_location='cpu',weights_only=False)
    starts = sample['solver']['state']['best_fitness'].numel()//len(sample['basin_ids'])
    basin, latent, score = frozen_parameters(d,gen,starts)
    norm = torch.sigmoid(latent.to(torch.float64))
    if ids is not None:
        index = [np.where(basin == int(i))[0][0] for i in ids]
        return norm[index]
    return basin,norm

def dpl(name,ids,kind='seen'):
    if kind != 'seen': raise FileNotFoundError('OOB checkpoint selection requires fold-specific contract; no fallback to seen parameters')
    p = ROOT / 'benchmarkparam/results/dpl_canonical_v2_20260831/runs' / name / 'best.pt'
    if not p.exists(): p = UPSTREAM_ROOT / 'project/benchmark/results/dpl_canonical_v2_20260831/runs' / name / 'best.pt'
    payload = torch.load(p,map_location='cpu',weights_only=False)
    from dpl.nn_parameterizer import CatchmentParameterizer
    # Producer canonical_v2/run_canonical_v2_model.py builds Dropout(0.05),
    # even in eval mode its slots are essential to strict checkpoint binding.
    network = CatchmentParameterizer(in_features=len(CAMELS_35_ATTRIBUTES),out_features=len(PARAM_INFO[name]),hidden_dims=[256,256],dropout=0.05).to(dtype=torch.float64)
    network.load_state_dict(payload['network']); network.eval()
    # Canonical normalization is fitted on the complete frozen seen basin population,
    # not independently recomputed on each evaluation chunk.
    population = all_ids()
    attrs = CatchmentAttributeBuilder(ROOT/'data').build_normalized_attributes(population,device='cpu')
    index = [np.where(population == int(i))[0][0] for i in ids]
    with torch.no_grad(): norm = network(attrs[index].to(next(network.parameters()).dtype))
    mapping = payload.get('job_config',{}).get('mapping',payload.get('job_config',{}).get('parameter_mapping','auto'))
    return norm.to(torch.float64),mapping,str(p)

def convert_linear_to_mapping(name, normalized, mapping):
    result=normalized.clone()
    for j,(p,bounds) in enumerate(PARAM_INFO[name].items()):
        if HydrologyModel._should_use_log_mapping(bounds,mapping,CFG['log_mapping_span_threshold']):
            physical=bounds[0]+normalized[:,j]*(bounds[1]-bounds[0])
            result[:,j]=(physical.log()-np.log(bounds[0]))/(np.log(bounds[1])-np.log(bounds[0]))
    return result
def simulate(name,x,norm,warm=0,mapping='linear',backend='eager'):
    m=build(name,warm,mapping,x.dtype,str(x.device),backend)
    return m({'x_phy':x},(None,norm.to(x.device,dtype=x.dtype)))['streamflow']

def kge(pred,target):
    return 1-KgeLoss()(pred,target,reduction='none')

def training_kge(pred,target):
    """Exact current native runner's IC-compatible streaming metric/reduction.
    Source: benchmark/scripts/run_dpl_benchmark_dmg_native.py; importing that
    entrypoint is forbidden because it mkdirs readonly result paths at import.
    """
    from src.objective import streaming_kge
    scores,invalid=streaming_kge(pred.unsqueeze(-1).unsqueeze(-1),target,eps=0.1)
    scores=scores[:,0,0];valid=(~invalid[:,0,0]) & torch.isfinite(scores)
    return 1-torch.where(valid,scores,torch.zeros_like(scores)).sum()/valid.sum().clamp_min(1),scores
def register_claims():
    """Stream cell-level provenance; identifiers/versions/dates with digits are
    registered as metadata too, avoiding untraced numerical-looking text.
    """
    provenance=DEV/'cache/producers.json'
    producers=json.loads(provenance.read_text()) if provenance.exists() else {}
    columns=['claim_id','value','meaning','source_file','source_row','source_column','source_sha256','generation_script','dmotpy_commit']
    with (MAN/'claims_registry.csv').open('w',newline='') as output:
        writer=csv.DictWriter(output,fieldnames=columns);writer.writeheader()
        for path in sorted(MAN.rglob('*.csv')):
            if path.name=='claims_registry.csv':continue
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            with path.open(newline='') as handle:
                for index,row in enumerate(csv.DictReader(handle)):
                    for col,value in row.items():
                        if not value or (not any(c.isdigit() for c in value) and value.lower() not in {'inf','-inf','nan'}):continue
                        writer.writerow({'claim_id':hashlib.sha256(f'{path.relative_to(MAN)}:{index}:{col}'.encode()).hexdigest()[:20],
                            'value':value,'meaning':f'{path.stem}: row={index}; column={col}',
                            'source_file':str(path.relative_to(DEV)),'source_row':index,'source_column':col,
                            'source_sha256':digest,'generation_script':producers.get(str(path.relative_to(MAN)),'analyses/run.py'),'dmotpy_commit':SHA})
