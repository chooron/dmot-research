"""Non-invasive conservation observers. All step/routing functions are imported.
Routing queue content is measured from the public impulse response (future weight
mass times past input), not inferred as cumulative input-minus-output.
"""
import inspect
import numpy as np
import torch
from .core import *

def observe(name,x,raw,mapping,warm):
    model=build(name,warm,mapping,x.dtype,str(x.device));batch=x.shape[1]
    queues=[];kernels=[];physical=[];unrouted=[];evaps=[];pre_delta=[];pre_flows=[]
    ns=model.n_states;np_=len(PARAM_INFO[name])
    signs={'ihacres':[-1.],'penman':[1.,-1.,1.],'tcm':[1.,-1.,1.,1.],'topmodel':[1.,-1.]}.get(name,[1.]*ns)
    def hook(module,args,out):
        inflow,params=args
        length=module.max_lag
        impulse=torch.zeros((inflow.shape[0],length),dtype=inflow.dtype,device=inflow.device);impulse[:,0]=1
        # Calling .forward directly avoids recursively invoking this observer.
        response=module.forward(impulse,params)
        weights=response.detach().cpu().numpy()
        total=weights.sum(-1)
        kernels.append({'kernel_class':type(module).__name__,'min_weight_sum':float(total.min()),'max_weight_sum':float(total.max()),'max_sum_error':float(np.max(np.abs(total-1)))})
        tail=1-np.cumsum(weights,axis=-1)
        source=inflow.detach().cpu().numpy()
        pending=np.stack([np.convolve(a,b)[:source.shape[1]] for a,b in zip(source,tail)],axis=1)
        change=np.diff(np.concatenate([np.zeros((1,pending.shape[1])),pending],axis=0),axis=0)
        queues.append(change)
    for module in model.modules():
        if hasattr(module,'get_weights') and hasattr(module,'max_lag'):
            module.register_forward_hook(hook)
    if name in INTERMEDIATE_UH_CONFIG:
        if name=='gr4j':raise NotImplementedError('GR4J exchange losses and specialized UH observer not mapped')
        cfg=INTERMEDIATE_UH_CONFIG[name]
        pre_orig=model.step_pre_fn;post_orig=model.step_post_fn
        pre_np=len(cfg['pre_params']);pre_ns=cfg['n_pre_states'];post_ns=cfg['n_post_states'];passthru=cfg['n_pre_passthru']
        parallel_states=list(model._init_states(batch,1)[pre_ns:])
        def pre(*args,**kw):
            out=pre_orig(*args,**kw)
            old=args[3+pre_np:3+pre_np+pre_ns];new=out[2+passthru:]
            pre_delta.append(sum(b-a for a,b in zip(old,new)).reshape(-1))
            pre_flows.append(out[:2]);return out
        counter=[0]
        def post(*args,**kw):
            nonlocal parallel_states
            out=post_orig(*args,**kw)
            begin=2+passthru
            old=args[begin:begin+post_ns];new=out[2:]
            physical.append(sum(b-a for a,b in zip(old,new)).reshape(-1))
            alternative=list(args);alternative[:2]=pre_flows[counter[0]]
            alternative[begin:begin+post_ns]=parallel_states
            independent=post_orig(*alternative,**kw)
            unrouted.append((sum(b-a for a,b in zip(parallel_states,independent[2:])),independent[0],independent[1]))
            parallel_states=list(independent[2:])
            evaps.append(out[1].reshape(-1));counter[0]+=1
            return out
        model.step_pre_fn=pre;model.step_post_fn=post
        with torch.no_grad():q=model({'x_phy':x},(None,raw))['streamflow']
        dp=torch.stack(pre_delta).numpy()
        ds=torch.stack(physical).numpy()
        ea=torch.stack(evaps).numpy()
        raw_delta=torch.stack([v[0].reshape(-1) for v in unrouted]).numpy()
        raw_q=torch.stack([v[1].reshape(-1) for v in unrouted]).numpy()
        raw_ea=torch.stack([v[2].reshape(-1) for v in unrouted]).numpy()
        precip=x[:,:,0].numpy()
        before=dp[warm:]+raw_delta[warm:]-(precip[warm:]-raw_ea[warm:]-raw_q[warm:])
        queue_delta=sum(queues) if queues else np.zeros_like(precip)
        after=dp[warm:]+ds[warm:]+queue_delta[warm:]-(precip[warm:]-ea[warm:]-q.numpy())
        return before,after,kernels
    orig=model.step_fn;counter=[0];raw_q=[]
    def step(*args,**kw):
        old=args[3+np_:3+np_+ns]
        use_diag='return_diagnostics' in inspect.signature(orig).parameters
        out=orig(*args,**dict(kw,return_diagnostics=True)) if use_diag else orig(*args,**kw)
        diag=out[-1] if use_diag and isinstance(out[-1],dict) else {}
        if use_diag:out=out[:-1]
        new=out[2:2+ns]
        delta=sum(s*(b-a) for s,a,b in zip(signs,old,new))
        extra=diag.get('external_losses',torch.zeros_like(out[0]))
        physical.append((delta-(args[0]-out[1]-out[0]-extra)).reshape(-1))
        raw_q.append(out[0].reshape(-1));counter[0]+=1
        return out
    model.step_fn=step
    with torch.no_grad():q=model({'x_phy':x},(None,raw))['streamflow']
    before=torch.stack(physical)[warm:].numpy()
    preq=torch.stack(raw_q)[warm:].numpy()
    if queues:
        after=before+sum(queues)[warm:]+q.numpy()-preq
    else:after=before.copy()
    return before,after,kernels
