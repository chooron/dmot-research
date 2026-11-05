"""Explicit open-boundary observer; model code and calculated fluxes remain native."""
import inspect,sys
from scipy.signal import fftconvolve
from .core import *

SIGNS={'ihacres':[-1.],'penman':[1.,-1.,1.],'tcm':[1.,-1.,1.,1.],'topmodel':[1.,-1.]}

def local_return(fn,args,kwargs,keys):
    observed={};code=getattr(fn,'__code__',None);previous=sys.getprofile()
    if code is None:raise NotImplementedError('Eager callable exposes no frame; boundary flux not inferred from residual')
    def profile(frame,event,arg):
        if event=='return' and frame.f_code is code:
            for key in keys:
                if key in frame.f_locals:observed[key]=frame.f_locals[key]
    sys.setprofile(profile)
    try:out=fn(*args,**kwargs)
    finally:sys.setprofile(previous)
    return out,observed

def observe(name,x,z,mapping,warm):
    model=build(name,warm,mapping,x.dtype,str(x.device));B=x.shape[1];ns=model.n_states
    signs=SIGNS.get(name,[1.]*ns);queues=[];route_losses=[];kernels=[];pre_delta=[];before_records=[];post_delta=[];evap=[];incoming=[];outgoing=[];pre_flux=[];native_q=[]
    def routing_hook(module,args,out):
        source,params=args;L=module.max_lag
        impulse=torch.zeros((source.shape[0],L),dtype=source.dtype,device=source.device);impulse[:,0]=1
        weights=module.forward(impulse,params).detach().cpu().numpy();total=weights.sum(-1)
        # Queue is ONLY future output mass of the measured finite kernel.
        # Missing kernel mass is an explicit open sink, never fictitious inventory.
        tail=total[:,None]-np.cumsum(weights,axis=-1);a=source.detach().cpu().numpy()
        stock=np.stack([fftconvolve(v,w)[:a.shape[1]] for v,w in zip(a,tail)],axis=1)
        queues.append(np.diff(np.vstack([np.zeros((1,stock.shape[1])),stock]),axis=0))
        route_losses.append((a*(1-total[:,None])).T)
        kernels.append({'kernel':type(module).__name__,'max_lag':L,'weight_sum_min':float(total.min()),'weight_sum_max':float(total.max()),'max_sum_error':float(np.abs(total-1).max()),'minimum_weight':float(weights.min()),'positive_weight_count_min':int((weights>0).sum(-1).min())})
    for unit in model.modules():
        if hasattr(unit,'get_weights') and hasattr(unit,'max_lag'):unit.register_forward_hook(routing_hook)
    if name in INTERMEDIATE_UH_CONFIG:
        cfg=INTERMEDIATE_UH_CONFIG[name];npre=cfg['n_pre_states'];npost=cfg['n_post_states'];passthru=cfg['n_pre_passthru'];np_pre=len(cfg['pre_params'])
        pre_orig=model.step_pre_fn;post_orig=model.step_post_fn;parallel=list(model._init_states(B,1)[npre:]);counter=[0]
        def pre(*args,**kwargs):
            result=pre_orig(*args,**kwargs);old=args[3+np_pre:3+np_pre+npre]
            pre_delta.append(sum(b-a for a,b in zip(old,result[2:] if name=='gr4j' else result[2+passthru:])).reshape(-1));pre_flux.append(result[:2] if name!='gr4j' else (result[0]*0.9,result[0]*0.1));return result
        def post(*args,**kwargs):
            nonlocal parallel
            begin=2+passthru
            if name=='gr4j':begin=2
            old=args[begin:begin+npost]
            if name=='gr4j':
                result,fields=local_return(post_orig,args,kwargs,['F_total_actual','f_actual_s2','f_actual_q1'])
                f=fields['F_total_actual'];ephys=args[-2]
            else:result=post_orig(*args,**kwargs);f=torch.zeros_like(result[0]);ephys=result[1]
            post_delta.append(sum(b-a for a,b in zip(old,result[2:])).reshape(-1));evap.append(ephys.reshape(-1));incoming.append(f.clamp_min(0).reshape(-1));outgoing.append((-f).clamp_min(0).reshape(-1))
            alternative=list(args);alternative[:2]=pre_flux[counter[0]];alternative[begin:begin+npost]=parallel
            if name=='gr4j':independent,fields2=local_return(post_orig,alternative,kwargs,['F_total_actual']);f2=fields2['F_total_actual'];e2=alternative[-2]
            else:independent=post_orig(*alternative,**kwargs);f2=torch.zeros_like(independent[0]);e2=independent[1]
            d2=sum(b-a for a,b in zip(parallel,independent[2:]));parallel=list(independent[2:])
            before_records.append((d2.reshape(-1),independent[0].reshape(-1),e2.reshape(-1),f2.reshape(-1)));counter[0]+=1;return result
        model.step_pre_fn=pre;model.step_post_fn=post
        with torch.no_grad():q=model({'x_phy':x},(None,z.unsqueeze(-1)))['streamflow']
        if q.ndim==3:q=q.squeeze(-1)
        dp=torch.stack(pre_delta).numpy();ds=torch.stack(post_delta).numpy();ea=torch.stack(evap).numpy();fin=torch.stack(incoming).numpy();fout=torch.stack(outgoing).numpy()
        db,qb,eb,fb=(torch.stack([r[i] for r in before_records]).numpy() for i in range(4))
        before_ea=eb;before_in=np.maximum(fb,0);before_out=np.maximum(-fb,0)
        before=dp+db-(x[:,:,0].numpy()+fb-eb-qb)
        after=(dp+ds)[warm:]-(x[warm:,:,0].numpy()+fin[warm:]-fout[warm:]-ea[warm:]-q.numpy())
    else:
        original=model.step_fn;np_=len(PARAM_INFO[name]);diag_supported='return_diagnostics' in inspect.signature(original).parameters
        def step(*args,**kwargs):
            if name=='modhydrolog':result,fields=local_return(original,args,kwargs,['flux_SEEP']);seep=fields['flux_SEEP']
            else:result=original(*args,**dict(kwargs,return_diagnostics=True)) if diag_supported else original(*args,**kwargs);seep=torch.zeros_like(result[0])
            diagnostic=result[-1] if diag_supported and isinstance(result[-1],dict) else {}
            if diagnostic:result=result[:-1]
            old=args[3+np_:3+np_+ns];new=result[2:2+ns];delta=sum(s*(b-a) for s,a,b in zip(signs,old,new))
            loss=diagnostic.get('external_losses',torch.zeros_like(result[0]))+seep;ephys=result[1]-seep
            before_records.append((delta-(args[0]-ephys-result[0]-loss)).reshape(-1));native_q.append(result[0].reshape(-1));evap.append(ephys.reshape(-1));incoming.append(torch.zeros_like(result[0]).reshape(-1));outgoing.append(loss.reshape(-1));return result
        model.step_fn=step
        with torch.no_grad():q=model({'x_phy':x},(None,z.unsqueeze(-1)))['streamflow']
        if q.ndim==3:q=q.squeeze(-1)
        ea=torch.stack(evap).numpy();fin=torch.stack(incoming).numpy();fout=torch.stack(outgoing).numpy();before=torch.stack(before_records).numpy()
        before_ea=ea;before_in=fin;before_out=fout
        after=before[warm:]+q.numpy()-torch.stack(native_q)[warm:].numpy()
    route_loss=sum(route_losses) if route_losses else np.zeros_like(ea)
    queue_delta=sum(queues) if queues else np.zeros_like(ea)
    after=after+queue_delta[warm:]+route_loss[warm:]
    return {'before':before[warm:],'after':after,'before_Ephysical':before_ea[warm:],'before_Fin':before_in[warm:],'before_Fout':before_out[warm:],'Ephysical':ea[warm:],'Fin':fin[warm:],'Fout':fout[warm:]+route_loss[warm:],'routing_loss':route_loss[warm:],'kernels':kernels,'boundary_contract':'GR4J actual native F; modhydrolog native seep separate from Ea proxy; diagnostic external_losses; finite impulse-response queue and routing missing-mass sink'}
