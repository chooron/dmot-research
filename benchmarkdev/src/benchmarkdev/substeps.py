"""Non-invasive storage subcycling; the original model retains DAILY UH routing.
No step/flux expression is copied. Time-unit conversions are recorded explicitly.
N=1 is the exact original call path; other N values only wrap imported step calls.
"""
import inspect,re
import torch
from benchmarkdev.core import PARAM_INFO,STATE_INFO,STFN_INFO


# Dimensional review against the imported step/flux implementations. Exact-day
# values below are not integration intervals: routing delays belong to the
# unchanged daily UH, calendar coordinates remain in day-of-year space, and
# Susannah1 `a` is a storage-depth scale despite its stale [d] description.
DAY_VALUE_ROLES={
    ('flexb','nlagf'):'routing delay; intermediate UH is applied after daily pre-step integration',
    ('flexb','nlags'):'routing delay; intermediate UH is applied after daily pre-step integration',
    ('flexi','nlagf'):'routing delay; intermediate UH is applied after daily pre-step integration',
    ('flexi','nlags'):'routing delay; intermediate UH is applied after daily pre-step integration',
    ('flexis','nlagf'):'routing delay; intermediate UH is applied after daily pre-step integration',
    ('flexis','nlags'):'routing delay; intermediate UH is applied after daily pre-step integration',
    ('hbv96','maxbas'):'endpoint UH delay; routing remains daily and core argument is unused',
    ('hillslope','th'):'endpoint UH delay; routing remains daily',
    ('ihacres','tau_q'):'endpoint UH delay; routing remains daily',
    ('ihacres','tau_s'):'endpoint UH delay; routing remains daily',
    ('mopex4','is_time'):'calendar day-of-year phase in interception cosine; preserve across substeps',
    ('mopex5','is_time'):'calendar day-of-year phase in interception cosine; preserve across substeps',
    ('newzealand2','d_delay'):'endpoint UH delay; routing remains daily and core argument is unused',
    ('plateau','tp'):'endpoint UH delay; routing remains daily',
    ('smar','nk_delay'):'endpoint UH delay; routing remains daily',
    ('susannah1','a'):'storage-depth scale [mm] in interflow_7/baseflow_2; descriptor [d] is inconsistent with formulas',
    ('vic','ishift'):'calendar day-of-year phase in seasonal interception; preserve across substeps',
}

# Non-parameter arguments whose per-call flux/rate scale must follow dt.
# TCM mean_P is a daily abstraction-rate input; GSFB tau is a flux-scale
# smoothing width. Both are applied inside the substep function and use 1/N.
STEP_ARGUMENT_EXPONENTS={('gsfb','tau'):-1,('tcm','mean_P'):-1}

def parameter_time_units(name):
    module=inspect.getmodule(STFN_INFO[name]);descriptions={}
    for key,value in vars(module).items():
        if key.endswith('_PARAMS_DESC') and isinstance(value,dict):descriptions.update(value)
    factors={};audit=[]
    for parameter in PARAM_INFO[name]:
        text=descriptions.get(parameter,'');unit=' '.join(re.findall(r'\[([^]]+)\]',text)).lower().strip()
        role=DAY_VALUE_ROLES.get((name,parameter))
        if role:
            exponent=0
            category=('depth [mm]' if name=='susannah1' and parameter=='a' else
                      'calendar phase' if 'calendar' in role else 'time scale / daily routing delay')
        elif any(s in unit for s in ['d-1','d^-1','d⁻¹','1/d','day-1','day^-1','mm d'] ) or re.search(r'/(?:d|day)(?:$|/|\b)',unit):
            exponent=-1;category='rate per day'
        elif unit in {'d','day','days'}:
            raise ValueError(f'Unreviewed time-valued parameter {name}.{parameter}: {text!r}')
        elif any(s in unit for s in ('mm-1','mm^-1','mm−1')):
            exponent=0;category='inverse depth'
        elif 'mm' in unit:
            exponent=0;category='depth'
        elif any(s in text.lower() for s in ['fraction','proportion','fractional']):
            exponent=0;category='per-step proportion / dimensionless fraction'
        elif any(s in unit for s in ['°c','oc','celsius']):
            exponent=0;category='temperature threshold'
        else:
            exponent=0;category='dimensionless / shape'
        factors[parameter]=exponent
        audit.append({'model':name,'parameter':parameter,'description':text,'unit_category':category,
                      'time_exponent':exponent,'effective_scale':'N^-1' if exponent==-1 else 'unchanged',
                      'reviewed_role':role or 'per imported formula and dimensional unit',
                      'source_file':inspect.getsourcefile(module),
                      'contract':'N=1 unchanged; subdaily rate per-day terms divided by N; daily routing and calendar coordinates unchanged'})
    return factors,audit
def add_flux(a,b):
    if isinstance(a,(tuple,list)):return type(a)(add_flux(x,y) for x,y in zip(a,b))
    if isinstance(a,dict):return {k:add_flux(a[k],b[k]) for k in a}
    return a+b

class StorageSubcycling:
    def __init__(self,model,N):
        self.model=model;self.N=int(N);self.originals={};self.ea=[];self.output_records=[]
        self.factors,self.audit=parameter_time_units(model.model_name)
        n=len(PARAM_INFO[model.model_name]);states=STATE_INFO[model.model_name]
        if hasattr(model,'step_pre_fn'):
            # Signature names, rather than a copied GR4J/Flex equation, identify stores.
            self.install('step_pre_fn','pre')
            self.install('step_post_fn','post')
        else:self.install('step_fn','core',list(range(3+n,3+n+states)))

    def install(self,attribute,stage,state_positions=None):
        original=getattr(self.model,attribute);signature=inspect.signature(original)
        names=list(signature.parameters)
        if state_positions is None:state_positions=[i for i,n in enumerate(names) if re.fullmatch(r'[Ss]\d+(?:_\w+)?',n)]
        if not state_positions:raise NotImplementedError(f'Unidentified imported {stage} state arguments: {names}')
        forcing=[i for i,n in enumerate(names) if n.lower() in {'p','pet','precip','precipitation','flux_q9','flux_q1','flux_qf','flux_qs','e_physical','ea_passthrough'}]
        if stage=='core':forcing=[0,2]
        elif stage=='pre':forcing=[0,2]
        elif stage=='post':forcing=[i for i,n in enumerate(names) if n.startswith('flux_') or n.lower() in {'e_physical','ea_passthrough'}]
        output_start=2
        if stage=='pre':
            from dmotpy.models.intermediate_uh_model import INTERMEDIATE_UH_CONFIG
            if self.model.model_name=='gr4j':output_start=2
            else:output_start=2+INTERMEDIATE_UH_CONFIG[self.model.model_name].get('n_pre_passthru',0)
        state_indices=list(range(output_start,output_start+len(state_positions)))
        self.originals[attribute]=original
        def wrapped(*args,**kwargs):
            if self.N==1:
                out=original(*args,**kwargs)
                if stage=='core' or stage=='pre':self.ea.append((out[1] if stage=='core' or self.model.model_name=='gr4j' else sum(out[2:output_start])).detach())
                return out
            current=list(args);accumulated=None
            for sub in range(self.N):
                step=list(current)
                for i in forcing:
                    if i<len(step):step[i]=step[i]/self.N
                for i,name in enumerate(names):
                    exponent=self.factors.get(name,STEP_ARGUMENT_EXPONENTS.get((self.model.model_name,name),0))
                    if exponent and i<len(step):step[i]=step[i]*self.N**exponent
                step_kwargs=dict(kwargs)
                for name,value in list(step_kwargs.items()):
                    exponent=self.factors.get(name,STEP_ARGUMENT_EXPONENTS.get((self.model.model_name,name),0))
                    if exponent:step_kwargs[name]=value*self.N**exponent
                for i,name in enumerate(names):
                    exponent=STEP_ARGUMENT_EXPONENTS.get((self.model.model_name,name),0)
                    if exponent and i>=len(step) and name not in step_kwargs:
                        default=signature.parameters[name].default
                        if default is not inspect.Parameter.empty:step_kwargs[name]=default*self.N**exponent
                out=original(*step,**step_kwargs)
                # State values are carried verbatim; Q/Ea and optional routing-flux
                # tuples are integrated. The model's UH sees one aggregated day.
                if accumulated is None:accumulated=list(out)
                else:
                    for i in range(len(out)):
                        if i not in state_indices:accumulated[i]=add_flux(accumulated[i],out[i])
                for pos,index in zip(state_positions,state_indices):current[pos]=out[index];accumulated[index]=out[index]
            if stage=='core' or stage=='pre':self.ea.append((accumulated[1] if stage=='core' or self.model.model_name=='gr4j' else sum(accumulated[2:output_start])).detach())
            return tuple(accumulated)
        setattr(self.model,attribute,wrapped)

    def restore(self):
        for name,value in self.originals.items():setattr(self.model,name,value)
