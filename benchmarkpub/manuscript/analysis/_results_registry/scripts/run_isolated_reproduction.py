#!/usr/bin/env python3
"""Run the existing analysis script inventory in a disposable, write-guarded copy."""
from __future__ import annotations
import ast
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parents[2]
REPO = ANALYSIS.parents[3]
OUT = ANALYSIS / '_results_registry/tables'
PYTHON = REPO / '.venv/bin/python'
GUARD = r'''
import errno, json, os, pathlib, sys
sys.dont_write_bytecode = True
sandbox = pathlib.Path(os.environ['REGISTRY_REPRO_SANDBOX']).resolve()
old_mkdir = os.mkdir

def resolved(path, dir_fd=None):
    if isinstance(path, int): return None
    p = pathlib.Path(os.fsdecode(path))
    if not p.is_absolute() and dir_fd is not None and dir_fd >= 0:
        p = pathlib.Path(os.readlink('/proc/self/fd/'+str(dir_fd))) / p
    return p.resolve()
def writable(path):
    p = resolved(path)
    return p is None or p == sandbox or sandbox in p.parents or str(p) == '/dev/null'

def mkdir(path, mode=0o777, *, dir_fd=None):
    p = resolved(path, dir_fd)
    if not writable(p) and p.is_dir():
        raise FileExistsError(errno.EEXIST, 'Read-only directory already exists', str(p))
    return old_mkdir(path, mode, dir_fd=dir_fd)
os.mkdir = mkdir
old_open = os.open
def open_at(path, flags, mode=0o777, *, dir_fd=None):
    if dir_fd is not None and not isinstance(path, int):
        path = resolved(path, dir_fd)
        dir_fd = None
    return old_open(path, flags, mode, dir_fd=dir_fd)
os.open = open_at

def audit(event, args):
    paths = []
    if event == 'open':
        path, mode, flags = args
        if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND): paths = [path]
    elif event in {'os.mkdir','os.chmod'}:
        paths = [resolved(args[0], args[2])]
    elif event in {'os.remove','os.rmdir'}:
        paths = [resolved(args[0], args[1])]
    elif event == 'os.utime':
        paths = [resolved(args[0], args[-1])]
    elif event == 'os.truncate':
        paths = [args[0]]
    elif event in {'os.rename','os.replace','os.link'}:
        paths = [resolved(args[0], args[2]), resolved(args[1], args[3])]
    elif event == 'os.symlink':
        paths = [resolved(args[1], args[2])]
    for path in paths:
        if not writable(path):
            detail = json.dumps({'event':event,'path':str(resolved(path))}, sort_keys=True)
            print('ISOLATION_VIOLATION_S2='+detail, file=sys.stderr, flush=True)
            raise PermissionError('BLOCKED(S2): attempted write outside disposable copy: '+detail)
sys.addaudithook(audit)
'''

def rows(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        rd = csv.DictReader(f); return list(rd.fieldnames or []), list(rd)

def write(path, fields, records):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fields, lineterminator='\n'); w.writeheader(); w.writerows(records)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def inventory():
    tree = ast.parse((ANALYSIS / 'verify_reproducibility.py').read_text())
    declaration = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SCRIPTS' for t in n.targets))
    def expression(n):
        if isinstance(n,ast.Name) and n.id=='BASE': return ''
        if isinstance(n,ast.Constant) and isinstance(n.value,str): return n.value
        if isinstance(n,ast.BinOp) and isinstance(n.op,ast.Div): return '/'.join(x for x in (expression(n.left),expression(n.right)) if x)
        raise ValueError('Unrecognized script inventory expression')
    scripts = [expression(n) for n in declaration.elts]
    # Maintain section dependencies; order each section's scripts by its numeric suffix.
    groups = {}
    for name in scripts: groups.setdefault(name.split('/')[0], []).append(name)
    import re
    return [name for names in groups.values() for name in sorted(names,key=lambda s:(int(re.search(r'_(\d+)_',Path(s).name).group(1)) if re.search(r'_(\d+)_',Path(s).name) else names.index(s), names.index(s)))]

def numeric(value):
    try:
        x=Decimal(value)
        return x if x.is_finite() else None
    except (InvalidOperation,ValueError,TypeError): return None

def compare(original, replica):
    fields, a = rows(original); other_fields, b = rows(replica)
    identifiers = {'basin_id','gage_id','gauge_id','gauge_id_str','condition','target','target_pct','type','type_a','type_b','model','fold','seed','k','rule','tier','strategy','comparison','component','test','instability_metric','metric','estimand','solution_id','solution_index','draw_index','attribute','quantity_name','quantity','index','family','block_years'}
    key_fields = [f for f in fields if f in identifiers]
    # Categorical schema labels disambiguate summary records without selecting a numeric result as a key.
    key_fields += [f for f in fields if f not in key_fields and a and any(numeric(r.get(f,'')) is None and r.get(f,'') for r in a) and f not in {'notes','note','definition','direction_definition','sampling_pool','file_path','script','source_file','interpretation','value','sha256','hash','source_sha256','mtime','mtime_ns','elapsed_seconds','byte_count'}]
    def key(r):
        values = []
        for field in key_fields:
            value = r.get(field,'') or ''
            if field in {'basin_id','gage_id','gauge_id','gauge_id_str'} and value.isdigit(): value=value.zfill(8)
            elif numeric(value) is not None: value=str(numeric(value).normalize())
            values.append(value)
        return tuple(values)
    ka,kb = Counter(key(r) for r in a),Counter(key(r) for r in b)
    unique = all(n==1 for n in ka.values()) and all(n==1 for n in kb.values())
    pairs = []
    if unique:
        ma,mb={key(r):r for r in a},{key(r):r for r in b}
        pairs=[(k,ma[k],mb[k]) for k in ma.keys()&mb.keys()]
    elif ka==kb and len(a)==len(b):
        # Duplicate keys are explicitly reported; preserve within-key order for numeric comparison.
        ga,gb={},{}
        for r in a:ga.setdefault(key(r),[]).append(r)
        for r in b:gb.setdefault(key(r),[]).append(r)
        pairs=[(k,x,y) for k in ga for x,y in zip(ga[k],gb[k])]
    maximum=Decimal(0); differences=[]; changed_cells=0; missing_cells=0
    for k,x,y in pairs:
        for f in set(fields)&set(other_fields)-set(key_fields):
            u,v=numeric(x.get(f,'')),numeric(y.get(f,''))
            if (u is None) != (v is None): missing_cells += 1
            if u is not None and v is not None:
                delta=abs(u-v)
                if delta:
                    changed_cells+=1; maximum=max(maximum,delta)
                    differences.append(dict(table=str(original.relative_to(ANALYSIS)),key=json.dumps(dict(zip(key_fields,k)),sort_keys=True),column=f,original_value=x[f],replica_value=y[f],absolute_difference=str(delta)))
    key_difference = len(set(ka)^set(kb))
    mismatch = len(a)!=len(b) or ka!=kb or fields!=other_fields or maximum!=0 or missing_cells!=0
    return dict(table=str(original.relative_to(ANALYSIS)),original_rows=len(a),replica_rows=len(b),key_fields='|'.join(key_fields),unique_keys=unique,
                key_set_difference=key_difference,key_multiplicity_equal=ka==kb,column_schema_equal=fields==other_fields,numeric_changed_cells=changed_cells,numeric_missingness_changed_cells=missing_cells,
                maximum_absolute_difference=str(maximum),consistent=not mismatch),differences

def main():
    scripts = inventory()
    sys.path.insert(0,str(Path(__file__).parent))
    import build_results_artifacts as registry_builder
    from close_registry_m1_m2 import protected_snapshot
    upstream_before = protected_snapshot(registry_builder)
    original_tables = {p.relative_to(ANALYSIS): (p.stat().st_mtime_ns,p.stat().st_size,digest(p)) for p in ANALYSIS.rglob('*.csv')
                       if 'tables' in p.parts and '_results_registry' not in p.parts}
    script_reports=[]; table_reports=[]; differences=[]; stop=''; copy_deleted=False
    workspace=Path(tempfile.mkdtemp(prefix='registry-reproduction-'))
    try:
        copy_repo=workspace/'repo'; copy_analysis=copy_repo/'project/benchmarkpub/manuscript/analysis'
        shutil.copytree(ANALYSIS,copy_analysis,symlinks=True)
        # Preserve dependency layout via read-only symlinks; write guard resolves their real targets.
        for p in REPO.iterdir():
            target=copy_repo/p.name
            if p.name not in {'project','.git','.pi'} and not target.exists():
                target.symlink_to(p,target_is_directory=p.is_dir())
        for p in (REPO/'project').iterdir():
            target=copy_repo/'project'/p.name
            if p.name!='benchmarkpub' and not target.exists():
                target.symlink_to(p,target_is_directory=p.is_dir())
        for source,destination,exclude in (
            (REPO/'project/benchmarkpub',copy_repo/'project/benchmarkpub',{'manuscript'}),
            (REPO/'project/benchmarkpub/manuscript',copy_repo/'project/benchmarkpub/manuscript',{'analysis'})):
            for p in source.iterdir():
                target=destination/p.name
                if p.name not in exclude and not target.exists():
                    target.symlink_to(p,target_is_directory=p.is_dir())
        for p in copy_analysis.rglob('*.py'):
            text=p.read_text()
            patched=text.replace(str(REPO),str(copy_repo))
            if patched!=text:p.write_text(patched)
        bridge=workspace/'bridge';bridge.mkdir();(bridge/'sitecustomize.py').write_text(GUARD)
        tmp=workspace/'tmp';tmp.mkdir()
        env=dict(os.environ,PYTHONPATH=str(bridge),PYTHONDONTWRITEBYTECODE='1',REGISTRY_REPRO_SANDBOX=str(workspace),
                 TMPDIR=str(tmp),MPLCONFIGDIR=str(workspace/'mpl'),XDG_CACHE_HOME=str(workspace/'cache'),
                 OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        baseline={p.relative_to(copy_analysis):p.stat().st_mtime_ns for p in copy_analysis.rglob('*.csv') if 'tables' in p.parts and '_results_registry' not in p.parts}
        for index,name in enumerate(scripts,1):
            script=copy_analysis/name; started=time.monotonic()
            try:
                process=subprocess.run([str(PYTHON),'-B',str(script)],cwd=script.parent,env=env,capture_output=True,text=True,timeout=3600)
                code,stderr,stdout=process.returncode,process.stderr,process.stdout
            except subprocess.TimeoutExpired as exc:
                code,stderr,stdout='timeout',str(exc),' '
            violation='ISOLATION_VIOLATION_S2=' in stderr
            script_reports.append(dict(script=name,exit_status=code,elapsed_seconds=time.monotonic()-started,write_guard_violation=violation,stderr=stderr[-12000:],stdout=stdout[-2000:]))
            print(json.dumps(dict(script=name,index=index,exit_status=code),sort_keys=True),flush=True)
            if violation:
                stop='BLOCKED(S2)';break
        for p in sorted(copy_analysis.rglob('*.csv')):
            if 'tables' not in p.parts or '_results_registry' in p.parts:continue
            relative=p.relative_to(copy_analysis)
            recomputed = p.stat().st_mtime_ns != baseline.get(relative)
            original=ANALYSIS/relative
            if not original.exists():
                table_reports.append(dict(table=str(relative),recomputed=recomputed,original_rows='',replica_rows=len(rows(p)[1]),key_fields='',unique_keys='',key_set_difference='',key_multiplicity_equal='',column_schema_equal='',numeric_changed_cells='',numeric_missingness_changed_cells='',maximum_absolute_difference='',consistent=False));continue
            report,details=compare(original,p);report['recomputed']=recomputed;table_reports.append(report);differences.extend(details)
        upstream_after = protected_snapshot(registry_builder)
        upstream_checks = [dict(file_path=name,unchanged=upstream_before.get(name)==upstream_after.get(name),
                                before_sha256=upstream_before.get(name,(None,None,''))[2],after_sha256=upstream_after.get(name,(None,None,''))[2])
                           for name in sorted(set(upstream_before)|set(upstream_after))]
        write(OUT/'T3_isolated_upstream_integrity.csv',['file_path','unchanged','before_sha256','after_sha256'],upstream_checks)
        changed_inputs = [r['file_path'] for r in upstream_checks if not r['unchanged']]
        if changed_inputs: stop='BLOCKED(S2)'
        changes=[str(p) for p,(mtime,size,h) in original_tables.items() if not (ANALYSIS/p).exists() or ((ANALYSIS/p).stat().st_mtime_ns,(ANALYSIS/p).stat().st_size,digest(ANALYSIS/p))!=(mtime,size,h)]
        if changes:stop='BLOCKED(S1)'
        differences.sort(key=lambda r:Decimal(r['absolute_difference']),reverse=True)
        top_differences=[]; per_table=Counter()
        for item in differences:
            if per_table[item['table']] < 5:
                top_differences.append(item);per_table[item['table']]+=1
        write(OUT/'T3_isolated_scripts.csv',['script','exit_status','elapsed_seconds','write_guard_violation','stderr','stdout'],script_reports)
        write(OUT/'T3_isolated_table_comparison.csv',['table','recomputed','original_rows','replica_rows','key_fields','unique_keys','key_set_difference','key_multiplicity_equal','column_schema_equal','numeric_changed_cells','numeric_missingness_changed_cells','maximum_absolute_difference','consistent'],table_reports)
        write(OUT/'T3_isolated_top_differences.csv',['table','key','column','original_value','replica_value','absolute_difference'],top_differences)
        summary=dict(status=stop or 'DONE',script_total=len(scripts),script_attempted=len(script_reports),failed_scripts=sum(r['exit_status']!=0 for r in script_reports),
                     failed_script_names=[r['script'] for r in script_reports if r['exit_status']!=0],compared_tables=len(table_reports),recomputed_tables=sum(r['recomputed'] for r in table_reports),protected_inputs=len(upstream_checks),changed_inputs=changed_inputs,inconsistent_tables=sum(not r['consistent'] for r in table_reports),
                     maximum_absolute_difference=max((Decimal(r['maximum_absolute_difference']) for r in table_reports if r['maximum_absolute_difference']),default=Decimal(0)).__str__(),
                     original_frozen_tables=len(original_tables),original_changed_tables=changes,copy_deleted=False)
    finally:
        shutil.rmtree(workspace)
        copy_deleted=not workspace.exists()
    summary['copy_deleted']=copy_deleted
    (OUT/'T3_isolated_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print('ISOLATED_SUMMARY='+json.dumps(summary,sort_keys=True),flush=True)
    if stop:sys.exit(2)

if __name__=='__main__':main()
