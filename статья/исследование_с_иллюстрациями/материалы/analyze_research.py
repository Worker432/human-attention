import sys,json,csv,math,hashlib,collections
from pathlib import Path
import numpy as np
sys.path.insert(0,'/Users/zaxharmerkurev/Documents/dimplom/project/gaze-mvp')
from gaze_analysis import read_recordings,extract_fixations
base=Path('/Users/zaxharmerkurev/Documents/dimplom');out=base/'статья/исследование_с_иллюстрациями/материалы'
protocol,trials=read_recordings(base/'project/gaze-mvp/collected/p001/dataset.sqlite3');complete=[(t,s) for t,s in trials if t['status']=='completed']
records=[];sensitivity=[]
for t,s in complete:
 g=json.loads(t['geometry']);w,h=g['image_width'],g['image_height'];f=extract_fixations(s,w,h)
 records.append(dict(trial=t['id'],image=Path(t['path']).name,duration=t['duration'],samples=len(s),valid=sum(bool(x['valid']) for x in s),usable=sum(bool(x['valid'] and x['inside_image']) for x in s),fixations=len(f),fixation_seconds=sum(x['duration'] for x in f),median_fixation=np.median([x['duration'] for x in f]) if f else None))
for disp in [.02,.04,.06]:
 for dur in [.08,.10,.15]:
  fs=[]
  for t,s in complete:
   g=json.loads(t['geometry']);fs+=extract_fixations(s,g['image_width'],g['image_height'],dispersion_fraction=disp,min_duration=dur)
  sensitivity.append(dict(dispersion=disp,min_duration=dur,n=len(fs),seconds=sum(x['duration'] for x in fs)))
with open(out/'recording_quality.csv','w') as f:
 wr=csv.DictWriter(f,fieldnames=list(records[0]));wr.writeheader();wr.writerows(records)
with open(out/'fixation_sensitivity.csv','w') as f:
 wr=csv.DictWriter(f,fieldnames=list(sensitivity[0]));wr.writeheader();wr.writerows(sensitivity)
summary={'n_completed':len(complete),'n_interrupted':len(trials)-len(complete),'samples':sum(r['samples'] for r in records),'valid':sum(r['valid'] for r in records),'usable':sum(r['usable'] for r in records),'duration':sum(r['duration'] for r in records),'fixations':sum(r['fixations'] for r in records),'fixation_seconds':sum(r['fixation_seconds'] for r in records),'median_duration':float(np.median([r['duration'] for r in records])),'duration_q':np.quantile([r['duration'] for r in records],[0,.25,.5,.75,1]).tolist(),'usable_q':np.quantile([r['usable']/r['samples'] for r in records],[0,.25,.5,.75,1]).tolist(),'sensitivity':sensitivity}
# OSIE plots: paired per-image deltas and mean intervals
rows=list(csv.DictReader(open(base/'docs/attention_results_4ab63426da20/per_image_scores.csv')))
ims=list(dict.fromkeys(r['image'] for r in rows));delta=[]
for im in ims:
 a=next(r for r in rows if r['image']==im and r['model']=='DeepGaze IIE');b=next(r for r in rows if r['image']==im and r['model']=='DeepGaze III');delta.append(float(b['ll_gain_uniform_bits'])-float(a['ll_gain_uniform_bits']))
summary['positive_images']=sum(x>0 for x in delta)
# Cached generation cohort strictly matching image, steps, seed and resolution; no pooling different protocols.
cache=[]
for p in (base/'project/gaze-mvp/collected/predictions').glob('*.json'):
 try:r=json.loads(p.read_text())
 except:continue
 if r.get('data_source')=='model_prediction':cache.append(r)
cohort=[];convergence=[]
for name in ['059738.jpg','041647.jpg']:
 rs=[r for r in cache if Path(r.get('image_path','')).name==name and r['parameters']['steps']==8 and r['parameters']['max_side']==1024]
 groups=collections.defaultdict(list)
 for r in rs:groups[(r['image_sha256'],r['model_size'][0],r['model_size'][1])].append(r)
 group=next((g for g in groups.values() if set(range(42,72))<=set(r['parameters']['seed'] for r in g)),None)
 if not group:continue
 ordered=[next(r for r in group if r['parameters']['seed']==k) for k in range(42,72)]
 boxes=ordered[0]['objects']
 def vector(r):
  c=collections.Counter()
  for p in r['fixations']:
   hits=[b['object_id'] for b in boxes if b['xyxy'][0]<=p['x']<=b['xyxy'][2] and b['xyxy'][1]<=p['y']<=b['xyxy'][3]]
   c[str(hits[0]) if len(hits)==1 else 'ambiguous' if hits else 'outside']+=1
  return np.array([c[str(b['object_id'])] for b in boxes]+[c['ambiguous'],c['outside']],float)
 counts=np.stack([vector(r) for r in ordered]);full=counts.sum(0)/240
 for n in [1,5,10,20,30]:convergence.append({'image':name,'variants':n,'max_abs_delta_pp':float(np.max(np.abs(counts[:n].sum(0)/(8*n)-full))*100)})
 cohort.append({'image':name,'sha256':ordered[0]['image_sha256'],'variants':30,'points':240,'boxes':boxes,'counts':counts.sum(0).astype(int).tolist(),'shares':full.tolist(),'per_variant':counts.astype(int).tolist()})
summary['cohort']=cohort;summary['convergence']=convergence
(out/'research_analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False)[:12000])
