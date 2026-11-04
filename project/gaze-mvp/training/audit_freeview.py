"""Read-only audit of the official trainval JSON; never invents a test split."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def audit(path):
    raw=Path(path).read_bytes();rows=json.loads(raw)
    if not isinstance(rows,list) or not rows: raise ValueError('expected nonempty list of trials')
    splits=Counter();images=defaultdict(set);subjects=set();seen=set();duplicates=0
    invalid=Counter();points=0;initial=[]
    for index,row in enumerate(rows):
        for field in ('name','subject','X','Y','T','length','split'):
            if field not in row: raise ValueError(f'trial {index}: missing {field}')
        split=row['split']
        if split not in ('train','valid','test'):raise ValueError(f'unknown official split: {split}')
        splits[split]+=1;images[split].add(row['name']);subjects.add(row['subject'])
        key=json.dumps(row,sort_keys=True)
        if key in seen:duplicates+=1
        seen.add(key)
        if len(row['X'])!=len(row['Y']) or len(row['X'])!=len(row['T']) or len(row['X'])!=row['length']:
            invalid['length_mismatch']+=1;continue
        for x,y,t in zip(row['X'],row['Y'],row['T']):
            points+=1
            if not all(isinstance(a,(int,float)) and math.isfinite(a) for a in (x,y,t)):
                invalid['nonfinite_or_nonnumeric']+=1;continue
            if not (0<=x<1680 and 0<=y<1050):invalid['out_of_display']+=1
            if t<0:invalid['negative_duration']+=1
        if row['length']:initial.append([row['X'][0],row['Y'][0]])
    return dict(schema_version=1,source_sha256=hashlib.sha256(raw).hexdigest(),trials=len(rows),fixations=points,
                splits=dict(splits),images_per_split={k:len(v) for k,v in images.items()},
                image_overlap={f'{a}/{b}':len(images[a]&images[b]) for a,b in [('train','valid'),('train','test'),('valid','test')]},
                subjects=sorted(subjects),duplicate_trials=duplicates,invalid=dict(invalid),
                coordinate_space='1680x1050 presentation canvas, aspect-preserving zero padding',
                initial_fixation='included in source; preserve separately as start condition',
                duration_units='not explicitly specified in downloaded README; do not convert yet',
                test_available=bool(splits['test']),filtered=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('fixations');p.add_argument('--output',required=True)
    args=p.parse_args();result=audit(args.fixations)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
