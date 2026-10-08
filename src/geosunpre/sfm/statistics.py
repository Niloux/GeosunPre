"""Fitted-observation residual summaries."""

import numpy as np


def error_summary(errors):
    errors=np.asarray(errors,dtype=float)
    if not len(errors):
        return {'count':0,'finite_count':0,'median':None,'p90':None,'p99':None,
                'fraction_le_1px':None,'fraction_le_2px':None,'fraction_le_3px':None}
    values=np.where(np.isfinite(errors),errors,np.inf)
    q=np.quantile(values,[.5,.9,.99],method='nearest')
    return {'count':len(values),'finite_count':int(np.isfinite(values).sum()),
            **{k:float(v) if np.isfinite(v) else None for k,v in zip(['median','p90','p99'],q)},
            **{f'fraction_le_{t}px':float(np.mean(values<=t)) for t in [1,2,3]}}
