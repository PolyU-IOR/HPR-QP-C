"""Live ownership bridge. All models use the same S734 correction."""
import time
from .precision_driver import run
from .access import SparseAccess,SpatialAccess,AccessDecline
from .band import qualified_access
from hpr_polish.adapters import selected

def polish_common(model,u,cfg):
    tick=time.monotonic()
    family,_,_=selected(model)
    # No benchmark identity; these gates select only equivalent equation access.
    name='spatial' if family=='spatial' else 'band' if family=='banded_native' else 'sparse'
    factory=qualified_access if name=='band' else None
    try:
        result=run(model,u,name,access_factory=factory,seconds=max(1e-9,cfg.seconds-(time.monotonic()-tick)))
        result.pop('first_direction',None)
        assert result.pop('native_candidate',None) is None
    except (AccessDecline,RuntimeError,ValueError,MemoryError,FloatingPointError) as exc:
        result=dict(candidate=None,status='ACCESS_OR_NUMERICAL_REJECT',certified=False,
                    rejection_type=type(exc).__name__,reason=str(exc),trace=[],
                    derivative_refreshes=0,active_set_updates=0)
    result.update(correction='S734_common_projected_KKT',detected_structure=family,
                  requested_access=name,adapter_s=time.monotonic()-tick,
                  iterative_ssn=False,relinearizations=0,partial_feedback=False)
    return result
