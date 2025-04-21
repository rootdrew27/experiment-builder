from typing import Callable, Any
from multipledispatch import dispatch # type: ignore

class Action:
    def __init__(self, func:Callable):
        self.func = func
    
    @dispatch(tuple)
    def __call__(self, params:tuple) -> Any:
        return self.func

    @dispatch(dict)
    def __call__(self, kw_params:dict) -> Any:
        return 2