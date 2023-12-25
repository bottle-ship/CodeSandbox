import functools
import typing as t
from inspect import signature

from pydantic import (
    create_model,
    ValidationError
)

__all__ = ["validate_params"]


def _make_annotation(type_hint: t.Any, constraints: t.Any) -> t.Any:
    if isinstance(constraints, (tuple, list)) and type_hint not in constraints:
        return (type_hint,) + tuple(constraints)
    else:
        return type_hint, constraints if type(constraints) != type(type_hint) else constraints


def validate_params(parameter_constraints: t.Dict[str, t.Any]):

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(func.__qualname__)

            parameters = signature(func).bind(*args, **kwargs)
            parameters.apply_defaults()
            params = {k: v for k, v in parameters.arguments.items() if k in parameter_constraints}

            func_type_hints = t.get_type_hints(func)
            constraints = {
                k: _make_annotation(func_type_hints[k], v)
                for k, v in parameter_constraints.items()
                if k in func_type_hints
            }
            print(constraints)

            try:
                validator = create_model("ParamValidator", **constraints)
                validator(**params)

                return func(*args, **kwargs)
            except ValidationError as e:
                raise ValueError(e)

        return wrapper

    return decorator
