# Copyright 2021-2022 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Iterator, Optional, Protocol, Union

from .partition import Restriction

if TYPE_CHECKING:
    from .partition import PartitionBase
    from .store import Store
    from .transform import Restrictions


class Expr(Protocol):
    @property
    def ndim(self) -> int:
        ...

    def subst(self, mapping: dict[PartSym, PartitionBase]) -> Expr:
        ...

    def reduce(self) -> Expr:
        ...

    def unknowns(self) -> Iterator[PartSym]:
        ...

    def __eq__(self, rhs: Expr) -> Constraint:  # type: ignore [override]
        return Alignment(self, rhs)

    def __le__(self, rhs: Expr) -> Constraint:
        return Containment(self, rhs)

    def __add__(self, offset: tuple[int]) -> Translate:
        if not isinstance(offset, tuple):
            raise ValueError("Offset must be a tuple")
        elif self.ndim != len(offset):
            raise ValueError("Dimensions don't match")
        return Translate(self, offset)


class Lit(Expr):
    def __init__(self, part: Any) -> None:
        self._part = part

    @property
    def ndim(self) -> int:
        raise NotImplementedError("ndim not implemented for literals")

    @property
    def closed(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"Lit({self._part})"

    def subst(self, mapping: dict[PartSym, PartitionBase]) -> Expr:
        return self

    def reduce(self) -> Lit:
        return self

    def unknowns(self) -> Iterator[PartSym]:
        pass


class PartSym(Expr):
    def __init__(
        self,
        op_hash: int,
        op_name: str,
        store: Store,
        id: int,
        disjoint: bool,
        complete: bool,
    ) -> None:
        self._op_hash = op_hash
        self._op_name = op_name
        self._store = store
        self._id = id
        self._disjoint = disjoint
        self._complete = complete

    @property
    def ndim(self) -> int:
        return self._store.ndim

    @property
    def store(self) -> Store:
        return self._store

    @property
    def closed(self) -> bool:
        return False

    def __repr__(self) -> str:
        disj = "D" if self._disjoint else "A"
        comp = "C" if self._complete else "I"
        return f"X{self._id}({disj},{comp})@{self._op_name}"

    def __hash__(self) -> int:
        return hash((self._op_hash, self._id))

    def subst(self, mapping: dict[PartSym, PartitionBase]) -> Expr:
        return Lit(mapping[self])

    def reduce(self) -> PartSym:
        return self

    def unknowns(self) -> Iterator[PartSym]:
        yield self

    def broadcast(
        self, axes: Optional[Union[int, Iterable[int]]] = None
    ) -> Broadcast:
        if axes is None:
            axes = set(range(self.ndim))
        else:
            if isinstance(axes, Iterable):
                axes = set(axes)
            else:
                axes = {axes}
        restrictions = tuple(
            Restriction.RESTRICTED if i in axes else Restriction.UNRESTRICTED
            for i in range(self.ndim)
        )
        return Broadcast(self, restrictions)


class Translate(Expr):
    # TODO: For now we will interpret this expression as `expr + [1, offset]`.
    def __init__(self, expr: Expr, offset: tuple[int]) -> None:
        if not isinstance(expr, (PartSym, Lit)):
            raise NotImplementedError(
                "Compound expression is not supported yet"
            )
        self._expr = expr
        self._offset = offset

    @property
    def ndim(self) -> int:
        return len(self._offset)

    @property
    def closed(self) -> bool:
        return self._expr.closed

    def __repr__(self) -> str:
        return f"{self._expr} + {self._offset}"

    def subst(self, mapping: dict[PartSym, PartitionBase]) -> Expr:
        return Translate(self._expr.subst(mapping), self._offset)

    def reduce(self) -> Lit:
        expr = self._expr.reduce()
        assert isinstance(expr, Lit)
        part = expr._part
        return Lit(part.translate_range(self._offset))

    def unknowns(self) -> Iterator[PartSym]:
        for unknown in self._expr.unknowns():
            yield unknown


class Constraint:
    pass


class Alignment(Constraint):
    def __init__(self, lhs: Expr, rhs: Expr) -> None:
        if not isinstance(lhs, PartSym) or not isinstance(rhs, PartSym):
            raise NotImplementedError(
                "Alignment between complex expressions is not supported yet"
            )
        self._lhs = lhs
        self._rhs = rhs

    def __repr__(self) -> str:
        return f"{self._lhs} == {self._rhs}"


class Containment(Constraint):
    def __init__(self, lhs: Expr, rhs: Expr):
        if not isinstance(rhs, PartSym):
            raise NotImplementedError(
                "Containment on a complex expression is not supported yet"
            )
        self._lhs = lhs
        self._rhs = rhs

    def __repr__(self) -> str:
        return f"{self._lhs} <= {self._rhs}"


class Broadcast(Constraint):
    def __init__(self, expr: PartSym, restrictions: Restrictions) -> None:
        self._expr = expr
        self._restrictions = restrictions

    def __repr__(self) -> str:
        return f"Broadcast({self._expr}, axes={self._restrictions})"
