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


class Expr(object):
    def __eq__(self, rhs):
        return Alignment(self, rhs)

    def __le__(self, rhs):
        return Containment(self, rhs)

    def __add__(self, offset):
        if not isinstance(offset, tuple):
            raise ValueError("Offset must be a tuple")
        elif self.ndim != len(offset):
            raise ValueError("Dimensions don't match")
        return Translate(self, offset)

    def broadcast(self):
        return Broadcast(self)


class Lit(Expr):
    def __init__(self, part):
        self._part = part

    @property
    def ndim(self):
        raise NotImplementedError("ndim not implemented for literals")

    @property
    def closed(self):
        return True

    def __repr__(self):
        return f"Lit({self._part})"

    def subst(self):
        return self

    def reduce(self):
        return self


class PartSym(Expr):
    def __init__(self, op_hash, op_name, store, id, disjoint, complete):
        self._op_hash = op_hash
        self._op_name = op_name
        self._store = store
        self._id = id
        self._disjoint = disjoint
        self._complete = complete

    @property
    def ndim(self):
        return self._store.ndim

    @property
    def store(self):
        return self._store

    @property
    def closed(self):
        return False

    def __repr__(self):
        disj = "D" if self._disjoint else "A"
        comp = "C" if self._complete else "I"
        return f"X{self._id}({disj},{comp})@{self._op_name}"

    def __hash__(self):
        return hash((self._op_hash, self._id))

    def subst(self, mapping):
        return Lit(mapping[self])

    def reduce(self):
        return self


class Translate(Expr):
    def __init__(self, expr, offset):
        if not isinstance(expr, (PartSym, Lit)):
            raise NotImplementedError(
                "Compound expression is not supported yet"
            )
        self._expr = expr
        self._offset = offset

    @property
    def ndim(self):
        return len(self._offset)

    @property
    def closed(self):
        return self._expr.closed

    def __repr__(self):
        return f"{self._expr} + {self._offset}"

    def subst(self, mapping):
        return Translate(self._expr.subst(mapping), self._offset)

    def reduce(self):
        expr = self._expr.reduce()
        assert isinstance(expr, Lit)
        part = expr._part
        return Lit(part.translate(self._offset))


class Constraint(object):
    pass


class Alignment(Constraint):
    def __init__(self, lhs, rhs):
        if not isinstance(lhs, PartSym) or not isinstance(rhs, PartSym):
            raise NotImplementedError(
                "Alignment between complex expressions is not supported yet"
            )
        self._lhs = lhs
        self._rhs = rhs

    def __repr__(self):
        return f"{self._lhs} == {self._rhs}"


class Containment(Constraint):
    def __init__(self, lhs, rhs):
        if not isinstance(rhs, PartSym):
            raise NotImplementedError(
                "Containment on a complex expression is not supported yet"
            )
        self._lhs = lhs
        self._rhs = rhs

    def __repr__(self):
        return f"{self._lhs} <= {self._rhs}"


class Broadcast(Constraint):
    def __init__(self, expr):
        self._expr = expr

    def __repr__(self):
        return f"Broadcast({self._expr})"
