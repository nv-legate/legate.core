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

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from .. import legion

if TYPE_CHECKING:
    import numpy.typing as npt

    from . import Point


class Transform:
    trans: npt.NDArray[np.int64]

    def __init__(self, M: int, N: int, eye: bool = True):
        """
        A Transform wraps an `legion_transform_{m}x{n}_t` in the Legion C API.
        A transform is simply an MxN matrix that can be used to convert Point
        objects from one coordinate space to another.
        """

        self.M = M
        self.N = N
        if eye:
            self.trans = np.eye(M, N, dtype=np.int64)
        else:
            self.trans = np.zeros((M, N), dtype=np.int64)
        self.handle: Optional[Any] = None

    def apply(self, point: Point) -> tuple[float, ...]:
        """
        Convert an N-D Point into an M-D point using this transform
        """
        if len(point) != self.N:
            raise ValueError("Dimension mismatch")
        result: list[float] = []
        for m in range(self.M):
            value = 0
            for n in range(self.N):
                value += self.trans[m, n] * point[n]
            result.append(value)
        return tuple(result)

    def compose(self, outer: Transform) -> Transform:
        """
        Construct a composed transform of this transform with another transform
        """
        if outer.N != self.M:
            raise ValueError("Dimension mismatch")
        result = Transform(outer.M, self.N, eye=False)
        np.matmul(outer.trans, self.trans, out=result.trans)
        return result

    def raw(self) -> Any:
        if self.handle is None:
            self.handle = legion.legion_domain_transform_identity(
                self.M, self.N
            )
        self.handle.m = self.M
        self.handle.n = self.N
        for m in range(self.M):
            for n in range(self.N):
                self.handle.matrix[m * self.N + n] = self.trans[m, n]
        return self.handle

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Transform):
            return NotImplemented
        return (
            self.M == other.M
            and self.N == other.N
            and np.array_equal(self.trans, other.trans)
        )

    def __hash__(self) -> int:
        return hash(self.trans.tobytes())

    def __str__(self) -> str:
        return np.array_repr(self.trans).replace("\n", "").replace(" ", "")


# An Affine Transform for points in one space to points in another
class AffineTransform:
    transform: npt.NDArray[np.int64]

    def __init__(self, M: int, N: int, eye: bool = True):
        """
        An AffineTransform wraps a `legion_affine_transform_{m}x{n}_t` in the
        Legion C API. The AffineTransform class represents an affine transform
        as a MxN affine transform as an (M+1)x(N+1) matrix and can used to
        transform N-D Point objects into M-D Point objects. AffineTransform
        objects can also be naturally composed to construct new
        AffineTransforms.
        """

        self.M = M
        self.N = N
        if eye:
            self.transform = np.eye(M + 1, N + 1, dtype=np.int64)
        else:
            self.transform = np.zeros((M + 1, N + 1), dtype=np.int64)
            self.transform[self.M, self.N] = 1
        self.handle: Optional[Any] = None

    @property
    def offset(self) -> npt.NDArray[np.int64]:
        return self.transform[: self.M, self.N]

    @offset.setter
    def offset(self, offset: float) -> None:
        self.transform[: self.M, self.N] = offset

    @property
    def trans(self) -> npt.NDArray[np.int64]:
        return self.transform[: self.M, : self.N]

    @trans.setter
    def trans(self, transform: Transform) -> None:
        self.transform[: self.M, : self.N] = transform

    def apply(self, point: Point) -> tuple[Any, ...]:
        """
        Convert an N-D Point into an M-D point using this transform
        """
        if len(point) != self.N:
            raise ValueError("Dimension mismatch")
        pin = np.ones(self.N + 1, dtype=np.int64)
        pin[: self.N] = point
        pout = np.dot(self.transform, pin)
        return tuple(pout[: self.M])

    def compose(self, outer: AffineTransform) -> AffineTransform:
        """
        Construct a composed transform of this transform with another transform
        """
        if outer.N != self.M:
            raise ValueError("Dimension mismatch")
        result = AffineTransform(outer.M, self.N, eye=False)
        np.matmul(outer.transform, self.transform, out=result.transform)
        return result

    def raw(self) -> Any:
        if self.handle is None:
            self.handle = legion.legion_domain_affine_transform_identity(
                self.M, self.N
            )
        self.handle.transform.m = self.M
        self.handle.transform.n = self.N
        self.handle.offset.dim = self.M
        for m in range(self.M):
            for n in range(self.N):
                self.handle.transform.matrix[m * self.N + n] = self.transform[
                    m, n
                ]
            self.handle.offset.point_data[m] = self.transform[m, self.N]
        return self.handle

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AffineTransform):
            return NotImplemented
        return (
            self.M == other.M
            and self.N == other.N
            and np.array_equal(self.transform, other.transform)
        )

    def __hash__(self) -> int:
        return hash(self.transform.tobytes())

    def __str__(self) -> str:
        return np.array_repr(self.transform).replace("\n", "").replace(" ", "")
