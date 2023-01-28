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

import sys
from unittest.mock import MagicMock

import pytest

import legate.rc as m


@pytest.fixture
def mock_has_legion_context(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    stub = MagicMock()
    monkeypatch.setattr("legate.rc.has_legion_context", stub)
    return stub


class Test_check_legion:
    def test_True(self, mock_has_legion_context: MagicMock) -> None:
        mock_has_legion_context.return_value = True
        assert m.check_legion() is None  # type: ignore[func-returns-value]

    def test_True_with_msg(self, mock_has_legion_context: MagicMock) -> None:
        mock_has_legion_context.return_value = True
        assert m.check_legion(msg="custom") is None  # type: ignore[func-returns-value]  # noqa

    def test_False(self, mock_has_legion_context: MagicMock) -> None:
        mock_has_legion_context.return_value = False
        with pytest.raises(RuntimeError) as e:
            m.check_legion()
            assert str(e) == m.LEGION_WARNING

    def test_False_with_msg(self, mock_has_legion_context: MagicMock) -> None:
        mock_has_legion_context.return_value = False
        with pytest.raises(RuntimeError) as e:
            m.check_legion(msg="custom")
            assert str(e) == "custom"


@pytest.mark.skip
class Test_has_legion_context:
    def test_True(self) -> None:
        assert m.has_legion_context() is True

    # It does not seem possible to patch CFFI libs, so testing
    # the "False" branch is not really feasible
    @pytest.mark.skip
    def test_False(self) -> None:
        pass


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
