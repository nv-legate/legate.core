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

import json
from dataclasses import asdict

from pytest_mock import MockerFixture

import legate.jupyter.kernel as m
from legate.driver import LegateDriver
from legate.jupyter.config import Config
from legate.util.system import System

from ...util import Capsys


def test_LEGATE_JUPYTER_KERNEL_SPEC_KEY() -> None:
    assert m.LEGATE_JUPYTER_KERNEL_SPEC_KEY == "__LEGATE_JUPYTER_KERNEL_SPEC__"


def test_LEGATE_JUPYTER_METADATA_KEY() -> None:
    assert m.LEGATE_JUPYTER_METADATA_KEY == "legate"


system = System()


class Test_generate_kernel_spec:
    def test_defatul(self) -> None:
        config = Config([])
        driver = LegateDriver(config, system)

        spec = m.generate_kernel_spec(driver, config)

        expected_env = {
            k: v for k, v in driver.env.items() if k in driver.custom_env_vars
        }
        expected_env[
            m.LEGATE_JUPYTER_KERNEL_SPEC_KEY
        ] = config.kernel.spec_name

        assert spec.display_name == config.kernel.display_name
        assert spec.language == "python"  # type: ignore
        assert spec.argv[:-3] == list(driver.cmd)  # type: ignore
        assert spec.argv[-3].endswith("_legion_kernel.py")  # type: ignore
        assert spec.argv[-2:] == ["-f", "{connection_file}"]  # type: ignore
        assert spec.env == expected_env  # type: ignore
        assert m.LEGATE_JUPYTER_METADATA_KEY in spec.metadata
        metadata = spec.metadata[m.LEGATE_JUPYTER_METADATA_KEY]
        assert metadata == {
            "argv": config.argv[1:],
            "multi_node": asdict(config.multi_node),
            "memory": asdict(config.memory),
            "core": asdict(config.core),
        }


class Test_install_kernel_spec:
    def test_install(self, mocker: MockerFixture, capsys: Capsys) -> None:
        install_mock = mocker.patch(
            "jupyter_client.kernelspec.KernelSpecManager.install_kernel_spec"
        )

        config = Config(
            ["legate-jupyter", "--name", "____fake_test_kernel_123abc_____"]
        )
        driver = LegateDriver(config, system)

        spec = m.generate_kernel_spec(driver, config)

        m.install_kernel_spec(spec, config)

        assert install_mock.call_count == 1
        assert install_mock.call_args[0][1] == config.kernel.spec_name
        assert install_mock.call_args[1] == {
            "user": config.kernel.user,
            "prefix": config.kernel.prefix,
        }

        out, _ = capsys.readouterr()
        assert out == (
            f"Jupyter kernel spec {config.kernel.spec_name} "
            f"({config.kernel.display_name}) "
            "has been installed\n"
        )

    def test_install_verbose(
        self, mocker: MockerFixture, capsys: Capsys
    ) -> None:
        install_mock = mocker.patch(
            "jupyter_client.kernelspec.KernelSpecManager.install_kernel_spec"
        )

        config = Config(
            [
                "legate-jupyter",
                "-v",
                "--name",
                "____fake_test_kernel_123abc_____",
            ]
        )
        driver = LegateDriver(config, system)

        spec = m.generate_kernel_spec(driver, config)

        m.install_kernel_spec(spec, config)

        assert install_mock.call_count == 1
        assert install_mock.call_args[0][1] == config.kernel.spec_name
        assert install_mock.call_args[1] == {
            "user": config.kernel.user,
            "prefix": config.kernel.prefix,
        }

        out, _ = capsys.readouterr()
        assert out == (
            f"Wrote kernel spec file {config.kernel.spec_name}/kernel.json\n\n"
            f"Jupyter kernel spec {config.kernel.spec_name} "
            f"({config.kernel.display_name}) "
            "has been installed\n"
        )

    def test_install_verbose2(
        self, mocker: MockerFixture, capsys: Capsys
    ) -> None:
        install_mock = mocker.patch(
            "jupyter_client.kernelspec.KernelSpecManager.install_kernel_spec"
        )

        config = Config(
            [
                "legate-jupyter",
                "-vv",
                "--name",
                "____fake_test_kernel_123abc_____",
            ]
        )
        driver = LegateDriver(config, system)

        spec = m.generate_kernel_spec(driver, config)

        m.install_kernel_spec(spec, config)

        assert install_mock.call_count == 1
        assert install_mock.call_args[0][1] == config.kernel.spec_name
        assert install_mock.call_args[1] == {
            "user": config.kernel.user,
            "prefix": config.kernel.prefix,
        }

        out, _ = capsys.readouterr()
        spec_json = json.dumps(spec.to_dict(), sort_keys=True, indent=2)
        assert out == (
            f"Wrote kernel spec file {config.kernel.spec_name}/kernel.json\n\n"
            f"\n{spec_json}\n\n"
            f"Jupyter kernel spec {config.kernel.spec_name} "
            f"({config.kernel.display_name}) "
            "has been installed\n"
        )
