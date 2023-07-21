# Copyright 2023 NVIDIA Corporation
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

import importlib
import textwrap

from docutils import nodes
from docutils.parsers.rst.directives import unchanged
from docutils.statemachine import ViewList
from jinja2 import Template
from sphinx.errors import SphinxError
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import nested_parse_with_titles

from legate.util.settings import EnvOnlySetting, PrioritizedSetting, _Unset

SETTINGS_DETAIL = Template(
    """
{% for setting in settings %}

``{{ setting['name'] }}``
{{ "''''" +  "'" * setting['name']|length }}

:**Type**: {{ setting['type'] }}
:**Env var**: ``{{ setting['env_var'] }}``
:**Default**: {{ setting['default'] }}

{{ setting['help'] }}

{% endfor %}
"""
)


class SettingsDirective(SphinxDirective):
    has_content = True
    required_arguments = 1
    optional_arguments = 1
    option_spec = {"module": unchanged}

    def run(self):
        obj_name = " ".join(self.arguments)
        module_name = self.options["module"]

        try:
            module = importlib.import_module(module_name)
        except ImportError:
            raise SphinxError(
                f"Unable to generate reference docs for {obj_name}: "
                f"couldn't import module {module_name}"
            )

        obj = getattr(module, obj_name, None)
        if obj is None:
            raise SphinxError(
                f"Unable to generate reference docs for {obj_name}: "
                f"no model {obj_name} in module {module_name}"
            )

        settings = []
        for x in obj.__class__.__dict__.values():
            if isinstance(x, PrioritizedSetting):
                default = "(Unset)" if x.default is _Unset else repr(x.default)
            elif isinstance(x, EnvOnlySetting):
                default = repr(x.default)
                if x._test_default is not _Unset:
                    default += f" (test-mode default: {x._test_default!r})"
            else:
                continue

            settings.append(
                {
                    "name": x.name,
                    "env_var": x.env_var,
                    "type": x.convert_type,
                    "help": textwrap.dedent(x.help),
                    "default": default,
                }
            )

        rst_text = SETTINGS_DETAIL.render(
            name=obj_name, module_name=module_name, settings=settings
        )
        return self.parse(rst_text, "<settings>")

    def parse(self, rst_text, annotation):
        result = ViewList()
        for line in rst_text.split("\n"):
            result.append(line, annotation)
        node = nodes.paragraph()
        node.document = self.state.document
        nested_parse_with_titles(self.state, result, node)
        return node.children


def setup(app):
    """Required Sphinx extension setup function."""
    app.add_directive_to_domain("py", "settings", SettingsDirective)

    return dict(parallel_read_safe=True, parallel_write_safe=True)
