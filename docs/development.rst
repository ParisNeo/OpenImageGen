Development
===========

Install contributor tools:

.. code-block:: bash

   python -m pip install -e ".[dev,docs]"

Run tests:

.. code-block:: bash

   python -m pytest

The workflow tests use mocked remote transport and do not require a GPU.
Model-download and generation tests should be separated from fast unit tests
when adding heavier integration coverage.

Project layout
--------------

``openimagegen/app.py``
   FastAPI routes, authentication dependencies, static interfaces, and WebSockets.

``openimagegen/service.py``
   Job orchestration, model lifecycle, LoRA lifecycle, generation, and artifacts.

``openimagegen/workflow.py``
   Graph validation, dependency ordering, execution, and request compilation.

``openimagegen/models/``
   Registered model adapters and family-aware generation profiles.

``openimagegen/nodes/``
   Public node API, built-in nodes, and trusted plugin discovery.

``openimagegen/static/``
   Studio, workflow editor, first-run onboarding, hosted-model configuration,
   GGUF catalog, and download-center frontend.

Contribution guidelines
-----------------------

* Preserve typed ports and backward-compatible bare output references.
* Keep credentials out of logs, exports, templates, and returned node values.
* Validate dimensions, file paths, downloads, and user-provided remote settings.
* Restore cached model state after request-scoped LoRA or scheduler changes.
* Add focused tests for graph compilation and adapter-independent behavior.
* Update this documentation for new routes, configuration, or workflow nodes.

Build documentation
-------------------

From the repository root:

.. code-block:: bash

   python -m sphinx -W --keep-going -b html docs docs/_build/html

Open ``docs/_build/html/index.html``. On Windows, ``docs\make.bat html`` is also
available; Unix-like systems can use ``make -C docs html``.