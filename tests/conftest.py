"""
Test configuration.

Pre-stubs the `genericsuite.fastapilib.util.create_app` module to break a
circular import in the installed genericsuite package:

  framework_abs_layer (loading)
    → imports fastapilib.util.create_app
      → imports fastapilib.endpoints.storage_retrieval
        → imports Request from framework_abs_layer  ← still loading!

By registering a stub before any test module is collected, the
`importlib.import_module` call in framework_abs_layer finds the stub in
sys.modules and never triggers the real import chain.
"""
import sys
import types

sys.modules.setdefault(
    "genericsuite.fastapilib.util.create_app",
    types.ModuleType("genericsuite.fastapilib.util.create_app"),
)
