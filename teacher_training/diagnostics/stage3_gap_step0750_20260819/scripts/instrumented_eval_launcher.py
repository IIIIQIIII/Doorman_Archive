"""Launch the official DoorMan eval module with runtime-only diagnostics."""

import runpy

from stage3_diagnostics_hook import install_import_hook


install_import_hook()
runpy.run_module("gr00t.rl.eval_agent_trl", run_name="__main__")

