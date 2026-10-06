#!/usr/bin/env bash
# Start the Underworld3 transcript MCP server in this checkout's pixi
# environment (the one ./uw setup recorded in .pixi-env). Claude Code runs
# this through .mcp.json; the server speaks MCP on stdio.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_name="$(cat "$here/.pixi-env" 2>/dev/null || echo default)"
export PIXI_PROJECT_MANIFEST="$here/pixi.toml"
exec pixi run -e "$env_name" python -m underworld3.mcp
