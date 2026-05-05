#!/usr/bin/env bash
# activate-claude-auth.sh — sourced by pixi on dev-environment activation.
#
# Auto-exports CLAUDE_CODE_OAUTH_TOKEN from a user-managed file if it
# exists. Lets users skip the manual `export CLAUDE_CODE_OAUTH_TOKEN=...`
# in shell rcs and avoids leaking the token outside dev pixi envs.
#
# Token file is written by `./uw claude-set-token` (mode 0600).
# To remove: rm ~/.claude/uw-token

if [ -f "$HOME/.claude/uw-token" ]; then
    # Strip ALL whitespace (covers \r\n line endings, accidental leading
    # spaces, and any whitespace a manual editor leaves behind). $() alone
    # only trims trailing newlines, which is not enough here.
    _uw_claude_token="$(tr -d '[:space:]' < "$HOME/.claude/uw-token")"
    if [ -n "$_uw_claude_token" ]; then
        CLAUDE_CODE_OAUTH_TOKEN="$_uw_claude_token"
        export CLAUDE_CODE_OAUTH_TOKEN
    fi
    unset _uw_claude_token
fi
