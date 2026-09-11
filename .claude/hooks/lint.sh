#!/usr/bin/env bash
# PostToolUse hook: format + lint any Python file Claude just edited or wrote.
# Prints ruff findings so Claude sees and fixes them immediately.
set -u
f="$(jq -r '.tool_input.file_path // .tool_response.filePath // empty')"
[ -z "$f" ] && exit 0
case "$f" in *.py) ;; *) exit 0 ;; esac
[ -f "$f" ] || exit 0
cd "$(dirname "$0")/../.." || exit 0
uv run --quiet ruff format "$f" >/dev/null 2>&1 || uvx ruff format "$f" >/dev/null 2>&1
out="$(uv run --quiet ruff check --fix "$f" 2>&1 || uvx ruff check --fix "$f" 2>&1)"
rc=$?
if [ $rc -ne 0 ]; then
  echo "ruff findings in $f:" >&2
  echo "$out" >&2
  exit 2
fi
exit 0
