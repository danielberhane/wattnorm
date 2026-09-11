#!/usr/bin/env sh
# Usage: entrypoint.sh scorer | simulator [args] | train [args] | evaluate [args] | retrain [args]
set -e
role="$1"; shift || true
case "$role" in
  scorer)    exec python -m smartbuilding.service ;;
  simulator) exec python -m smartbuilding.simulator "$@" ;;
  train)     exec python scripts/train.py "$@" ;;
  evaluate)  exec python scripts/evaluate.py "$@" ;;
  retrain)   exec python scripts/retrain.py "$@" ;;
  *)         exec "$role" "$@" ;;
esac
