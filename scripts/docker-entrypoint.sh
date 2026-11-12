#!/bin/sh
set -ea

# The platform mounts configuration under /app/configs before this script runs.
# Resolve imports and relative application paths from the image's workspace.
cd /app

# Load optional runtime-mounted shell assignments and export them to Gateway.
# Values in the file override same-name variables supplied by the platform.
env_file="${SAFACTORY_ENV_FILE:-/app/configs/.env}"
if [ -f "${env_file}" ]; then
    . "${env_file}"
fi
set +a

# With no arguments, the script alone starts Gateway with the mounted config.
# SAFACTORY_GATEWAY_CONFIG can also be set in the mounted .env file.
if [ "$#" -eq 0 ]; then
    set -- --config "${SAFACTORY_GATEWAY_CONFIG:-/app/configs/gateway.yaml}"
fi

# Accept Gateway flags while preserving support for a full command override.
case "$1" in
    -*) set -- python -B -u -m gateway "$@" ;;
esac

# Python replaces the shell and receives container shutdown signals directly.
exec "$@"
