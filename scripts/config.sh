#!/usr/bin/env bash
set -euo pipefail

# Configure only the current repository. Override these defaults through the
# environment instead of mutating the user's global Git configuration.
git config user.name "${GIT_USER_NAME:-jianzhnie}"
git config user.email "${GIT_USER_EMAIL:-jianzhnie@126.com}"
