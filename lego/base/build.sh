#!/bin/bash
# Build the shared grid2op base image. Run from anywhere:
#   bash lego/base/build.sh [tag]
#
# Assembles a minimal build context (only the files the image needs) in a temp
# dir, so we never walk the large GridZero root (outputs/, wandb/, .sandbox-ws-*,
# root-owned dirs that trip the legacy builder's context check).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TAG="${1:-gridzero-rl-base:v1}"

CTX="$(mktemp -d /tmp/grz-base-ctx.XXXXXX)"
trap 'rm -rf "$CTX"' EXIT

# Copy exactly what the Dockerfile COPYs.
cp -r "$ROOT/backend"          "$CTX/backend"
mkdir -p "$CTX/cli"
cp     "$ROOT/cli/simctl"      "$CTX/cli/simctl"
cp     "$ROOT/AGENTS.md"       "$CTX/AGENTS.md"
cp -r  "$ROOT/docs"            "$CTX/docs"
cp -r  "$ROOT/recipes"         "$CTX/recipes"
mkdir -p "$CTX/lego/base"
cp     "$ROOT/lego/base/entrypoint.sh" "$CTX/lego/base/entrypoint.sh"
cp     "$ROOT/lego/base/Dockerfile"    "$CTX/lego/base/Dockerfile"

# Bake the grid2op env data in so each fresh container skips the 294MB download
# (grid2op uses ~/data_grid2op/<env> if present, else downloads).
mkdir -p "$CTX/data_grid2op"
cp -r "$HOME/data_grid2op/l2rpn_case14_sandbox" "$CTX/data_grid2op/l2rpn_case14_sandbox"
rm -rf "$CTX/data_grid2op/l2rpn_case14_sandbox/__pycache__"

docker build -f "$CTX/lego/base/Dockerfile" -t "$TAG" "$CTX"
echo "built $TAG"
docker image inspect --format '{{.Size}}' "$TAG" | awk '{printf "size: %.2f GB\n", $1/1e9}'
