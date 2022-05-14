#!/bin/bash
set -e

USER="sarusso"
VERSION="20220512"

# Images are also tagged and pushed as "latest" below

#============================================================

echo "Tagging $VERSION..."
docker tag timeseria-base-amd64 docker.io/$USER/timeseria-base-amd64:$VERSION
docker tag timeseria-base-arm64v8 docker.io/$USER/timeseria-base-arm64v8:$VERSION

echo "Pushing images for $VERSION"
docker push docker.io/$USER/timeseria-base-amd64:$VERSION
docker push docker.io/$USER/timeseria-base-arm64v8:$VERSION

echo "Creating manifest for $VERSION..."
# https://unix.stackexchange.com/questions/393310/preventing-shell-from-exiting-when-set-e-is-turned-on
docker manifest rm docker.io/$USER/timeseria-base:$VERSION || true
docker manifest create \
docker.io/$USER/timeseria-base:$VERSION \
--amend docker.io/$USER/timeseria-base-amd64:$VERSION \
--amend docker.io/$USER/timeseria-base-arm64v8:$VERSION

echo "Pushing manifest for $VERSION"
docker manifest push docker.io/$USER/timeseria-base:$VERSION

#============================================================

echo "Tagging latest..."
docker tag timeseria-base-amd64 docker.io/$USER/timeseria-base-amd64:latest
docker tag timeseria-base-arm64v8 docker.io/$USER/timeseria-base-arm64v8:latest

echo "Pushing images for latest"
docker push docker.io/$USER/timeseria-base-amd64:latest
docker push docker.io/$USER/timeseria-base-arm64v8:latest

echo "Creating manifest for latest..."
# https://unix.stackexchange.com/questions/393310/preventing-shell-from-exiting-when-set-e-is-turned-on
docker manifest rm docker.io/$USER/timeseria-base:latest || true
docker manifest create \
docker.io/$USER/timeseria-base:latest \
--amend docker.io/$USER/timeseria-base-amd64:latest \
--amend docker.io/$USER/timeseria-base-arm64v8:latest

echo "Pushing manifest for latest"
docker manifest push docker.io/$USER/timeseria-base:latest
