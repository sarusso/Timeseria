#!/bin/bash
set -e

USER="sarusso"
VERSION="v0.1.6"

echo "Tagging..."
docker tag timeseria-amd64 docker.io/$USER/timeseria-amd64:$VERSION
docker tag timeseria-arm64v8 docker.io/$USER/timeseria-arm64v8:$VERSION

echo "Pushing images"
docker push docker.io/$USER/timeseria-amd64:$VERSION
docker push docker.io/$USER/timeseria-arm64v8:$VERSION

echo "Creating manifest..."
docker manifest rm docker.io/$USER/timeseria:$VERSION
docker manifest create \
docker.io/$USER/timeseria:$VERSION \
--amend docker.io/$USER/timeseria-amd64:$VERSION \
--amend docker.io/$USER/timeseria-arm64v8:$VERSION

echo "Pushing manifest"
docker manifest push docker.io/$USER/timeseria:$VERSION
