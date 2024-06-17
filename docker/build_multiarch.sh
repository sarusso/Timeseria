#!/bin/bash
set -e

# Move to project root
cd ../

if [[ "x$CACHE" == "xFalse" ]]; then
    echo "Building without cache."
    docker buildx build --progress=plain --no-cache -f docker/Dockerfile ./ --platform linux/amd64 -t timeseria-amd64  --load #--push
    docker buildx build --progress=plain --no-cache -f docker/Dockerfile ./ --platform linux/arm64/v8 -t timeseria-arm64v8  --load #--push
else
    echo "Building with cache. Use CACHE=False to disable it."
    docker buildx build --cache-from=type=local,src=/tmp/.docker-buildx-cache --cache-to=type=local,dest=/tmp/.docker-buildx-cache --progress=plain -f docker/Dockerfile ./ --platform linux/amd64 -t timeseria-amd64 --load #--push
    docker buildx build --cache-from=type=local,src=/tmp/.docker-buildx-cache --cache-to=type=local,dest=/tmp/.docker-buildx-cache --progress=plain -f docker/Dockerfile ./ --platform linux/arm64/v8 -t timeseria-arm64v8  --load #--push
fi
