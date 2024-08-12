#!/bin/bash
set -e

# Check if we are in the right place
if [ ! -f ./Dockerfile ]; then
    echo "No Dockerfile found. Are you executing this command in the 'docker' subfolder?"
    exit 1
fi

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
