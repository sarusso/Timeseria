#!/bin/bash
set -e

# Move to project root
cd ../../

# Build Docker container
echo -e  "\n==============================="
echo -e  "|  Building Docker container  |"
echo -e  "===============================\n"

if [[ "x$CACHE" == "xFalse" ]]; then
    echo "Building without cache."
    docker buildx build --progress=plain --no-cache -f containers/Ubuntu_20.04/Dockerfile ./ --platform linux/amd64 -t timeseria-amd64  --load #--push
    docker buildx build --progress=plain --no-cache -f containers/Ubuntu_20.04/Dockerfile ./ --platform linux/arm64/v8 -t timeseria-arm64v8  --load #--push
else
    echo "Building with cache. Use CACHE=False to disable it."
    docker buildx build --progress=plain -f containers/Ubuntu_20.04/Dockerfile ./ --platform linux/amd64 -t timeseria-amd64 --load #--push
    docker buildx build --progress=plain -f containers/Ubuntu_20.04/Dockerfile ./ --platform linux/arm64/v8 -t timeseria-arm64v8  --load #--push
fi
