#!/bin/bash
docker manifest create \
docker.io/sarusso/timeseria \
--amend docker.io/sarusso/timeseria-amd64 \
--amend docker.io/sarusso/timeseria-arm64v8

# docker manifest push docker.io/sarusso/timeseria
