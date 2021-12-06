#!/bin/bash
docker manifest create \
docker.io/sarusso/timeseria:v0.1.2 \
--amend docker.io/sarusso/timeseria-amd64:v0.1.2 \
--amend docker.io/sarusso/timeseria-arm64v8:v0.1.2

#docker manifest push docker.io/sarusso/timeseria:v0.1.2