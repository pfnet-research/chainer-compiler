#!/bin/bash

if [ x"$@" = x"" ]; then
    echo "Usage: $0 <github-auth-token>"
    exit 1
fi

sed "s/%GITHUB_AUTH_TOKEN%/$@/" docker/Dockerfile.tmpl > docker/Dockerfile
nvidia-docker build docker
