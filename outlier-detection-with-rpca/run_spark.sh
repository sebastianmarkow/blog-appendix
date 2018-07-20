#!/usr/bin/env bash

set -o errexit
set -o pipefail

docker run --rm -p 8888:8888 -p 4040:4040 -v "$PWD":/home/jovyan/work jupyter/all-spark-notebook:3772fffc4aa4
