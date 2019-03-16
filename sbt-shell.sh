#! /bin/bash

gpuFlag=${2:-false}

heapSize=${1:-4096m}

sbt -Dheap=${heapSize} -Dgpu=${gpuFlag}