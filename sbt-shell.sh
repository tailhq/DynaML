#! /bin/bash

pTF=${3:-false}

gpuFlag=${2:-false}

heapSize=${1:-4096m}

sbt -Dheap=${heapSize} -Dgpu=${gpuFlag} -DpackagedTF=${pTF}