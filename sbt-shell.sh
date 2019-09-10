#! /bin/bash

pTF=${3:-false}

gpuFlag=${2:-false}

heapSize=${1:-4096m}

echo "Starting sbt shell with Build Executable Heap Size = $heapSize and GPU Flag = $gpuFlag"

sbt -Dheap=${heapSize} -Dgpu=${gpuFlag} -DpackagedTF=${pTF}