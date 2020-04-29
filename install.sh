#! /bin/bash

pTF=${3:-false}

gpuFlag=${2:-false}

heapSize=${1:-4096m}

sbt -Dheap=${heapSize} -Dgpu=${gpuFlag} -DpackagedTF=${pTF} "; stage; publishLocal"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

chmod +x ./target/universal/stage/bin/dynaml
sudo rm /usr/local/bin/dynaml
sudo ln -s ${DIR}/target/universal/stage/bin/dynaml /usr/local/bin/dynaml