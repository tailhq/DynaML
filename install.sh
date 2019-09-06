#! /bin/bash

pTF=${3:-false}

gpuFlag=${2:-false}

heapSize=${1:-4096m}

sbt -Dheap=${heapSize} -Dgpu=${gpuFlag} -DpackagedTF=${pTF} stage

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

chmod +x ./target/universal/stage/bin/dynaml
sudo rm /usr/local/bin/dynaml
sudo ln -s ${DIR}/target/universal/stage/bin/dynaml /usr/local/bin/dynaml
sed -i.bak '/export DYNAML_HOME/d' ~/.bash_profile
echo 'export DYNAML_HOME='${DIR} >>~/.bash_profile
source ~/.bash_profile