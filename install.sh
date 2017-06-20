#! /bin/bash

sbt stage

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

chmod +x ./target/universal/stage/bin/dynaml
sudo rm /usr/local/bin/dynaml
sudo ln -s ${DIR}/target/universal/stage/bin/dynaml /usr/local/bin/dynaml
echo 'export DYNAML_HOME='${DIR} >>~/.bash_profile
source ${HOME}/.bash_profile