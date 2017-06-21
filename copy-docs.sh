#! /bin/bash

mkdir -p ../transcendent-ai-labs.github.io/api_docs/DynaML/$1
mkdir -p ../transcendent-ai-labs.github.io/api_docs/DynaML/$1/dynaml-core
mkdir -p ../transcendent-ai-labs.github.io/api_docs/DynaML/$1/dynaml-pipes
mkdir -p ../transcendent-ai-labs.github.io/api_docs/DynaML/$1/dynaml-examples
mkdir -p ../transcendent-ai-labs.github.io/api_docs/DynaML/$1/dynaml-repl

sbt stage

cp -R dynaml-core/target/scala-2.11/api/* ../transcendent-ai-labs.github.io/api_docs/DynaML/$1/dynaml-core/
cp -R dynaml-pipes/target/scala-2.11/api/* ../transcendent-ai-labs.github.io/api_docs/DynaML/$1/dynaml-pipes/
cp -R dynaml-examples/target/scala-2.11/api/* ../transcendent-ai-labs.github.io/api_docs/DynaML/$1/dynaml-examples/
cp -R target/scala-2.11/api/* ../transcendent-ai-labs.github.io/api_docs/DynaML/$1/dynaml-repl/