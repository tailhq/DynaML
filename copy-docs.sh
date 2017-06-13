#! /bin/bash
sbt stage
cp -R dynaml-core/target/scala-2.11/api/* ../transcendent-ai-labs.github.io/api_docs/DynaML/v1.4.3/dynaml-core/
cp -R dynaml-pipes/target/scala-2.11/api/* ../transcendent-ai-labs.github.io/api_docs/DynaML/v1.4.3/dynaml-pipes/
cp -R dynaml-examples/target/scala-2.11/api/* ../transcendent-ai-labs.github.io/api_docs/DynaML/v1.4.3/dynaml-examples/
cp -R target/scala-2.11/api/* ../transcendent-ai-labs.github.io/api_docs/DynaML/v1.4.3/dynaml-repl/