#! /bin/bash

cp -R dynaml-core/target/scala-2.11/api/* docs/api_docs/v1.4.1/dynaml-core/
cp -R dynaml-pipes/target/scala-2.11/api/* docs/api_docs/v1.4.1/dynaml-pipes/
cp -R dynaml-examples/target/scala-2.11/api/* docs/api_docs/v1.4.1/dynaml-examples/
cp -R target/scala-2.11/api/* docs/api_docs/v1.4.1/dynaml-repl/