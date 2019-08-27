#! /bin/bash
flag=${1:-false}
s=$(cat ./build.sbt | grep 'val mainVersion = ')
s=${s#*'"'}; s=${s%'"'*} 

echo "Version is ${s}"

if [ $flag == "true" ]
then
    echo 'Running sbt makePom utility'
    sbt makePom notebook/makePom
fi

echo 'Copying poms'
cp target/scala-2.12/dynaml_2.12-${s}.pom ./pom.xml
cp dynaml-core/target/scala-2.12/dynaml-core_2.12-${s}.pom ./dynaml-core/pom.xml
cp dynaml-repl/target/scala-2.12/dynaml-repl_2.12-${s}.pom ./dynaml-repl/pom.xml
cp dynaml-pipes/target/scala-2.12/dynaml-pipes_2.12-${s}.pom ./dynaml-pipes/pom.xml
cp dynaml-examples/target/scala-2.12/dynaml-examples_2.12-${s}.pom ./dynaml-examples/pom.xml
cp dynaml-notebook/target/scala-2.12/dynaml-notebook_2.12-${s}.pom ./dynaml-notebook/pom.xml
