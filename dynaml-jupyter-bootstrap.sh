version=${1:-v2.0-SNAPSHOT}

if [ -e dynaml-kernel ]; then
    echo "DynaML kernel installer already exists in the current directory; proceeding to overwrite it."
    rm dynaml-kernel
fi

coursier bootstrap --embed-files=false \
    -r central \
    -r sonatype:snapshots \
    -r https://oss.sonatype.org/content/repositories/snapshots \
    -r http://maven.jzy3d.org/releases \
    -r https://nexus.bedatadriven.com/content/groups/public/ \
    -r http://dl.bintray.com/scalaz/releases/ \
    -r https://dl.bintray.com/cibotech/public/ \
    -r https://jitpack.io \
    io.github.transcendent-ai-labs:dynaml-notebook_2.12:$version \
    -o dynaml-kernel

./dynaml-kernel --install --force