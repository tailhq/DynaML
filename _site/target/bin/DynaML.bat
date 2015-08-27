@REM ----------------------------------------------------------------------------
@REM  Copyright 2001-2006 The Apache Software Foundation.
@REM
@REM  Licensed under the Apache License, Version 2.0 (the "License");
@REM  you may not use this file except in compliance with the License.
@REM  You may obtain a copy of the License at
@REM
@REM       http://www.apache.org/licenses/LICENSE-2.0
@REM
@REM  Unless required by applicable law or agreed to in writing, software
@REM  distributed under the License is distributed on an "AS IS" BASIS,
@REM  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@REM  See the License for the specific language governing permissions and
@REM  limitations under the License.
@REM ----------------------------------------------------------------------------
@REM
@REM   Copyright (c) 2001-2006 The Apache Software Foundation.  All rights
@REM   reserved.

@echo off

set ERROR_CODE=0

:init
@REM Decide how to startup depending on the version of windows

@REM -- Win98ME
if NOT "%OS%"=="Windows_NT" goto Win9xArg

@REM set local scope for the variables with windows NT shell
if "%OS%"=="Windows_NT" @setlocal

@REM -- 4NT shell
if "%eval[2+2]" == "4" goto 4NTArgs

@REM -- Regular WinNT shell
set CMD_LINE_ARGS=%*
goto WinNTGetScriptDir

@REM The 4NT Shell from jp software
:4NTArgs
set CMD_LINE_ARGS=%$
goto WinNTGetScriptDir

:Win9xArg
@REM Slurp the command line arguments.  This loop allows for an unlimited number
@REM of arguments (up to the command line limit, anyway).
set CMD_LINE_ARGS=
:Win9xApp
if %1a==a goto Win9xGetScriptDir
set CMD_LINE_ARGS=%CMD_LINE_ARGS% %1
shift
goto Win9xApp

:Win9xGetScriptDir
set SAVEDIR=%CD%
%0\
cd %0\..\.. 
set BASEDIR=%CD%
cd %SAVEDIR%
set SAVE_DIR=
goto repoSetup

:WinNTGetScriptDir
set BASEDIR=%~dp0\..

:repoSetup


if "%JAVACMD%"=="" set JAVACMD=java

if "%REPO%"=="" set REPO=%BASEDIR%\repo

set CLASSPATH="%BASEDIR%"\etc;"%REPO%"\scala-library-2.10.4.jar;"%REPO%"\scala-reflect-2.10.4.jar;"%REPO%"\scala-compiler-2.10.4.jar;"%REPO%"\config-1.2.1.jar;"%REPO%"\gremlin-java-2.6.0.jar;"%REPO%"\blueprints-core-2.6.0.jar;"%REPO%"\jettison-1.3.3.jar;"%REPO%"\stax-api-1.0.1.jar;"%REPO%"\hppc-0.6.0.jar;"%REPO%"\commons-configuration-1.6.jar;"%REPO%"\commons-digester-1.8.jar;"%REPO%"\commons-beanutils-core-1.8.0.jar;"%REPO%"\pipes-2.6.0.jar;"%REPO%"\frames-2.5.0.jar;"%REPO%"\gremlin-groovy-2.5.0.jar;"%REPO%"\groovy-1.8.9.jar;"%REPO%"\antlr-2.7.7.jar;"%REPO%"\asm-3.2.jar;"%REPO%"\asm-commons-3.2.jar;"%REPO%"\asm-util-3.2.jar;"%REPO%"\asm-analysis-3.2.jar;"%REPO%"\asm-tree-3.2.jar;"%REPO%"\ant-1.8.3.jar;"%REPO%"\ant-launcher-1.8.3.jar;"%REPO%"\jline-0.9.94.jar;"%REPO%"\javassist-3.18.0-GA.jar;"%REPO%"\guava-14.0.1.jar;"%REPO%"\breeze_2.10-0.10.jar;"%REPO%"\breeze-macros_2.10-0.3.1.jar;"%REPO%"\core-1.1.2.jar;"%REPO%"\arpack_combined_all-0.1.jar;"%REPO%"\opencsv-2.3.jar;"%REPO%"\jtransforms-2.4.0.jar;"%REPO%"\commons-math3-3.2.jar;"%REPO%"\spire_2.10-0.7.4.jar;"%REPO%"\spire-macros_2.10-0.7.4.jar;"%REPO%"\slf4j-api-1.7.5.jar;"%REPO%"\scala-csv_2.10-1.1.2.jar;"%REPO%"\log4j-1.2.17.jar;"%REPO%"\jline-2.9.0-1.jar;"%REPO%"\jansi-1.4.jar;"%REPO%"\scala-chart_2.10-0.4.2.jar;"%REPO%"\jfreechart-1.0.17.jar;"%REPO%"\jcommon-1.0.21.jar;"%REPO%"\xml-apis-1.3.04.jar;"%REPO%"\scala-swing-2.10.4.jar;"%REPO%"\scalax-0.1.jar;"%REPO%"\scala-pickling_2.10-0.9.1.jar;"%REPO%"\quasiquotes_2.10-2.0.1.jar;"%REPO%"\spark-core_2.10-1.3.1.jar;"%REPO%"\chill_2.10-0.5.0.jar;"%REPO%"\kryo-2.21.jar;"%REPO%"\reflectasm-1.07-shaded.jar;"%REPO%"\minlog-1.2.jar;"%REPO%"\objenesis-1.2.jar;"%REPO%"\chill-java-0.5.0.jar;"%REPO%"\hadoop-client-2.2.0.jar;"%REPO%"\hadoop-common-2.2.0.jar;"%REPO%"\commons-math-2.1.jar;"%REPO%"\xmlenc-0.52.jar;"%REPO%"\jackson-core-asl-1.8.8.jar;"%REPO%"\jackson-mapper-asl-1.8.8.jar;"%REPO%"\avro-1.7.4.jar;"%REPO%"\protobuf-java-2.5.0.jar;"%REPO%"\hadoop-auth-2.2.0.jar;"%REPO%"\hadoop-hdfs-2.2.0.jar;"%REPO%"\jetty-util-6.1.26.jar;"%REPO%"\hadoop-mapreduce-client-app-2.2.0.jar;"%REPO%"\hadoop-mapreduce-client-common-2.2.0.jar;"%REPO%"\hadoop-yarn-client-2.2.0.jar;"%REPO%"\guice-3.0.jar;"%REPO%"\javax.inject-1.jar;"%REPO%"\aopalliance-1.0.jar;"%REPO%"\jersey-test-framework-grizzly2-1.9.jar;"%REPO%"\jersey-test-framework-core-1.9.jar;"%REPO%"\javax.servlet-api-3.0.1.jar;"%REPO%"\jersey-client-1.9.jar;"%REPO%"\jersey-grizzly2-1.9.jar;"%REPO%"\grizzly-http-2.1.2.jar;"%REPO%"\grizzly-framework-2.1.2.jar;"%REPO%"\gmbal-api-only-3.0.0-b023.jar;"%REPO%"\management-api-3.0.0-b012.jar;"%REPO%"\grizzly-http-server-2.1.2.jar;"%REPO%"\grizzly-rcm-2.1.2.jar;"%REPO%"\grizzly-http-servlet-2.1.2.jar;"%REPO%"\javax.servlet-3.1.jar;"%REPO%"\jersey-server-1.9.jar;"%REPO%"\jersey-core-1.9.jar;"%REPO%"\jersey-json-1.9.jar;"%REPO%"\jaxb-impl-2.2.3-1.jar;"%REPO%"\jaxb-api-2.2.2.jar;"%REPO%"\activation-1.1.jar;"%REPO%"\jackson-jaxrs-1.8.3.jar;"%REPO%"\jackson-xc-1.8.3.jar;"%REPO%"\jersey-guice-1.9.jar;"%REPO%"\hadoop-yarn-server-common-2.2.0.jar;"%REPO%"\hadoop-mapreduce-client-shuffle-2.2.0.jar;"%REPO%"\hadoop-yarn-api-2.2.0.jar;"%REPO%"\hadoop-mapreduce-client-core-2.2.0.jar;"%REPO%"\hadoop-yarn-common-2.2.0.jar;"%REPO%"\hadoop-mapreduce-client-jobclient-2.2.0.jar;"%REPO%"\hadoop-annotations-2.2.0.jar;"%REPO%"\spark-network-common_2.10-1.3.1.jar;"%REPO%"\spark-network-shuffle_2.10-1.3.1.jar;"%REPO%"\jets3t-0.7.1.jar;"%REPO%"\commons-codec-1.3.jar;"%REPO%"\commons-httpclient-3.1.jar;"%REPO%"\curator-recipes-2.4.0.jar;"%REPO%"\curator-framework-2.4.0.jar;"%REPO%"\curator-client-2.4.0.jar;"%REPO%"\zookeeper-3.4.5.jar;"%REPO%"\javax.servlet-3.0.0.v201112011016.jar;"%REPO%"\commons-lang3-3.3.2.jar;"%REPO%"\jsr305-1.3.9.jar;"%REPO%"\jul-to-slf4j-1.7.10.jar;"%REPO%"\jcl-over-slf4j-1.7.10.jar;"%REPO%"\slf4j-log4j12-1.7.10.jar;"%REPO%"\compress-lzf-1.0.0.jar;"%REPO%"\snappy-java-1.1.1.6.jar;"%REPO%"\lz4-1.2.0.jar;"%REPO%"\RoaringBitmap-0.4.5.jar;"%REPO%"\commons-net-2.2.jar;"%REPO%"\akka-remote_2.10-2.3.4-spark.jar;"%REPO%"\akka-actor_2.10-2.3.4-spark.jar;"%REPO%"\netty-3.8.0.Final.jar;"%REPO%"\protobuf-java-2.5.0-spark.jar;"%REPO%"\uncommons-maths-1.2.2a.jar;"%REPO%"\akka-slf4j_2.10-2.3.4-spark.jar;"%REPO%"\json4s-jackson_2.10-3.2.10.jar;"%REPO%"\json4s-core_2.10-3.2.10.jar;"%REPO%"\json4s-ast_2.10-3.2.10.jar;"%REPO%"\scalap-2.10.0.jar;"%REPO%"\mesos-0.21.0-shaded-protobuf.jar;"%REPO%"\netty-all-4.0.23.Final.jar;"%REPO%"\stream-2.7.0.jar;"%REPO%"\metrics-core-3.1.0.jar;"%REPO%"\metrics-jvm-3.1.0.jar;"%REPO%"\metrics-json-3.1.0.jar;"%REPO%"\metrics-graphite-3.1.0.jar;"%REPO%"\jackson-databind-2.4.4.jar;"%REPO%"\jackson-annotations-2.4.0.jar;"%REPO%"\jackson-core-2.4.4.jar;"%REPO%"\jackson-module-scala_2.10-2.4.4.jar;"%REPO%"\paranamer-2.6.jar;"%REPO%"\ivy-2.4.0.jar;"%REPO%"\oro-2.0.8.jar;"%REPO%"\tachyon-client-0.5.0.jar;"%REPO%"\tachyon-0.5.0.jar;"%REPO%"\pyrolite-2.0.1.jar;"%REPO%"\py4j-0.8.2.1.jar;"%REPO%"\unused-1.0.0.jar;"%REPO%"\spark-mllib_2.10-1.3.1.jar;"%REPO%"\spark-streaming_2.10-1.3.1.jar;"%REPO%"\spark-sql_2.10-1.3.1.jar;"%REPO%"\spark-catalyst_2.10-1.3.1.jar;"%REPO%"\parquet-column-1.6.0rc3.jar;"%REPO%"\parquet-common-1.6.0rc3.jar;"%REPO%"\parquet-encoding-1.6.0rc3.jar;"%REPO%"\parquet-generator-1.6.0rc3.jar;"%REPO%"\parquet-hadoop-1.6.0rc3.jar;"%REPO%"\parquet-format-2.2.0-rc1.jar;"%REPO%"\parquet-jackson-1.6.0rc3.jar;"%REPO%"\jodd-core-3.6.3.jar;"%REPO%"\spark-graphx_2.10-1.3.1.jar;"%REPO%"\jblas-1.2.3.jar;"%REPO%"\maven-javadoc-plugin-2.10.3.jar;"%REPO%"\maven-core-2.2.1.jar;"%REPO%"\wagon-file-1.0-beta-6.jar;"%REPO%"\maven-plugin-parameter-documenter-2.2.1.jar;"%REPO%"\wagon-http-lightweight-1.0-beta-6.jar;"%REPO%"\wagon-http-shared-1.0-beta-6.jar;"%REPO%"\xercesMinimal-1.9.6.2.jar;"%REPO%"\nekohtml-1.9.6.2.jar;"%REPO%"\wagon-http-1.0-beta-6.jar;"%REPO%"\wagon-webdav-jackrabbit-1.0-beta-6.jar;"%REPO%"\jackrabbit-webdav-1.5.0.jar;"%REPO%"\jackrabbit-jcr-commons-1.5.0.jar;"%REPO%"\slf4j-nop-1.5.3.jar;"%REPO%"\slf4j-jdk14-1.5.6.jar;"%REPO%"\maven-profile-2.2.1.jar;"%REPO%"\maven-repository-metadata-2.2.1.jar;"%REPO%"\maven-error-diagnostics-2.2.1.jar;"%REPO%"\commons-cli-1.2.jar;"%REPO%"\wagon-ssh-external-1.0-beta-6.jar;"%REPO%"\wagon-ssh-common-1.0-beta-6.jar;"%REPO%"\maven-plugin-descriptor-2.2.1.jar;"%REPO%"\plexus-interactivity-api-1.0-alpha-4.jar;"%REPO%"\maven-monitor-2.2.1.jar;"%REPO%"\wagon-ssh-1.0-beta-6.jar;"%REPO%"\jsch-0.1.38.jar;"%REPO%"\classworlds-1.1.jar;"%REPO%"\plexus-sec-dispatcher-1.3.jar;"%REPO%"\plexus-cipher-1.4.jar;"%REPO%"\maven-project-2.2.1.jar;"%REPO%"\maven-plugin-registry-2.2.1.jar;"%REPO%"\plexus-interpolation-1.11.jar;"%REPO%"\maven-model-2.2.1.jar;"%REPO%"\maven-settings-2.2.1.jar;"%REPO%"\maven-plugin-api-2.2.1.jar;"%REPO%"\maven-artifact-2.2.1.jar;"%REPO%"\maven-artifact-manager-2.2.1.jar;"%REPO%"\backport-util-concurrent-3.1.jar;"%REPO%"\maven-toolchain-2.2.1.jar;"%REPO%"\maven-reporting-api-3.0.jar;"%REPO%"\maven-archiver-2.5.jar;"%REPO%"\maven-invoker-2.0.9.jar;"%REPO%"\maven-common-artifact-filters-1.3.jar;"%REPO%"\doxia-sink-api-1.4.jar;"%REPO%"\doxia-logging-api-1.4.jar;"%REPO%"\doxia-site-renderer-1.4.jar;"%REPO%"\doxia-core-1.4.jar;"%REPO%"\xercesImpl-2.9.1.jar;"%REPO%"\doxia-decoration-model-1.4.jar;"%REPO%"\doxia-module-xhtml-1.4.jar;"%REPO%"\doxia-module-fml-1.4.jar;"%REPO%"\plexus-i18n-1.0-beta-7.jar;"%REPO%"\plexus-velocity-1.1.7.jar;"%REPO%"\velocity-1.5.jar;"%REPO%"\velocity-tools-2.0.jar;"%REPO%"\commons-beanutils-1.7.0.jar;"%REPO%"\commons-chain-1.1.jar;"%REPO%"\commons-validator-1.3.1.jar;"%REPO%"\dom4j-1.1.jar;"%REPO%"\sslext-1.2-0.jar;"%REPO%"\struts-core-1.3.8.jar;"%REPO%"\struts-taglib-1.3.8.jar;"%REPO%"\struts-tiles-1.3.8.jar;"%REPO%"\commons-collections-3.2.1.jar;"%REPO%"\plexus-component-annotations-1.5.5.jar;"%REPO%"\wagon-provider-api-1.0-beta-6.jar;"%REPO%"\commons-lang-2.4.jar;"%REPO%"\commons-io-2.2.jar;"%REPO%"\httpclient-4.2.3.jar;"%REPO%"\httpcore-4.2.2.jar;"%REPO%"\commons-logging-1.1.1.jar;"%REPO%"\qdox-1.12.1.jar;"%REPO%"\plexus-container-default-1.0-alpha-9.jar;"%REPO%"\plexus-archiver-2.9.jar;"%REPO%"\plexus-io-2.4.jar;"%REPO%"\commons-compress-1.9.jar;"%REPO%"\plexus-utils-3.0.20.jar;"%REPO%"\dynaml-2.0-SNAPSHOT.jar
set EXTRA_JVM_ARGUMENTS=-Dlog4j.configuration=file:///home/mandar/Development/DynaML/conf/log4j.properties
goto endInit

@REM Reaching here means variables are defined and arguments have been captured
:endInit

%JAVACMD% %JAVA_OPTS% %EXTRA_JVM_ARGUMENTS% -classpath %CLASSPATH_PREFIX%;%CLASSPATH% -Dapp.name="DynaML" -Dapp.repo="%REPO%" -Dbasedir="%BASEDIR%" io.github.mandar2812.dynaml.DynaML %CMD_LINE_ARGS%
if ERRORLEVEL 1 goto error
goto end

:error
if "%OS%"=="Windows_NT" @endlocal
set ERROR_CODE=1

:end
@REM set local scope for the variables with windows NT shell
if "%OS%"=="Windows_NT" goto endNT

@REM For old DOS remove the set variables from ENV - we assume they were not set
@REM before we started - at least we don't leave any baggage around
set CMD_LINE_ARGS=
goto postExec

:endNT
@REM If error code is set to 1 then the endlocal was done already in :error.
if %ERROR_CODE% EQU 0 @endlocal


:postExec

if "%FORCE_EXIT_ON_ERROR%" == "on" (
  if %ERROR_CODE% NEQ 0 exit %ERROR_CODE%
)

exit /B %ERROR_CODE%
