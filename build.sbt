import sbt._
import java.io.File
import Dependencies._
import sbtbuildinfo.BuildInfoPlugin.autoImport._


val mainVersion = "v2.0-SNAPSHOT"
maintainer := "Mandar Chandorkar <mandar2812@gmail.com>"
packageSummary := "Scala Library/REPL for Machine Learning Research"
packageDescription := "DynaML is a Scala & JVM Machine Learning toolbox for research, education & industry."

val dataDirectory = settingKey[File]("The directory holding the data files for running example scripts")

val baseSettings = Seq(
  organization := "io.github.transcendent-ai-labs",
  scalaVersion in ThisBuild := scala,
  crossScalaVersions in ThisBuild := crossScala,
  resolvers in ThisBuild ++= Seq(
    "jzy3d-releases" at "http://maven.jzy3d.org/releases",
    "Scalaz Bintray Repo" at "http://dl.bintray.com/scalaz/releases",
    "BeDataDriven" at "https://nexus.bedatadriven.com/content/groups/public",
    Resolver.sonatypeRepo("public"),
    Resolver.sonatypeRepo("snapshots"),
    Resolver.typesafeIvyRepo("releases")),
  publishTo := sonatypePublishTo.value,
  useGpg := true,
  publishConfiguration := publishConfiguration.value.withOverwrite(true),
  publishLocalConfiguration := publishLocalConfiguration.value.withOverwrite(true)
)

lazy val commonSettings = Seq(
  libraryDependencies ++= (linearAlgebraDependencies ++ baseDependencies ++ loggingDependency ++ apacheSparkDependency)
)

lazy val settingsCore = Seq(
  libraryDependencies ++= (
      chartsDependencies ++
        tinkerpopDependency ++
        openMLDependency ++
        rejinDependency ++ rPackages ++
        imageDependencies ++
        dataFormatDependencies ++
        tensorflowDependency ++
        replDependency)
)

lazy val pipes = (project in file("dynaml-pipes")).settings(baseSettings:_*)
  .settings(commonSettings:_*)
  .settings(
    name := "dynaml-pipes",
    version := mainVersion
  )

lazy val core = (project in file("dynaml-core")).settings(baseSettings)
  .settings(commonSettings:_*)
  .settings(settingsCore:_*)
  .enablePlugins(JavaAppPackaging, BuildInfoPlugin)
  .dependsOn(pipes)
  .settings(
    name := "dynaml-core",
    version := mainVersion
  )

lazy val examples = (project in file("dynaml-examples"))
  .settings(baseSettings:_*)
  .settings(commonSettings:_*)
  .settings(
    name := "dynaml-examples",
    version := mainVersion
  ).dependsOn(pipes, core)


lazy val repl = (project in file("dynaml-repl")).enablePlugins(BuildInfoPlugin)
  .settings(baseSettings:_*)
  .settings(
    name := "dynaml-repl",
    version := mainVersion,
    buildInfoKeys := Seq[BuildInfoKey](name, version, scalaVersion, sbtVersion),
    buildInfoPackage := "io.github.mandar2812.dynaml.repl",
    buildInfoUsePackageAsPath := true,
    libraryDependencies ++= (baseDependencies ++ replDependency ++ commons_io)
  )

lazy val DynaML = (project in file(".")).enablePlugins(JavaAppPackaging, BuildInfoPlugin, sbtdocker.DockerPlugin)
  .settings(baseSettings:_*)
  .dependsOn(core, examples, pipes, repl)
  .settings(
    libraryDependencies ++= dynaServeDependencies,
    name := "DynaML",
    version := mainVersion,
    fork in run := true,
    fork in test := true,
    mainClass in Compile := Some("io.github.mandar2812.dynaml.DynaML"),
    buildInfoKeys := Seq[BuildInfoKey](name, version, scalaVersion, sbtVersion),
    buildInfoPackage := "io.github.mandar2812.dynaml.repl",
    buildInfoUsePackageAsPath := true,
    dataDirectory := new File("data"),
    mappings in Universal ++= dataDirectory.value.listFiles().toSeq.map(p => p -> s"data/${p.getName}"),
    mappings in Universal ++= Seq(
      {
        //Initialization script for the DynaML REPL
        val init = (resourceDirectory in Compile).value / "DynaMLInit.scala"
        init -> "conf/DynaMLInit.scala"
      },
      {
        val banner = (resourceDirectory in Compile).value / "dynamlBanner.txt"
        banner -> "conf/banner.txt"
      }
    ),
    javaOptions in test ++= Seq(
      "-Dlog4j.debug=true",
      "-Dlog4j.configuration=log4j.properties"),
    javaOptions in Universal ++= Seq(
      // -J params will be added as jvm parameters
      "-J-Xmx4096m",
      "-J-Xms64m", 
      "-XX:HeapBaseMinAddress=32G"
    ),
    scalacOptions in Universal ++= Seq("-Xlog-implicits"),
    initialCommands in console := """io.github.mandar2812.dynaml.DynaML.main(Array())""",
    dockerfile in docker := {
      val appDir: File = stage.value
      val targetDir = "/app"

      new Dockerfile {
        from("openjdk:8-jre")
        entryPoint(s"$targetDir/bin/${executableScriptName.value}")
        copy(appDir, targetDir, chown = "daemon:daemon")
      }
    },
    imageNames in docker := Seq(
      // Sets the latest tag
      ImageName(s"mandar2812/${name.value.toLowerCase}:latest"),

      // Sets a name with a tag that contains the project version
      ImageName(
        namespace = Some("mandar2812"),
        repository = name.value.toLowerCase,
        tag = Some(version.value)
      )
    )
  ).aggregate(core, pipes, examples, repl).settings(
    aggregate in publishM2 := true,
    aggregate in update := false)

