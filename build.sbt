import sbt._
import java.io.File
import Dependencies._
import sbtbuildinfo.BuildInfoPlugin.autoImport._

maintainer := "Mandar Chandorkar <mandar2812@gmail.com>"

packageSummary := "Scala Library/REPL for Machine Learning Research"

packageDescription := "DynaML is a scala library/repl for implementing and working with "+
  "general Machine Learning models.\n\nThe aim is to build a robust set of abstract classes and interfaces, "+
  "which can be extended easily to implement advanced models for small and large scale applications.\n\n"+
  "But the library can also be used as an educational/research tool for data analysis."

val mainVersion = "v1.4.2-beta.4"

val dataDirectory = settingKey[File]("The directory holding the data files for running example scripts")

val baseSettings = Seq(
  organization := "io.github.mandar2812",
  scalaVersion in ThisBuild := scala,
  resolvers in ThisBuild ++= Seq(
    "jzy3d-releases" at "http://maven.jzy3d.org/releases",
    "Scalaz Bintray Repo" at "http://dl.bintray.com/scalaz/releases",
    "BeDataDriven" at "https://nexus.bedatadriven.com/content/groups/public",
    Resolver.sonatypeRepo("public"))
)

lazy val commonSettings = Seq(
  libraryDependencies ++= (
    baseDependencies ++ apacheSparkDependency ++
      replDependency ++ loggingDependency ++
      linearAlgebraDependencies ++ chartsDependencies ++
      tinkerpopDependency ++ notebookInterfaceDependency ++
      openMLDependency ++ rejinDependency ++
      rPackages ++ cppCompatDependencies ++
      imageDependencies),

  scalacOptions ++= Seq("-optimise", "-Yclosure-elim", "-Yinline")
)

lazy val pipes = (project in file("dynaml-pipes")).settings(baseSettings:_*)
  .settings(commonSettings:_*)
  .settings(
    name := "dynaml-pipes",
    version := mainVersion
  )

lazy val core = (project in file("dynaml-core")).settings(baseSettings)
  .settings(commonSettings:_*)
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

lazy val DynaML = (project in file(".")).enablePlugins(JavaAppPackaging, BuildInfoPlugin)
  .settings(baseSettings:_*)
  .dependsOn(core, examples, pipes)
  .settings(
    name := "DynaML",
    version := mainVersion,
    fork in run := true,
    mainClass in Compile := Some("io.github.mandar2812.dynaml.DynaML"),
    buildInfoKeys := Seq[BuildInfoKey](name, version, scalaVersion, sbtVersion),
    buildInfoPackage := "io.github.mandar2812.dynaml.repl",
    buildInfoUsePackageAsPath := true,
    mappings in Universal ++= Seq({
      // we are using the reference.conf as default application.conf
      // the user can override settings here
      val init = (resourceDirectory in Compile).value / "DynaMLInit.scala"
      init -> "conf/DynaMLInit.scala"
    }, {
      val banner = (resourceDirectory in Compile).value / "dynamlBanner.txt"
      banner -> "conf/banner.txt"
    }),
    javaOptions in Universal ++= Seq(
      // -J params will be added as jvm parameters
      "-J-Xmx2048m",
      "-J-Xms64m"
    ),
    dataDirectory := new File("data/"),
    initialCommands in console := """io.github.mandar2812.dynaml.DynaML.main(Array())"""
  ).aggregate(core, pipes, examples).settings(
    aggregate in publishM2 := true)

