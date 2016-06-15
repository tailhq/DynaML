import sbt._
import java.io.File

import sbtbuildinfo.BuildInfoPlugin.autoImport._

maintainer := "Mandar Chandorkar <mandar2812@gmail.com>"

packageSummary := "Scala Library/REPL for Machine Learning Research"

packageDescription := "DynaML is a scala library/repl for implementing and working with "+
  "general Machine Learning models.\n\nThe aim is to build a robust set of abstract classes and interfaces, "+
  "which can be extended easily to implement advanced models for small and large scale applications.\n\n"+
  "But the library can also be used as an educational/research tool for data analysis."

val mainVersion = "v1.4-beta.8"

val dataDirectory = settingKey[File]("The directory holding the data files for running example scripts")

val baseSettings = Seq(
  organization := "io.github.mandar2812",
  scalaVersion in ThisBuild := "2.11.7",
  resolvers in ThisBuild ++= Seq("jzy3d-releases" at "http://maven.jzy3d.org/releases")
)

lazy val commonSettings = Seq(
  libraryDependencies ++= Seq(
    "org.scala-lang" % "scala-compiler" % scalaVersion.value % "compile",
    "org.scala-lang" % "scala-library" % scalaVersion.value % "compile",
    "org.scala-lang" % "scala-reflect" % scalaVersion.value % "compile",
    "com.typesafe" % "config" % "1.2.1" % "compile",
    "junit" % "junit" % "4.4",
    "org.specs" % "specs" % "1.2.5" % "test",
    "com.tinkerpop.gremlin" % "gremlin-java" % "2.6.0" % "compile",
    "com.tinkerpop" % "frames" % "2.5.0" % "compile",
    "org.scalanlp" % "breeze_2.11" % "0.11.2" % "compile",
    "org.scalanlp" % "breeze-natives_2.11" % "0.11.2" % "compile",
    "com.github.tototoshi" % "scala-csv_2.11" % "1.1.2" % "compile",
    "log4j" % "log4j" % "1.2.17" % "compile",
    "org.scala-lang" % "jline" % "2.11.0-M3" % "compile",
    "com.github.wookietreiber" % "scala-chart_2.11" % "0.4.2" % "compile",
    "org.scalaforge" % "scalax" % "0.1" % "compile",
    "org.scala-lang" % "scala-pickling_2.11" % "0.9.1" % "compile",
    "org.apache.spark" % "spark-core_2.11" % "1.6.1" % "compile",
    "org.apache.spark" % "spark-mllib_2.11" % "1.6.1" % "compile",
    "com.quantifind" % "wisp_2.11" % "0.0.4" % "compile",
    "org.jzy3d" % "jzy3d-api" % "0.9.1" % "compile",
    "com.lihaoyi" % "ammonite-repl_2.11.7" % "0.5.8"
  )
)

lazy val pipes = (project in file("dynaml-pipes")).settings(baseSettings:_*)
  .settings(
    name := "dynaml-pipes",
    version := "1.0"
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
    version := "1.0"
  ).dependsOn(pipes, core)

lazy val DynaML = (project in file(".")).enablePlugins(JavaAppPackaging, BuildInfoPlugin)
  .settings(baseSettings:_*)
  .settings(commonSettings: _*)
  .dependsOn(core, examples, pipes)
  .settings(
    //aggregate in update := false,
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
    initialCommands in console := """io.github.mandar2812.dynaml.DynaML.run(banner="""" +
      target.value.getPath + """/universal/stage/conf/banner.txt");"""

)

