import sbt._

lazy val commonSettings = Seq(
  name := "DynaML",
  organization := "io.github.mandar2812",
  version := "v1.4-beta.2",
  scalaVersion in ThisBuild := "2.11.7",
  mainClass in Compile := Some("io.github.mandar2812.dynaml.DynaML"),
  fork in run := true
)

resolvers in ThisBuild ++= Seq(
  "jzy3d-releases" at "http://maven.jzy3d.org/releases"
)

libraryDependencies += "org.scala-lang" % "scala-library" % scalaVersion.value % "compile"

libraryDependencies += "org.scala-lang" % "scala-reflect" % scalaVersion.value % "compile"

libraryDependencies += "org.scala-lang" % "scala-compiler" % scalaVersion.value % "compile"

libraryDependencies += "com.typesafe" % "config" % "1.2.1" % "compile"

libraryDependencies += "junit" % "junit" % "4.4"

libraryDependencies += "org.specs" % "specs" % "1.2.5" % "test"

libraryDependencies += "com.tinkerpop.gremlin" % "gremlin-java" % "2.6.0" % "compile"

libraryDependencies += "com.tinkerpop" % "frames" % "2.5.0" % "compile"

libraryDependencies += "org.scalanlp" % "breeze_2.11" % "0.11.2" % "compile"

libraryDependencies += "org.scalanlp" % "breeze-natives_2.11" % "0.11.2" % "compile"

libraryDependencies += "com.github.tototoshi" % "scala-csv_2.11" % "1.1.2" % "compile"

libraryDependencies += "log4j" % "log4j" % "1.2.17" % "compile"

libraryDependencies += "org.scala-lang" % "jline" % "2.11.0-M3" % "compile"

//libraryDependencies += "org.scalatest" % "scalatest_2.11" % "2.2.1" % "test"

libraryDependencies += "com.github.wookietreiber" % "scala-chart_2.11" % "0.4.2" % "compile"

libraryDependencies += "org.scalaforge" % "scalax" % "0.1" % "compile"

libraryDependencies += "org.scala-lang" % "scala-pickling_2.11" % "0.9.1" % "compile"

libraryDependencies += "org.apache.spark" % "spark-core_2.11" % "1.6.1" % "compile"

libraryDependencies += "org.apache.spark" % "spark-mllib_2.11" % "1.6.1" % "compile"

libraryDependencies += "com.quantifind" % "wisp_2.11" % "0.0.4" % "compile"

libraryDependencies += "org.jzy3d" % "jzy3d-api" % "0.9.1" % "compile"

lazy val DynaML = (project in file(".")).enablePlugins(JavaAppPackaging, BuildInfoPlugin)
  .settings(commonSettings: _*)
  .settings(
    buildInfoKeys := Seq[BuildInfoKey](name, version, scalaVersion, sbtVersion),
    buildInfoPackage := "io.github.mandar2812.dynaml.repl",
    buildInfoUsePackageAsPath := true
  )