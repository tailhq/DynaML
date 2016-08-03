name := "StateSpaceModels"

version := "1.0"

scalaVersion := "2.11.8"

resolvers ++= Seq(
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/",
  "Sonatype Snapshots" at "http://oss.sonatype.org/content/repositories/snapshots",
  Resolver.sonatypeRepo("public")
)

libraryDependencies  ++= Seq(
  "com.typesafe.akka" %% "akka-stream" % "2.4.6",
  "org.scalanlp" %% "breeze" % "0.10",
  "org.scalatest" %% "scalatest" % "2.2.4" % "test",
  "com.github.fommil.netlib" % "all" % "1.1.2"
)
