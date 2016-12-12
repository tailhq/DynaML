import sbt._

object Dependencies {

  val scala = "2.11.8"

  val baseDependencies = Seq(
    "org.scala-lang" % "scala-compiler" % scala % "compile",
    "org.scala-lang" % "scala-library" % scala % "compile",
    "org.scala-lang" % "scala-reflect" % scala % "compile",
    "com.typesafe" % "config" % "1.2.1" % "compile",
    "junit" % "junit" % "4.4",
    "com.github.tototoshi" % "scala-csv_2.11" % "1.1.2" % "compile",
    "org.scala-lang" % "jline" % "2.11.0-M3" % "compile",
    "org.scalaforge" % "scalax" % "0.1" % "compile",
    "org.scala-lang" % "scala-pickling_2.11" % "0.9.1" % "compile",
    "org.scalaz" %% "scalaz-core" % "7.2.0",
    "org.scalactic" % "scalactic_2.11" % "2.2.6",
    "org.scala-graph" %% "graph-core" % "1.11.3",
    "org.scalatest" % "scalatest_2.11" % "2.2.6" % "test",
    "com.typesafe.akka" %% "akka-stream" % "2.4.6",
    "com.github.scopt" % "scopt_2.11" % "3.5.0"
  )

  val apacheSpark = Seq(
    "javax.servlet" % "javax.servlet-api" % "3.1.0" % "test",
    "org.apache.spark" % "spark-core_2.11" % "2.0.0" % "compile",
    "org.apache.spark" % "spark-mllib_2.11" % "2.0.0" % "compile"
  )

  val loggingDependency = Seq("log4j" % "log4j" % "1.2.17" % "compile")

  val linearAlgebraDependencies = Seq(
    "org.scalanlp" % "breeze_2.11" % "0.11.2" % "compile",
    "org.scalanlp" % "breeze-natives_2.11" % "0.11.2" % "compile")

  val chartsDependencies = Seq(
    "com.github.wookietreiber" % "scala-chart_2.11" % "0.4.2" % "compile",
    "org.jzy3d" % "jzy3d-api" % "0.9.1" % "compile",
    "com.quantifind" % "wisp_2.11" % "0.0.4" % "compile",
    "co.theasi" % "plotly_2.11" % "0.1"
  )

  val replDependency = Seq(
    "com.lihaoyi" % "ammonite-repl_2.11.8" % "0.8.1"
  )

  val openMLDependency = Seq("org.openml" % "apiconnector" % "1.0.11")

  val tinkerpopDependency = Seq(
    "com.tinkerpop.gremlin" % "gremlin-java" % "2.6.0" % "compile",
    "com.tinkerpop" % "frames" % "2.5.0" % "compile"
  )

  val rejinDependency = Seq(
    "org.renjin" % "renjin-script-engine" % "0.8.2297"
  )

  val notebookInterfaceDependency = Seq()
}
