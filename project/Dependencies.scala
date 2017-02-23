import sbt._

object Dependencies {

  val scala = "2.11.8"

  val crossScalaVersions = Seq(
  "2.10.4", "2.10.5", "2.10.6", "2.11.3",
  "2.11.4", "2.11.5", "2.11.6", "2.11.7", "2.11.8"
  )
  
  val platform = {
    // Determine platform name using code similar to javacpp
    // com.googlecode.javacpp.Loader.java line 60-84
    val jvmName = System.getProperty("java.vm.name").toLowerCase
    var osName = System.getProperty("os.name").toLowerCase
    var osArch = System.getProperty("os.arch").toLowerCase
    if (jvmName.startsWith("dalvik") && osName.startsWith("linux")) {
      osName = "android"
    } else if (jvmName.startsWith("robovm") && osName.startsWith("darwin")) {
      osName = "ios"
      osArch = "arm"
    } else if (osName.startsWith("mac os x")) {
      osName = "macosx"
    } else {
      val spaceIndex = osName.indexOf(' ')
      if (spaceIndex > 0) {
        osName = osName.substring(0, spaceIndex)
      }
    }
    if (osArch.equals("i386") || osArch.equals("i486") || osArch.equals("i586") || osArch.equals("i686")) {
      osArch = "x86"
    } else if (osArch.equals("amd64") || osArch.equals("x86-64") || osArch.equals("x64")) {
      osArch = "x86_64"
    } else if (osArch.startsWith("arm")) {
      osArch = "arm"
    }
    val platformName = osName + "-" + osArch
    println("platform: " + platformName)
    platformName
  }


  val baseDependencies = Seq(
    "org.scala-lang" % "scala-compiler" % scala % "compile",
    "org.scala-lang" % "scala-library" % scala % "compile",
    "org.scala-lang" % "scala-reflect" % scala % "compile",
    "com.typesafe" % "config" % "1.2.1" % "compile",
    "junit" % "junit" % "4.11",
    "com.github.tototoshi" % "scala-csv_2.11" % "1.1.2" % "compile",
    "org.scala-lang" % "jline" % "2.11.0-M3" % "compile",
    "org.scalaforge" % "scalax" % "0.1" % "compile",
    "org.scala-lang" % "scala-pickling_2.11" % "0.9.1" % "compile",
    "org.scalaz" %% "scalaz-core" % "7.2.0",
    "org.scalactic" % "scalactic_2.11" % "2.2.6",
    "org.scala-graph" %% "graph-core" % "1.11.3",
    "org.scalatest" % "scalatest_2.11" % "2.2.6" % "test",
    "com.typesafe.akka" %% "akka-stream" % "2.4.6",
    "com.github.scopt" % "scopt_2.11" % "3.5.0",
    "com.nativelibs4java" % "scalaxy-streams_2.11" % "0.3.4" % "provided",
    "com.diffplug.matsim" % "matfilerw" % "3.0.0",
    "org.scalameta" % "scalameta_2.11" % "1.4.0",
    "com.signalcollect" % "signal-collect_2.11" % "8.0.6",
    "com.signalcollect" % "triplerush_2.11" % "9.0.0",
    "com.chuusai" %% "shapeless" % "2.3.2",
    compilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full),
    "com.github.julien-truffaut" %% "monocle-core" % "1.4.0-M2",    
    "io.github.nicolasstucki" %% "multisets" % "0.4",
    "net.java.dev.jna" % "jna" % "4.2.2",
    "org.apache.commons" % "commons-math3" % "3.6.1",
    "org.scala-lang.modules" %% "scala-parser-combinators" % "1.0.5",
    "org.scala-lang" % "scala-compiler" % scala,
    "org.spire-math" %% "spire" % "0.13.0"
  )

  val apacheSparkDependency = Seq(
    "javax.servlet" % "javax.servlet-api" % "3.1.0" % "test",
    "org.apache.spark" % "spark-core_2.11" % "2.0.0" % "compile",
    "org.apache.spark" % "spark-mllib_2.11" % "2.0.0" % "compile"
  )

  val loggingDependency = Seq("log4j" % "log4j" % "1.2.17" % "compile")

  val linearAlgebraDependencies = Seq(
    "org.scalanlp" % "breeze_2.11" % "0.12" % "compile",
    "org.scalanlp" % "breeze-natives_2.11" % "0.12" % "compile",
    "org.la4j" % "la4j" % "0.6.0" % "compile")

  val chartsDependencies = Seq(
    "com.github.wookietreiber" % "scala-chart_2.11" % "0.4.2" % "compile",
    "org.jzy3d" % "jzy3d-api" % "0.9.1" % "compile",
    "com.quantifind" % "wisp_2.11" % "0.0.4" % "compile",
    "co.theasi" % "plotly_2.11" % "0.1",
    "org.vegas-viz" % "vegas_2.11" % "0.3.6"
  )

  val replDependency = Seq(
    "com.lihaoyi" % "ammonite-repl_2.11.8" % "0.8.1",
    "com.lihaoyi" % "ammonite" % "0.8.1" cross CrossVersion.full
  )

  val openMLDependency = Seq("org.openml" % "apiconnector" % "1.0.11")

  val tinkerpopDependency = Seq(
    "com.tinkerpop.gremlin" % "gremlin-java" % "2.6.0" % "compile",
    "com.tinkerpop" % "frames" % "2.5.0" % "compile"
  )

  val rejinDependency = Seq(
    "org.renjin" % "renjin-script-engine" % "0.8.2297"
  )

  val rPackages = Seq(
    "org.renjin.cran" % "plyr" % "1.8.3-renjin-10",
    "org.renjin.cran" % "abc" % "2.1-b274"
  )

  val cppCompatDependencies = Seq(
    "com.nativelibs4java" % "jnaerator" % "0.12",
    "com.nativelibs4java" % "bridj" % "0.7.0",
    "org.bytedeco" % "javacpp" % "1.3",
    "org.bytedeco.javacpp-presets" % "tensorflow" % "0.11.0-1.3"
  )

  val notebookInterfaceDependency = Seq()
}
