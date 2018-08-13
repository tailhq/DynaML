import sbt._

object Dependencies {

  val scala = "2.11.8"

  val platform: String = {
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

  val tfscala_version = "0.2.4"

  //Set to true if, building with Nvidia GPU support.
  val gpuFlag: Boolean = false

  //Set to false if using self compiled tensorflow library
  val packagedTFFlag: Boolean = true

  val tensorflow_classifier: String = {
    val platform_splits = platform.split("-")
    val (os, arch) = (platform_splits.head, platform_splits.last)

    val tf_c =
      if (os.contains("macosx")) "darwin-cpu-"+arch
      else if(os.contains("linux")) {
        if(gpuFlag) "linux-gpu-"+arch else "linux-cpu-"+arch
      } else ""
    println("Tensorflow-Scala Classifier: "+tf_c)
    tf_c
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
    "org.scalatest" % "scalatest_2.11" % "3.0.1" % "test",
    "com.github.scopt" % "scopt_2.11" % "3.5.0",
    "com.nativelibs4java" % "scalaxy-streams_2.11" % "0.3.4" % "provided",
    "org.scalameta" % "scalameta_2.11" % "2.0.1",
    "javax.ws.rs" % "javax.ws.rs-api" % "2.0-m10"
  )

  val apacheSparkDependency = Seq(
    "javax.servlet" % "javax.servlet-api" % "3.1.0" % "test",
    "org.apache.spark" % "spark-core_2.11" % "2.2.0" % "compile",
    "org.apache.spark" % "spark-mllib_2.11" % "2.2.0" % "compile")
    .map(_.exclude("org.slf4j", "slf4j-log4j12"))
    .map(_.exclude("org.scalanlp", "breeze_2.11"))
    .map(_.exclude("javax.ws.rs" , "javax.ws.rs-api"))

  val loggingDependency = Seq("log4j" % "log4j" % "1.2.17" % "compile")

  val linearAlgebraDependencies = Seq(
    "org.typelevel" % "spire_2.11" % "0.14.1",
    "org.scalanlp" % "breeze_2.11" % "0.13.2" % "compile",
    "org.scalanlp" % "breeze-natives_2.11" % "0.13.2" % "compile",
    "org.la4j" % "la4j" % "0.6.0" % "compile",
    "com.github.vagmcs" % "optimus_2.11" % "2.0.0")
    .map(_.exclude("org.spire-math", "spire_2.11"))

  val chartsDependencies = Seq(
    "com.github.wookietreiber" % "scala-chart_2.11" % "0.4.2" % "compile",
    "org.jzy3d" % "jzy3d-api" % "1.0.2" % "compile",
    "com.quantifind" % "wisp_2.11" % "0.0.4" % "compile",
    "co.theasi" % "plotly_2.11" % "0.1",
    ("org.vegas-viz" % "vegas_2.11" % "0.3.11").exclude("io.circe", "circe-parser")
  )

  val replDependency = Seq(
    "com.lihaoyi" % "ammonite-repl_2.11.8" % "1.1.0",
    "com.lihaoyi" % "ammonite-sshd_2.11.8" % "1.1.0"
  )

  val commons_io = Seq("commons-io" % "commons-io" % "2.6")

  val openMLDependency = Seq("org.openml" % "apiconnector" % "1.0.11")

  val tinkerpopDependency = Seq(
    "com.tinkerpop.gremlin" % "gremlin-java" % "2.6.0" % "compile",
    "com.tinkerpop" % "frames" % "2.5.0" % "compile"
  )

  val rejinDependency = Seq(
    "org.renjin" % "renjin-script-engine" % "0.9.2643"
  )

  val rPackages = Seq(
    "org.renjin.cran" % "plyr" % "1.8.4-b82",
    "org.renjin.cran" % "abc" % "2.1-b294",
    "org.renjin.cran" % "ggplot2" % "2.2.1-b112"
  )

  val dynaServeDependencies = Seq(
    "com.typesafe.akka" % "akka-actor_2.11" % "2.5.3",
    "com.typesafe.akka" % "akka-stream_2.11" % "2.5.3",
    "com.typesafe.akka" % "akka-testkit_2.11" % "2.5.3",
    "com.typesafe.akka" % "akka-http_2.11" % "10.0.9",
    "com.typesafe.akka" % "akka-http-spray-json_2.11" % "10.0.9",
    "com.typesafe.akka" % "akka-http-testkit_2.11" % "10.0.9"
  )

  val imageDependencies = Seq(
    "com.sksamuel.scrimage" % "scrimage-core_2.11" % "2.1.8",
    "com.sksamuel.scrimage" % "scrimage-io-extra_2.11" % "2.1.8",
    "com.sksamuel.scrimage" % "scrimage-filters_2.11" % "2.1.8"
  )

  val dataFormatDependencies = Seq(
    "info.folone" % "poi-scala_2.11" % "0.18",
    "com.diffplug.matsim" % "matfilerw" % "3.0.0"
  )

  val tensorflowDependency = Seq(
    "org.platanios" % "tensorflow_2.11" % tfscala_version classifier tensorflow_classifier,
    "org.platanios" % "tensorflow-data_2.11" % tfscala_version
  ).map(_.exclude("org.typelevel", "spire_2.11"))
}
