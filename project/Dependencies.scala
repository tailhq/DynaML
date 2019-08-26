import sbt._

object Dependencies {

  val scala_major = 2.12

  val scala_minor = 8

  val scala = s"$scala_major.$scala_minor"

  //val crossScala = Seq("2.11.8", "2.12.4")

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

  val tfscala_version = "0.4.2-SNAPSHOT"

  private def process_flag(s: String) = if(s.toLowerCase == "true" || s == "1") true else false
  
  //Set to true if, building with Nvidia GPU support.
  val gpuFlag: Boolean = process_flag(Option(System.getProperty("gpu")).getOrElse("false"))

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
    "com.typesafe" % "config" % "1.2.1" % "compile",
    "junit" % "junit" % "4.11",
    "com.github.tototoshi" %% "scala-csv" % "1.3.5" % "compile",
    "org.scala-lang" % "jline" % "2.11.0-M3" % "compile",
    "org.scalaforge" % "scalax" % "0.1" % "compile",
    "org.scalaz" %% "scalaz-core" % "7.2.7",
    "org.scala-graph" %% "graph-core" % "1.12.5",
    "org.scalatest" %% "scalatest" % "3.0.1" % "test",
    "com.github.scopt" %% "scopt" % "3.5.0",
    "org.scalameta" %% "scalameta" % "2.0.1",
    "javax.ws.rs" % "javax.ws.rs-api" % "2.0-m10",
    "org.json4s" %% "json4s-jackson" % "3.6.5",
    "ws.unfiltered" %% "unfiltered-filter" % "0.9.1",
    "ws.unfiltered" %% "unfiltered-jetty" % "0.9.1",
    "org.apache.commons" % "commons-math3" % "3.4.1",
    "commons-io" % "commons-io" % "2.4",
    "com.github.nscala-time" %% "nscala-time" % "2.22.0",
    "jline" % "jline" % "2.13"
  )

  val apacheSparkDependency = Seq(
    "javax.servlet" % "javax.servlet-api" % "3.1.0" % "test",
    "org.apache.spark" %% "spark-core" % "2.4.3",
    "org.apache.spark" %% "spark-mllib" % "2.4.3",
    "com.fasterxml.jackson.core" % "jackson-databind" % "2.9.9.3",
    "com.fasterxml.jackson.module" %% "jackson-module-scala" % "2.9.9")
    .map(_.withExclusions(
      Vector(
        //"org.slf4j" % "slf4j-log4j12", 
        "org.slf4j" % "jul-to-slf4j",
        "org.slf4j" % "jcl-over-slf4j",
        "log4j" % "log4j",
        "org.scalanlp" %% "breeze", 
        "javax.ws.rs" %% "javax.ws.rs-api"))
    )
    

  val loggingDependency = Seq("log4j" % "log4j" % "1.2.17")

  val linearAlgebraDependencies = Seq(
    "org.typelevel" %% "spire" % "0.14.1",
    "org.scalanlp" %% "breeze" % "1.0" % "compile",
    "org.scalanlp" %% "breeze-natives" % "1.0" % "compile"
  ).map(_.withExclusions(Vector("org.spire-math" %% "spire")))

  val chartsDependencies = Seq(
    "com.github.wookietreiber" %% "scala-chart" % "0.5.1" % "compile",
    "org.jzy3d" % "jzy3d-api" % "1.0.2" % "compile",
    "com.cibo" %% "evilplot" % "0.7.0" 
  )

  val replDependency = Seq(
    "com.lihaoyi" %% "ammonite-repl" % "1.6.9" cross CrossVersion.full,
    "com.lihaoyi" %% "ammonite-sshd" % "1.6.9" cross CrossVersion.full
  )

  val commons_io = Seq("commons-io" % "commons-io" % "2.6")

  val openMLDependency = Seq("org.openml" % "apiconnector" % "1.0.11")

  val tinkerpopDependency = Seq(
    "com.tinkerpop.gremlin" % "gremlin-java" % "2.6.0" % "compile",
    "com.tinkerpop" % "frames" % "2.5.0" % "compile"
  )

  val rejinDependency = Seq(
    "org.renjin" % "renjin-script-engine" % "0.9.2726"
  )

  val rPackages = Seq(
    "org.renjin.cran" % "plyr" % "1.8.4-b107",
    "org.renjin.cran" % "abc" % "2.1-b295",
    "org.renjin.cran" % "ggplot2" % "2.2.1-b113"
  )

  val dynaServeDependencies = Seq(
    "com.typesafe.akka" %% "akka-actor" % "2.5.16",
    "com.typesafe.akka" %% "akka-stream" % "2.5.16",
    "com.typesafe.akka" %% "akka-testkit" % "2.5.16",
    "com.typesafe.akka" %% "akka-http" % "10.0.9",
    "com.typesafe.akka" %% "akka-http-spray-json" % "10.0.9",
    "com.typesafe.akka" %% "akka-http-testkit" % "10.0.9"
  )

  val imageDependencies = Seq(
    "com.sksamuel.scrimage" %% "scrimage-core" % "2.1.8",
    "com.sksamuel.scrimage" %% "scrimage-io-extra" % "2.1.8",
    "com.sksamuel.scrimage" %% "scrimage-filters" % "2.1.8"
  )

  val dataFormatDependencies = Seq(
    "com.diffplug.matsim" % "matfilerw" % "3.0.1"
  )

  val tensorflowDependency = Seq(
    "org.platanios" %% "tensorflow" % tfscala_version classifier tensorflow_classifier,
    "org.platanios" %% "tensorflow-data" % tfscala_version
  ).map(_.withExclusions(Vector("org.typelevel" %% "spire")))

  val scalaStan = Seq(
    "com.cibo" %% "scalastan" % "0.9.0"
  )

  val coursier_deps = Seq(
    "io.get-coursier" %% "coursier" % "2.0.0-RC2-6",
    "io.get-coursier" % "interface" % "0.0.10"
  )

  val almond = Seq(
    "sh.almond" %% "scala-interpreter" % "0.7.0" cross CrossVersion.full,
    "sh.almond" %% "scala-kernel-api" % "0.7.0" cross CrossVersion.full,
    "sh.almond" %% "kernel" % "0.7.0",
    "com.github.alexarchambault" %% "case-app" % "2.0.0-M9"
  )
}
