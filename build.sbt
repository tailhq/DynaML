import sbt._
import java.io.File
import Dependencies._
import sbtbuildinfo.BuildInfoPlugin.autoImport._

maintainer := "Mandar Chandorkar <mandar2812@gmail.com>"

packageSummary := "Scala Library/REPL for Machine Learning Research"

packageDescription := "DynaML is a Scala environment for conducting research and education in Machine Learning. DynaML comes packaged with a powerful library of classes for various predictive models and a Scala REPL where one can not only build custom models but also play around with data work-flows. It can also be used as an educational/research tool for data analysis."

val mainVersion = "v1.5.3-beta.3"

val dataDirectory = settingKey[File]("The directory holding the data files for running example scripts")

val baseSettings = Seq(
  organization := "io.github.mandar2812",
  scalaVersion in ThisBuild := scala,
  resolvers in ThisBuild ++= Seq(
    "jzy3d-releases" at "http://maven.jzy3d.org/releases",
    "Scalaz Bintray Repo" at "http://dl.bintray.com/scalaz/releases",
    "BeDataDriven" at "https://nexus.bedatadriven.com/content/groups/public",
    Resolver.sonatypeRepo("public"),
    Resolver.sonatypeRepo("snapshots")),
  scalacOptions ++= Seq("-optimise", "-Yclosure-elim", "-Yinline", "-target:jvm-1.8")
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

lazy val notebook = (project in file("dynaml-notebook")).enablePlugins(JavaServerAppPackaging)
  .settings(baseSettings:_*)
  .settings(
    name := "dynaml-notebook",
    version := mainVersion,
    libraryDependencies ++= notebookInterfaceDependency
  ).dependsOn(core, examples, pipes, repl)
  .settings(
    mappings in Universal ++= Seq({
      // we are using the reference.conf as default application.conf
      // the user can override settings here
      val init = (resourceDirectory in Compile).value / "DynaMLInit.scala"
      init -> "conf/DynaMLInit.scala"
    }, {
      val banner = (resourceDirectory in Compile).value / "dynamlBanner.txt"
      banner -> "conf/banner.txt"
    }, {
      val zeppelin_env = (resourceDirectory in Compile).value / "zeppelin-site.xml"
      zeppelin_env -> "conf/zeppelin-site.xml"
    }, {
      val zeppelin_shiro = (resourceDirectory in Compile).value / "shiro.ini.template"
      zeppelin_shiro -> "conf/shiro.ini"
    }, {
      val zeppelinConf = (resourceDirectory in Compile).value / "interpreter-setting.json"
      zeppelinConf -> "lib/interpreter-setting.json"
    }, {
      val common = (resourceDirectory in Compile).value / "common.sh"
      common -> "bin/common.sh"
    }, {
      val intp = (resourceDirectory in Compile).value / "interpreter.sh"
      intp -> "bin/interpreter.sh"
    })
  )

lazy val DynaML = (project in file(".")).enablePlugins(JavaAppPackaging, BuildInfoPlugin, sbtdocker.DockerPlugin)
  .settings(baseSettings:_*)
  .dependsOn(core, examples, pipes, repl)
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
    }, {
      val zeppelin_env = (resourceDirectory in Compile).value / "zeppelin-site.xml"
      zeppelin_env -> "conf/zeppelin-site.xml"
    }, {
      val zeppelin_shiro = (resourceDirectory in Compile).value / "shiro.ini.template"
      zeppelin_shiro -> "conf/shiro.ini"
    }, {
      val zeppelinConf = (resourceDirectory in Compile).value / "interpreter-setting.json"
      zeppelinConf -> "lib/interpreter-setting.json"
    }, {
      val common = (resourceDirectory in Compile).value / "common.sh"
      common -> "bin/common.sh"
    }, {
      val intp = (resourceDirectory in Compile).value / "interpreter.sh"
      intp -> "bin/interpreter.sh"
    }),
    javaOptions in Universal ++= Seq(
      // -J params will be added as jvm parameters
      "-J-Xmx2048m",
      "-J-Xms64m"
    ),
    dataDirectory := new File("data/"),
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
      ImageName(s"${organization.value}/${name.value.toLowerCase}:latest"),

      // Sets a name with a tag that contains the project version
      ImageName(
        namespace = Some(organization.value),
        repository = name.value.toLowerCase,
        tag = Some(version.value)
      )
    )
  ).aggregate(core, pipes, examples, repl).settings(
    aggregate in publishM2 := true,
    aggregate in update := false)

