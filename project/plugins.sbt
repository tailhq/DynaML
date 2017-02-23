logLevel := Level.Info

enablePlugins(GitVersioning)

addSbtPlugin("com.typesafe.sbt" % "sbt-git" % "0.8.5")

addSbtPlugin("com.typesafe.sbt" % "sbt-native-packager" % "1.1.1")

addSbtPlugin("com.eed3si9n" % "sbt-buildinfo" % "0.6.1")

//addSbtPlugin("org.bytedeco" % "sbt-javacpp" % "1.9")
