logLevel := Level.Error

addSbtPlugin("com.typesafe.sbt" % "sbt-native-packager" % "1.3.4")

addSbtPlugin("se.marcuslonnberg" % "sbt-docker" % "1.5.0")

addSbtPlugin("com.eed3si9n" % "sbt-buildinfo" % "0.7.0")

addSbtPlugin("org.scoverage" % "sbt-scoverage" % "1.3.5")

addSbtPlugin("com.dwijnand" % "sbt-compat" % "1.2.6")
addSbtPlugin("io.get-coursier" % "sbt-coursier" % "1.1.0-M7")

//addSbtPlugin("org.ensime" % "sbt-ensime" % "2.1.0")
//addSbtPlugin("ch.epfl.scala.index" % "sbt-scaladex" % "0.1.3")
