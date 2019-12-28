import sbt._

logLevel := Level.Error

resolvers ++= Seq(Resolver.sonatypeRepo("snapshots"))

addSbtPlugin("io.get-coursier" % "sbt-coursier" % "2.0.0-RC3-3")

addSbtPlugin("com.typesafe.sbt" % "sbt-native-packager" % "1.4.1")

addSbtPlugin("se.marcuslonnberg" % "sbt-docker" % "1.5.0")

addSbtPlugin("com.eed3si9n" % "sbt-buildinfo" % "0.7.0")

addSbtPlugin("org.scoverage" % "sbt-scoverage" % "1.6.1")
addSbtPlugin("org.scoverage" % "sbt-coveralls" % "1.2.7")

addSbtPlugin("com.dwijnand" % "sbt-compat" % "1.2.6")

//addSbtPlugin("ch.epfl.scala.index" % "sbt-scaladex" % "0.1.3")

addSbtPlugin("org.xerial.sbt" % "sbt-sonatype" % "2.6")

addSbtPlugin("com.jsuereth" % "sbt-pgp" % "2.0.1")
addSbtPlugin("ch.epfl.scala" % "sbt-scalafix" % "0.9.6")
addSbtPlugin("org.scalameta" % "sbt-mdoc" % "1.3.6" )
addSbtPlugin("com.eed3si9n" % "sbt-unidoc" % "0.4.2")
