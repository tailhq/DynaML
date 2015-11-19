package io.github.mandar2812.dynaml

import java.io.{FileInputStream, CharArrayWriter, PrintWriter}
import java.util.Properties
import scala.tools.nsc.interpreter.ILoop
import scala.tools.nsc.Settings


object DynaML extends App {

  override def main(args: Array[String]) = {
    val repl = new ILoop {
      override def prompt = "DynaML>"

      addThunk {
        intp.beQuietDuring {
          intp.addImports("breeze.linalg._")
          intp.addImports("io.github.mandar2812.dynaml.models._")
          intp.addImports("io.github.mandar2812.dynaml.models.svm._")
          intp.addImports("io.github.mandar2812.dynaml.utils")
          intp.addImports("io.github.mandar2812.dynaml.kernels._")
          intp.addImports("io.github.mandar2812.dynaml.examples._")
          intp.addImports("org.apache.spark.SparkContext")
          intp.addImports("org.apache.spark.SparkConf")
          intp.addImports("io.github.mandar2812.dynaml.pipes._")
        }
      }

      override def printWelcome() {
        val prop = new Properties()
        prop.load(getClass.getClassLoader.getResourceAsStream("dynaml.properties"))
        echo("    ___       ___       ___       ___       ___       ___ "+
          "  \n   /\\  \\     /\\__\\     /\\__\\     /\\  \\     /\\__\\ "+
          "    /\\__\\  \n  /::\\  \\   |::L__L   /:| _|_   /::\\  \\   /::L_L_ "+
          "  /:/  /  \n /:/\\:\\__\\  |:::\\__\\ /::|/\\__\\ /::\\:\\__\\ "+
          "/:/L:\\__\\ /:/__/   \n \\:\\/:/  /  /:;;/__/ \\/|::/  / \\/\\::/ "+
          " / \\/_/:/  / \\:\\  \\   \n  \\::/  /   \\/__/      |:/  /    /:/ "+
          " /    /:/  /   \\:\\__\\  \n   \\/__/               \\/__/    "+
          " \\/__/     \\/__/     \\/__/  ")
        val version = prop.getProperty("dynaml.version")
        echo("\nWelcome to DynaML v "+version+"\nInteractive Scala shell")
      }
    }
    val settings = new Settings
    settings.Yreplsync.value = true

    if (isRunFromSBT) {
      settings.embeddedDefaults[DynaML.type]
    } else {
      settings.usejavacp.value = true
    }

    def isRunFromSBT = {
      val c = new CharArrayWriter()
      new Exception().printStackTrace(new PrintWriter(c))
      c.toString().contains("at sbt.")
    }

    repl.process(settings)
  }
}