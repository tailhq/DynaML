package org.kuleuven.esat

import java.io.{CharArrayWriter, PrintWriter}
import scala.tools.nsc.interpreter.ILoop
import scala.tools.nsc.Settings


object DynaML extends App {

  override def main(args: Array[String]) = {
    val repl = new ILoop {
      override def prompt = "DynaML>"

      addThunk {
        intp.beQuietDuring {
          intp.addImports("breeze.linalg._")
          intp.addImports("org.kuleuven.esat.graphicalModels._")
          intp.addImports("org.kuleuven.esat.svm._")
          intp.addImports("org.kuleuven.esat.utils")
          intp.addImports("org.kuleuven.esat.kernels._")
          intp.addImports("org.apache.spark.SparkContext")
          intp.addImports("org.apache.spark.SparkConf")
        }
      }

      override def printWelcome() {
        echo("    ___       ___       ___       ___       ___       ___ "+
          "  \n   /\\  \\     /\\__\\     /\\__\\     /\\  \\     /\\__\\ "+
          "    /\\__\\  \n  /::\\  \\   |::L__L   /:| _|_   /::\\  \\   /::L_L_ "+
          "  /:/  /  \n /:/\\:\\__\\  |:::\\__\\ /::|/\\__\\ /::\\:\\__\\ "+
          "/:/L:\\__\\ /:/__/   \n \\:\\/:/  /  /:;;/__/ \\/|::/  / \\/\\::/ "+
          " / \\/_/:/  / \\:\\  \\   \n  \\::/  /   \\/__/      |:/  /    /:/ "+
          " /    /:/  /   \\:\\__\\  \n   \\/__/               \\/__/    "+
          " \\/__/     \\/__/     \\/__/  ")

        echo("\nWelcome to DynaML v 1.2\nInteractive Scala shell")
        echo("STADIUS ESAT KU Leuven (2015)\n")
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