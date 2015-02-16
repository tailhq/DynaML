package org.kuleuven.esat

import java.io.{CharArrayWriter, PrintWriter}
import java.net.URLClassLoader

import scala.tools.nsc.interpreter.ILoop
import scala.tools.nsc.Settings


object bayesLearn extends App {
  override def main(args: Array[String]) = {
    //println(System.getProperty("java.class.path"))
    val repl = new ILoop {
      override def prompt = "==> "

      addThunk {
        intp.beQuietDuring {
          intp.addImports("breeze.linalg._")
          intp.addImports("org.kuleuven.esat.graphicalModels._")
          intp.addImports("org.kuleuven.esat.utils")
          intp.addImports("org.kuleuven.esat.kernels._")
        }
      }

      override def printWelcome() {
        echo("\n" +
          "         \\,,,/\n" +
          "         (o o)\n" +
          "-----oOOo-(_)-oOOo-----")
      }
    }
    val settings = new Settings
    settings.Yreplsync.value = true

    if (isRunFromSBT) {
      settings.embeddedDefaults[bayesLearn.type]
    } else {
      settings.usejavacp.value = true
    }

    def isRunFromSBT = {
      val c = new CharArrayWriter()
      new Exception().printStackTrace(new PrintWriter(c))
      c.toString().contains("at sbt.")
    }

    repl.process(settings)
    //repl.addImports("import breeze.linalg._")
    //repl.addImports("org.kuleuven.esat.graphicalModels._")

  }
}