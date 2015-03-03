package org.kuleuven.esat

import java.io.{CharArrayWriter, PrintWriter}
import scala.tools.nsc.interpreter.ILoop
import scala.tools.nsc.Settings


object bayesLearn extends App {

  override def main(args: Array[String]) = {
    val repl = new ILoop {
      override def prompt = "bayeslearn>"

      addThunk {
        intp.beQuietDuring {
          intp.addImports("breeze.linalg._")
          intp.addImports("org.kuleuven.esat.graphicalModels._")
          intp.addImports("org.kuleuven.esat.utils")
          intp.addImports("org.kuleuven.esat.kernels._")
        }
      }

      override def printWelcome() {
        echo("      ___           ___           ___           ___           ___"+
          "              \n     /\\  \\         /\\  \\         |\\__\\         "+
          "/\\  \\         /\\  \\             \n    /::\\  \\       /::\\  \\   "+
          "     |:|  |       /::\\  \\       /::\\  \\            \n   /:/\\:\\  "+
          "\\     /:/\\:\\  \\       |:|  |      /:/\\:\\  \\     /:/\\ \\  \\   "+
          "        \n  /::\\~\\:\\__\\   /::\\~\\:\\  \\      |:|__|__   /::\\~\\"+
          ":\\  \\   _\\:\\~\\ \\  \\          \n /:/\\:\\ \\:|__| /:/\\:\\ \\:\\__\\"+
          "     /::::\\__\\ /:/\\:\\ \\:\\__\\ /\\ \\:\\ \\ \\__\\         \n "+
          "\\:\\~\\:\\/:/  / \\/__\\:\\/:/  /    /:/~~/~    \\:\\~\\:\\ \\/__/ "+
          "\\:\\ \\:\\ \\/__/         \n  \\:\\ \\::/  /       \\::/  /    /:/  /"+
          "       \\:\\ \\:\\__\\    \\:\\ \\:\\__\\           \n   \\:\\/:/  /  "+
          "      /:/  /     \\/__/         \\:\\ \\/__/     \\:\\/:/  /        "+
          "   \n    \\::/__/        /:/  /                     \\:\\__\\        "+
          "\\::/  /            \n     ~~            \\/__/                       "+
          "\\/__/         \\/__/             \n      ___       ___           ___ "+
          "          ___           ___                  \n     /\\__\\     /\\  \\"+
          "         /\\  \\         /\\  \\         /\\__\\                 \n    "+
          "/:/  /    /::\\  \\       /::\\  \\       /::\\  \\       /::|  |      "+
          "          \n   /:/  /    /:/\\:\\  \\     /:/\\:\\  \\     /:/\\:\\  \\ "+
          "    /:|:|  |                \n  /:/  /    /::\\~\\:\\  \\   /::\\~\\:\\  "+
          "\\   /::\\~\\:\\  \\   /:/|:|  |__              \n /:/__/    /:/\\:\\ "+
          "\\:\\__\\ /:/\\:\\ \\:\\__\\ /:/\\:\\ \\:\\__\\ /:/ |:| /\\__\\       "+
          "      \n \\:\\  \\    \\:\\~\\:\\ \\/__/ \\/__\\:\\/:/  / \\/_|::\\/:/ "+
          " / \\/__|:|/:/  /             \n  \\:\\  \\    \\:\\ \\:\\__\\        "+
          "\\::/  /     |:|::/  /      |:/:/  /              \n   \\:\\  \\    \\:\\"+
          " \\/__/        /:/  /      |:|\\/__/       |::/  /               \n    "+
          "\\:\\__\\    \\:\\__\\         /:/  /       |:|  |         /:/  /       "+
          "         \n     \\/__/     \\/__/         \\/__/         \\|__|         "+
          "\\/__/                 ")

        echo("\nWelcome to Bayes Learn v 0.12\nInteractive Scala shell")
        echo("STADIUS ESAT KU Leuven (2015)\n")
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
  }
}