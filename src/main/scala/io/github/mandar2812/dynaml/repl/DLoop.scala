package io.github.mandar2812.dynaml.repl

import java.io.BufferedReader

import scala.tools.nsc.interpreter._
import scala.tools.nsc.{Settings, util}

//looper class to read user inputs and let interpreter evaluate at loop() function.
class DLoop(in0: Option[BufferedReader], override protected val out: JPrintWriter) extends ILoop {
  def this() = this(None, new JPrintWriter(Console.out, true))

  import LoopCommand.cmd

  //custom commands list
  lazy val myCommandSeq = Seq(
    cmd("myCommand", "[-v] <expr>", "my command!", myCommand)
  )

  //add to my custom commands to default commands.
  override def commands = super.commands ++ myCommandSeq

  override def prompt = "DynaML>"

  //myCommand implementation
  private def myCommand(line0: String): Result = {
    line0.trim match {
      case "" => ":myCommand [-v] <expression>"
      case s => s"""This is a custom command example. """+
        """You can do something from value "${s}" with custom Scala interpreter."""
    }
  }

  //show welcome message when user login
  override def printWelcome(): Unit = {
    echo("    ___       ___       ___       ___       ___       ___ "+
      "  \n   /\\  \\     /\\__\\     /\\__\\     /\\  \\     /\\__\\ "+
      "    /\\__\\  \n  /::\\  \\   |::L__L   /:| _|_   /::\\  \\   /::L_L_ "+
      "  /:/  /  \n /:/\\:\\__\\  |:::\\__\\ /::|/\\__\\ /::\\:\\__\\ "+
      "/:/L:\\__\\ /:/__/   \n \\:\\/:/  /  /:;;/__/ \\/|::/  / \\/\\::/ "+
      " / \\/_/:/  / \\:\\  \\   \n  \\::/  /   \\/__/      |:/  /    /:/ "+
      " /    /:/  /   \\:\\__\\  \n   \\/__/               \\/__/    "+
      " \\/__/     \\/__/     \\/__/  ")
    val version = BuildInfo.version
    echo("\nWelcome to DynaML "+version+
      "\nInteractive Scala shell for Machine Learning Research")
  }

  settings = new Settings
  settings.embeddedDefaults[this.type]

  //MyInterpreter initialization
  override def createInterpreter(): Unit = {
    if (addedClasspath != "")
      settings.classpath append addedClasspath

    intp = new MyInterpreter()
  }

  //Interpreter class. Use DynaMLClassLoader to load any classes
  class MyInterpreter extends IMain(settings, out) {
    private var myClassLoader: Option[DynaMLClassLoader] = None

    override def resetClassLoader(): Unit = {
      myClassLoader = None
    }

    override def classLoader: util.AbstractFileClassLoader = {
      myClassLoader.getOrElse {
        myClassLoader = Some(new DynaMLClassLoader(replOutput.dir, parentClassLoader, out))
        myClassLoader.get
      }
    }
  }

}
