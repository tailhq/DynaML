package io.github.mandar2812.dynaml.repl

import java.io.{InputStream, OutputStream}

import ammonite.repl.{Bind, Ref, Repl, Storage}

/**
  * Created by mandar on 1/6/16.
  */
class DynaMLRepl(input: InputStream,
                 output: OutputStream,
                 error: OutputStream,
                 storage: Ref[Storage],
                 predef: String = "",
                 replArgs: Seq[Bind[_]] = Nil) extends
  Repl(input, output, error, storage, predef, replArgs) {

  override val prompt = Ref("DynaML>")

  override def printBanner(): Unit = {
    printStream.println("    ___       ___       ___       ___       ___       ___ "+
      "  \n   /\\  \\     /\\__\\     /\\__\\     /\\  \\     /\\__\\ "+
      "    /\\__\\  \n  /::\\  \\   |::L__L   /:| _|_   /::\\  \\   /::L_L_ "+
      "  /:/  /  \n /:/\\:\\__\\  |:::\\__\\ /::|/\\__\\ /::\\:\\__\\ "+
      "/:/L:\\__\\ /:/__/   \n \\:\\/:/  /  /:;;/__/ \\/|::/  / \\/\\::/ "+
      " / \\/_/:/  / \\:\\  \\   \n  \\::/  /   \\/__/      |:/  /    /:/ "+
      " /    /:/  /   \\:\\__\\  \n   \\/__/               \\/__/    "+
      " \\/__/     \\/__/     \\/__/  ")
    val version = BuildInfo.version
    printStream.println("\nWelcome to DynaML "+version+
      "\nInteractive Scala shell for Machine Learning Research")
    printStream.println(s"(Scala $scalaVersion Java $javaVersion)")
  }

}
