package io.github.mandar2812.dynaml.repl

import java.io.{InputStream, OutputStream}

import ammonite.repl.{Bind, Ref, Repl, Storage}

import scala.io.Source

/**
  * Created by mandar on 1/6/16.
  */
class DynaMLRepl(input: InputStream,
                 output: OutputStream,
                 error: OutputStream,
                 storage: Ref[Storage],
                 predef: String = "",
                 replArgs: Seq[Bind[_]] = Nil,
                 promptStr: String,
                 bannerText: String) extends
  Repl(input, output, error, storage, predef, replArgs) {

  override val prompt = Ref(promptStr)

  val banner = Source.fromFile(bannerText).getLines.mkString("\n")

  override def printBanner(): Unit = {
    printStream.println(banner)
    val version = BuildInfo.version
    printStream.println("\nWelcome to DynaML "+version+
      "\nInteractive Scala shell for Machine Learning Research")
    printStream.println(s"(Scala $scalaVersion Java $javaVersion)")
  }

}
