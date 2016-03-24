/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
* */
package io.github.mandar2812.dynaml

import java.io.{FileInputStream, CharArrayWriter, PrintWriter}
import java.util.Properties
import scala.tools.nsc.interpreter.ILoop
import scala.tools.nsc.Settings


object DynaML extends App {

  override def main(args: Array[String]) = {
    val repl = new ILoop {
      override def prompt = "DynaML>"


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
        echo("\nWelcome to DynaML v "+version+
          "\nInteractive Scala shell for Machine Learning Research")
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

    new sys.SystemProperties += ("scala.repl.autoruncode" ->
      "conf/DynaMLInit.scala")

    repl.process(settings)
  }
}