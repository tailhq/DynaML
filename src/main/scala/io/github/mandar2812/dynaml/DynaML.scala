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

import scala.reflect.io.File
import scala.tools.nsc.GenericRunnerCommand.{AsJar, AsObject, AsScript, Error}
import scala.tools.nsc._

import repl._

//entry point to start my custom Scala REPL
object DynaML extends DynaML {
  def main(args: Array[String]): Unit = {
    new sys.SystemProperties += (
      "scala.repl.autoruncode" -> "conf/DynaMLInit.scala")
    if (!process(args)) sys.exit(1)
  }
}

class DynaML {
  def errorFn(str: String, e: Option[Throwable] = None, isFailure: Boolean = true): Boolean = {
    if (str.nonEmpty) Console.err println str
    e foreach (_.printStackTrace())
    !isFailure
  }

  def process(args: Array[String]): Boolean = {
    val command = new GenericRunnerCommand(args.toList, (x: String) => errorFn(x))
    import command.{settings, howToRun, thingToRun, shortUsageMsg, shouldStopWithInfo}
    settings.usejavacp.value = true
    def sampleCompiler = new Global(settings) // def so it's not created unless needed

    def run(): Boolean = {
      def isE = !settings.execute.isDefault
      def dashe = settings.execute.value

      def isI = !settings.loadfiles.isDefault
      def dashi = settings.loadfiles.value

      // Deadlocks on startup under -i unless we disable async.
      if (isI)
        settings.Yreplsync.value = true

      def combinedCode = {
        val files = if (isI) dashi map (file => File(file).slurp()) else Nil
        val str = if (isE) List(dashe) else Nil

        files ++ str mkString "\n\n"
      }

      def runTarget(): Either[Throwable, Boolean] = howToRun match {
        case AsObject =>
          ObjectRunner.runAndCatch(settings.classpathURLs, thingToRun, command.arguments)
        case AsScript =>
          ScriptRunner.runScriptAndCatch(settings, thingToRun, command.arguments)
        case AsJar =>
          JarRunner.runJar(settings, thingToRun, command.arguments)
        case Error =>
          Right(false)
        case _ =>
          // We start the repl when no arguments are given.
          // taisukeoe(oeuia.t@gmail.com modifies below line) on 2014/12/7
          Right(new DLoop process settings)
      }

      /** If -e and -i were both given, we want to execute the -e code after the
        * -i files have been included, so they are read into strings and prepended to
        * the code given in -e.  The -i option is documented to only make sense
        * interactively so this is a pretty reasonable assumption.
        *
        * This all needs a rewrite though.
        */
      if (isE) {
        ScriptRunner.runCommand(settings, combinedCode, thingToRun +: command.arguments)
      }
      else runTarget() match {
        case Left(ex) => errorFn("", Some(ex)) // there must be a useful message of hope to offer here
        case Right(b) => b
      }
    }

    if (!command.ok)
      errorFn(f"%n$shortUsageMsg")
    else if (shouldStopWithInfo)
      errorFn(command getInfoMessage sampleCompiler, isFailure = false)
    else
      run()
  }
}
