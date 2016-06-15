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

import java.io.File
import ammonite.ops.{read, _}
import ammonite.repl.{ArgParseException, _}
import repl._
import ammonite.ops._
import scala.reflect.runtime.universe.TypeTag
import language.experimental.macros


//entry point to start my custom Scala REPL
object DynaML {

  case class Config(predef: String =
                    """import ammonite.ops._;""" +
                      """load.module(cwd / RelPath("conf/DynaMLInit.scala"))""",
                    predefFile: Option[Path] = None,
                    code: Option[String] = None,
                    ammoniteHome: Path = defaultAmmoniteHome,
                    file: Option[Path] = None,
                    args: Seq[String] = Vector.empty,
                    kwargs: Map[String, String] = Map.empty,
                    time: Boolean = false)
  
  def defaultAmmoniteHome = Path(System.getProperty("user.home"))/".ammonite"

  /**
    * The command-line entry point, which does all the argument parsing before
    * delegating to [[run]]
    */
  def main(args: Array[String]) = {
    val parser = new scopt.OptionParser[Config]("ammonite") {
      head("ammonite", ammonite.Constants.version)
      opt[String]('p', "predef")
        .action((x, c) => c.copy(predef = x))
        .text("Any commands you want to execute at the start of the REPL session")
      opt[String]('f', "predef-file")
        .action((x, c) => c.copy(predefFile = Some(Path(x, cwd))))
        .text("Lets you load your predef from a custom location")
      opt[String]('c', "code")
        .action((x, c) => c.copy(code = Some(x)))
        .text("Pass in code to be run immediately in the REPL")
      opt[Unit]('t', "time")
        .action((_, c) => c.copy(time = true))
        .text("Print time taken for each step")
      opt[File]('h', "home")
        .valueName("<file>")
        .action((x, c) => c.copy(ammoniteHome = Path(x, cwd)))
        .text("The home directory of the REPL; where it looks for config and caches")
      arg[String]("<file-args>...")
        .optional()
        .action { (x, c) => c.copy(file = Some(Path(x, cwd))) }
        .text("The Ammonite script file you want to execute")
      arg[String]("<args>...")
        .optional()
        .unbounded()
        .action { (x, c) => c.copy(args = c.args :+ x) }
        .text("Any arguments you want to pass to the Ammonite script file")
    }
    val (before, after) = args.splitAt(args.indexOf("--") match {
      case -1 => Int.MaxValue
      case n => n
    })
    val keywordTokens = after.drop(1)
    assert(
      keywordTokens.length % 2 == 0,
      s"""Only pairs of keyword arguments can come after `--`.
          |Invalid number of tokens: ${keywordTokens.length}""".stripMargin
    )

    val kwargs = for(Array(k, v) <- keywordTokens.grouped(2)) yield{

      assert(
        k.startsWith("--") &&
          scalaparse.syntax
            .Identifiers
            .Id
            .parse(k.stripPrefix("--"))
            .isInstanceOf[fastparse.core.Parsed.Success[_]],
        s"""Only pairs of keyword arguments can come after `--`.
            |Invalid keyword: $k""".stripMargin
      )
      (k.stripPrefix("--"), v)
    }

    for(c <- parser.parse(before, Config())){
      Timer.show = c.time
      run(
        c.predef,
        c.ammoniteHome,
        c.code,
        c.predefFile,
        c.file,
        c.args,
        kwargs.toMap
      )
    }
  }

  implicit def ammoniteReplArrowBinder[T](t: (String, T))(implicit typeTag: TypeTag[T]) = {
    Bind(t._1, t._2)(typeTag)
  }

  /**
    * The debug entry-point: embed this inside any Scala program to open up
    * an ammonite REPL in-line with access to that program's variables for
    * inspection.
    */
  def debug(replArgs: Bind[_]*): Any = {

    val storage = Storage(defaultAmmoniteHome, None)
    val repl = new DynaMLRepl(
      input = System.in, output = System.out, error = System.err,
      storage = Ref(storage),
      predef = "",
      replArgs = replArgs,
      promptStr = "DynaML>",
      bannerText = "conf/banner.txt"
    )

    repl.run()
  }

  /**
    * The main entry-point after partial argument de-serialization.
    */
  def run(predef: String =
          """import ammonite.ops._;""" + """load.module(cwd / RelPath("conf/DynaMLInit.scala"))""",
          ammoniteHome: Path = defaultAmmoniteHome,
          code: Option[String] = None,
          predefFile: Option[Path] = None,
          file: Option[Path] = None,
          args: Seq[String] = Vector.empty,
          kwargs: Map[String, String] = Map.empty,
          prompt: String = "DynaML>",
          banner: String = "conf/banner.txt") = {

    Timer("Repl.run Start")
    def storage = Storage(ammoniteHome, predefFile)
    lazy val repl = new DynaMLRepl(
      System.in, System.out, System.err,
      storage = Ref(storage),
      predef = predef, promptStr = prompt,
      bannerText = banner
    )
    (file, code) match{
      case (None, None) => println("Loading..."); repl.run()
      case (Some(path), None) => runScript(repl, path, args, kwargs)
      case (None, Some(code)) => repl.interp.replApi.load(code)
    }
    Timer("Repl.run End")
  }

  def runScript(repl: Repl, path: Path, args: Seq[String], kwargs: Map[String, String]): Unit = {
    val imports = repl.interp.processModule(read(path))
    repl.interp.init()
    imports.find(_.toName == "main").foreach { i =>
      val quotedArgs =
        args.map(pprint.PPrinter.escape)
          .map(s => s"""arg("$s")""")

      val quotedKwargs =
        kwargs.mapValues(pprint.PPrinter.escape)
          .map { case (k, s) => s"""$k=arg("$s")""" }
      try{
        repl.interp.replApi.load(s"""
                                    |import ammonite.repl.ScriptInit.{arg, callMain, pathRead}
                                    |callMain{
                                    |main(${(quotedArgs ++ quotedKwargs).mkString(", ")})
                                    |}
        """.stripMargin)
      } catch {
        case e: ArgParseException =>
          // For this semi-expected invalid-argument exception, chop off the
          // irrelevant bits of the stack trace to reveal only the part which
          // describes how parsing failed
          e.setStackTrace(Array())
          e.cause.setStackTrace(e.cause.getStackTrace.takeWhile( frame =>
            frame.getClassName != "ammonite.repl.ScriptInit$" ||
              frame.getMethodName != "parseScriptArg"
          ))
          throw e
      }
    }
  }
  
  
  
}
