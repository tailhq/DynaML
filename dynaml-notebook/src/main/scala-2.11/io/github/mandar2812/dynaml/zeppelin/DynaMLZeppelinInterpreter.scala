package io.github.mandar2812.dynaml.zeppelin

import java.util.Properties

import ammonite.interp.{Parsers, Preprocessor}
import ammonite.repl.Repl
import ammonite.runtime.{History, Storage}
import ammonite.util.{Name, Res}
import ammonite.util.Util.CodeSource
import fastparse.core.Parsed
import io.github.mandar2812.dynaml.DynaML
import io.github.mandar2812.dynaml.repl.{Defaults, DynaMLInterpreter}
import org.apache.zeppelin.interpreter.thrift.InterpreterCompletion
import org.apache.zeppelin.interpreter.{Interpreter, InterpreterContext, InterpreterResult}

import scala.collection.JavaConversions._


class DynaMLZeppelinInterpreter(properties: Properties)
  extends Interpreter(properties) {

  protected val dynaml_instance = new DynaML()

  protected var CURRENT_LINE : Int = 0

  var lastException: Throwable = null

  var history = new History(Vector())

  val maybe_interp = dynaml_instance.instantiateDynaZepplinInterpreter()

  var dynaml_interp: DynaMLInterpreter = null

  val storageBackend: Storage = new Storage.Folder(Defaults.ammoniteHome)

  override def cancel(interpreterContext: InterpreterContext) = {
    dynaml_interp.mainThread.interrupt()
  }

  override def getFormType = Interpreter.FormType.NATIVE

  override def getProgress(interpreterContext: InterpreterContext) = 0

  override def close() = {

  }

  override def completion(buf: String, cursor: Int) = {
    val comp = dynaml_interp.compilerManager.complete(cursor, dynaml_interp.predefImports.toString(), buf)
    comp._2.zip(comp._3).map(s => new InterpreterCompletion(s._1, s._2))
  }

  def addHistory(code: String) = if (code != "") {
    storageBackend.fullHistory() = storageBackend.fullHistory() :+ code
    history = history :+ code
  }


  override def interpret(s: String, interpreterContext: InterpreterContext) = {
    addHistory(s)

/*
    val res = dynaml_interp.processExec(s, CURRENT_LINE, () => CURRENT_LINE += 1)

    if (res.isSuccess) {
      new InterpreterResult(InterpreterResult.Code.SUCCESS, res.toString)
    } else new InterpreterResult(InterpreterResult.Code.ERROR, res.toString)
*/

    val wrapperName = Name("cmd" + CURRENT_LINE)
    val fileName = wrapperName.encoded + ".sc"
    val result = for {
      blocks <- Preprocessor.splitScript(ammonite.interp.Interpreter.skipSheBangLine(s), fileName)

      metadata <- dynaml_interp.processAllScriptBlocks(
        blocks.map(_ => None),
        Res.Success(blocks),
        dynaml_interp.predefImports ++ dynaml_interp.frameImports,
        CodeSource(
          wrapperName,
          Seq(),
          Seq(Name("ammonite"), Name("$sess")),
          Some(dynaml_interp.wd/"(console)")
        ),
        (processed, indexedWrapperName) =>
          dynaml_interp.evaluateLine(
            processed, dynaml_interp.printer, fileName,
            indexedWrapperName, false, () => CURRENT_LINE += 1),
        autoImport = true,
        ""
      )
    } yield {
      metadata
    }

    if(result.isSuccess) {

      /*val processedResult = result.flatMap(d => {
        Res(Some(d._2.mkString("\n")), "")
      })

      val Res.Success(resStr) = processedResult*/

      val resStr = result.flatMap(d => {
        Res(Some(d.blockInfo.map(blockm => blockm.finalImports.value.map(d => d.fromName.raw).mkString("\n")).mkString("\n")), "")
      })

      new InterpreterResult(InterpreterResult.Code.SUCCESS, resStr.toString)
    } else  {
      new InterpreterResult(InterpreterResult.Code.ERROR, result.toString)
    }

    /*Parsers.CompilationUnit.parse(s) match {
      case Parsed.Success(value, idx) =>
        val res = dynaml_interp.processLine(value._1, value._2, CURRENT_LINE, false, () => CURRENT_LINE += 1)
        Repl.handleOutput(dynaml_interp, res)
        if (res.isSuccess) new InterpreterResult(InterpreterResult.Code.SUCCESS, res.toString)
        else new InterpreterResult(InterpreterResult.Code.ERROR, res.toString)

      case Parsed.Failure(_, index, extra) =>
        new InterpreterResult(
          InterpreterResult.Code.ERROR,
          fastparse.core.ParseError.msg(extra.input, extra.traced.expected, index)
        )
    }*/
  }

  override def open() = {
    if (maybe_interp.isRight) dynaml_interp = maybe_interp.right.get
  }
}
