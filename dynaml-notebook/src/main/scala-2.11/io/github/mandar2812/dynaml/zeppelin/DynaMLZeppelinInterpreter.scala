package io.github.mandar2812.dynaml.zeppelin

import java.nio.charset.Charset
import java.util.Properties

import ammonite.interp.{Parsers, Preprocessor}
import ammonite.repl.Repl
import ammonite.runtime.{History, Storage}
import ammonite.util.{Name, Res}
import ammonite.util.Util.CodeSource
import fastparse.core.Parsed
import io.github.mandar2812.dynaml.DynaZeppelin
import io.github.mandar2812.dynaml.repl.{Defaults, DynaMLInterpreter}
import org.apache.commons.io.output.ByteArrayOutputStream
import org.apache.zeppelin.interpreter.thrift.InterpreterCompletion
import org.apache.zeppelin.interpreter.{Interpreter, InterpreterContext, InterpreterResult}

import scala.collection.JavaConversions._


class DynaMLZeppelinInterpreter(properties: Properties) extends Interpreter(properties) {

  protected val outputBuffer = new ByteArrayOutputStream()

  protected val dynaml_instance = new DynaZeppelin(outputStream = outputBuffer)

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

  override def open() = {
    if (maybe_interp.isRight) dynaml_interp = maybe_interp.right.get
  }

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

    val wrapperName = Name("cmd" + CURRENT_LINE)
    val fileName = wrapperName.encoded + ".sc"
    val result = for {
      blocks <- Preprocessor.splitScript(ammonite.interp.Interpreter.skipSheBangLine(s), fileName)

      codeSource = CodeSource(
        wrapperName,
        Seq(),
        Seq(Name("ammonite"), Name("$sess")),
        Some(dynaml_interp.wd/"(console)")
      )

      metadata <- dynaml_interp.processAllScriptBlocks(
        blocks.map(_ => None),
        Res.Success(blocks),
        dynaml_interp.predefImports ++ dynaml_interp.frameImports,
        codeSource,
        (processed, indexedWrapperName) =>
          dynaml_interp.evaluateLine(
            processed, dynaml_interp.printer, fileName,
            indexedWrapperName, silent = false, () => CURRENT_LINE += 1),
        autoImport = true,
        ""
      )
    } yield {
      metadata
    }

    if(result.isSuccess) {

      val output = outputBuffer.toString(Charset.defaultCharset())

      outputBuffer.reset()
      val resStr = result.flatMap(d => {
        Res(Some(d.blockInfo.map(blockm => blockm.finalImports.value.map(d => d.fromName.raw).mkString("\n")).mkString("\n")), "")
      })

      new InterpreterResult(InterpreterResult.Code.SUCCESS, output)
    } else  {
      new InterpreterResult(InterpreterResult.Code.ERROR, result.toString)
    }
  }

  def evaluate(s: String) = {
    addHistory(s)

    val wrapperName = Name("cmd" + CURRENT_LINE)
    val fileName = wrapperName.encoded + ".sc"
    val result = for {
      blocks <- Preprocessor.splitScript(ammonite.interp.Interpreter.skipSheBangLine(s), fileName)

      codeSource = CodeSource(
        wrapperName,
        Seq(),
        Seq(Name("ammonite"), Name("$sess")),
        Some(dynaml_interp.wd/"(console)")
      )

      metadata <- dynaml_interp.processAllScriptBlocks(
        blocks.map(_ => None),
        Res.Success(blocks),
        dynaml_interp.predefImports ++ dynaml_interp.frameImports,
        codeSource,
        (processed, indexedWrapperName) =>
          dynaml_interp.evaluateLine(
            processed, dynaml_interp.printer, fileName,
            indexedWrapperName, silent = false, () => CURRENT_LINE += 1),
        autoImport = true,
        ""
      )
    } yield {
      metadata
    }

    if(result.isSuccess) {

      val output = outputBuffer.toString(Charset.defaultCharset())

      outputBuffer.reset()
      val resStr = result.flatMap(d => {
        Res(Some(d.blockInfo.map(blockm => blockm.finalImports.value.map(d => d.fromName.raw).mkString("\n")).mkString("\n")), "")
      })

      output
    } else  {
      result.toString
    }
  }

}
