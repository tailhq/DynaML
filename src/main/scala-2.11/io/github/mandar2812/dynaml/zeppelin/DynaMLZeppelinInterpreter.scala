package io.github.mandar2812.dynaml.zeppelin

import java.util.Properties

import io.github.mandar2812.dynaml.DynaML
import org.apache.zeppelin.interpreter.thrift.InterpreterCompletion
import org.apache.zeppelin.interpreter.{Interpreter, InterpreterContext, InterpreterResult}
import scala.collection.JavaConversions._


class DynaMLZeppelinInterpreter(properties: Properties) extends Interpreter(properties) {

  protected val dynaml_instance = new DynaML()

  val maybe_interp = dynaml_instance.instantiateInterpreter()

  var dynaml_interp: ammonite.interp.Interpreter = null

  override def cancel(interpreterContext: InterpreterContext) = {
    dynaml_interp.mainThread.interrupt()
  }


  override def getFormType = Interpreter.FormType.SIMPLE

  override def getProgress(interpreterContext: InterpreterContext) = 0

  override def close() = {

  }

  override def completion(buf: String, cursor: Int) = {
    val comp = dynaml_interp.compilerManager.complete(cursor, dynaml_interp.predefImports.toString(), buf)
    comp._2.zip(comp._3).map(s => new InterpreterCompletion(s._1, s._2))
  }

  override def interpret(s: String, interpreterContext: InterpreterContext) = {
    val res = dynaml_interp.processExec(s, 0, () => Unit)

    if (res.isSuccess) new InterpreterResult(InterpreterResult.Code.SUCCESS, res.toString)
    else new InterpreterResult(InterpreterResult.Code.ERROR, res.toString)

  }

  override def open() = {
    if (maybe_interp.isRight) dynaml_interp = maybe_interp.right.get
  }
}
