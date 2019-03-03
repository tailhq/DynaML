package io.github.mandar2812.dynaml.utils.sumac

import org.scalatest.FunSuite
import org.scalatest.Matchers
import scala.collection._

class ArgumentParserTest extends FunSuite with Matchers {

  test("parse") {
    val c = SimpleClass("a", 0, 1.4, 2)
    val fieldArgs = classOf[SimpleClass].getDeclaredFields.map{f => FieldArgAssignable("",f, c)}
    val argParser = new ArgumentParser(fieldArgs)

    {
      val parsed = getSimpleNameToArgMap(argParser.parse(Array("--name", "foo")))
      parsed.size should be (1)
      parsed should contain key ("name")
      parsed("name") should be ("foo")
    }


    {
      val parsed = getSimpleNameToArgMap(argParser.parse(Array("--count", "5", "--dummy", "7.4e3", "--name", "ooga")))
      parsed.size should be (3)
      parsed should contain key ("count")
      parsed("count") should be (5)
      parsed should contain key ("dummy")
      parsed("dummy") should be (7.4e3)
      parsed should contain key ("name")
      parsed("name") should be ("ooga")
    }

    {
      val parsed = getSimpleNameToArgMap(argParser.parse("--count  5  --dummy 7.4e3 --name ooga"))
      parsed.size should be (3)
      parsed should contain key ("count")
      parsed("count") should be (5)
      parsed should contain key ("dummy")
      parsed("dummy") should be (7.4e3)
      parsed should contain key ("name")
      parsed("name") should be ("ooga")
    }
  }

  test("reserved arguments should be filtered") {
    val c = SimpleClass("a", 0, 1.4, 2)
    val fieldArgs = classOf[SimpleClass].getDeclaredFields.map{f => FieldArgAssignable("",f, c)}
    val argParser = new ArgumentParser(fieldArgs)

    val parsed = getSimpleNameToArgMap(argParser.parse(Array("--count", "5", "--dummy", "7.4e3", "--name", "ooga", "--sumac.debugArgs", "true")))

    parsed should not contain key("sumac.debugArgs")

  }

  def getSimpleNameToArgMap(parsedArgs : Map[_ <: ArgAssignable, ValueHolder[_]]) = {
    parsedArgs.map{kv => kv._1.getName -> kv._2.value}.toMap[String, Any]
  }

  test("remove newlines") {
    ArgumentParser.argCLIStringToArgList(
      """--x "5 \\\\\" 6\"\\" \
         --y 6 \
         --blank "" \
         """ + "--z '\t\n'") should be (Array("--x", """5 \\\\\" 6\"\\""", "--y", "6","--blank", "", "--z", "\t\n"))
  }
}


case class SimpleClass(val name: String, val count: Int, val dummy: Double, val count2: Int)
