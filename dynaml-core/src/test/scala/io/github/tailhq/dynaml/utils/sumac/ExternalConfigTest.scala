package io.github.tailhq.dynaml.utils.sumac

import org.scalatest.FunSuite
import collection._
import org.scalatest.Matchers

class ExternalConfigTest extends FunSuite with Matchers {
  test("modifying args"){
    val args = new ExtArgs() with DefaultNumeroUno
    args.parse(Array("--x", "5", "--y", "hi there"))
    args.x should be (5)
    args.y should be ("hi there")

    //Let x get defaulted by external config.  Note that this is NOT the right way to provide argument default.
    // This is just an easy way to test the functionality
    args.parse(Array("--y", "blah"))
    args.x should be (17)
    args.y should be ("blah")


    args.saveConfig()
    args.numeroUnoSave should be (true)
  }

  test("stacked traits"){
    //right-most trait goes first
    val arg1 = new ExtArgs() with DefaultNumeroUno with DefaultNumberTwo
    arg1.parse(Array[String]())
    arg1.x should be (21)
    arg1.ooga should be (3.5f)

    val arg2 = new ExtArgs() with DefaultNumberTwo with DefaultNumeroUno
    arg2.parse(Array[String]())
    arg2.parse(Array[String]())
    arg2.x should be (17)
    arg2.ooga should be (3.5f)

    arg1.saveConfig()
    arg1.numeroUnoSave should be (true)
    arg1.numberTwoSave should be (true)

    arg2.saveConfig()
    arg2.numeroUnoSave should be (true)
    arg2.numberTwoSave should be (true)
  }


  class ExtArgs extends FieldArgs {
    var x: Int = _
    var y: String = _
    var ooga: Float = _
  }


  trait DefaultNumeroUno extends ExternalConfig {
    self: Args =>
    var numeroUnoSave = false
    abstract override def readArgs(originalArgs: Map[String,String]): Map[String,String] = {
      super.readArgs(
        if (originalArgs.contains("x"))
          originalArgs
        else
          originalArgs ++ Map("x" -> "17")
      )
    }

    abstract override def saveConfig() {
      numeroUnoSave = true
      super.saveConfig()
    }
  }

  trait DefaultNumberTwo extends ExternalConfig {
    self: Args =>
    var numberTwoSave = false
    abstract override def readArgs(originalArgs: Map[String,String]): Map[String,String] = {
      val withX = if (originalArgs.contains("x"))
        originalArgs
      else
        originalArgs ++ Map("x" -> "21")

      val withOoga = if (withX.contains("ooga"))
        withX
      else
        withX ++ Map("ooga" -> "3.5")

      super.readArgs(withOoga)
    }

    abstract override def saveConfig() {
      numberTwoSave  = true
      super.saveConfig()
    }

  }

}
