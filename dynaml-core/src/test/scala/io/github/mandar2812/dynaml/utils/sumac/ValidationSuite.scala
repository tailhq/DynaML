package io.github.mandar2812.dynaml.utils.sumac

import org.scalatest.FunSuite
import org.scalatest.Matchers
import io.github.mandar2812.dynaml.utils.sumac.validation.{FileExists, Positive, Range, Required}
import java.io.File

//import io.github.mandar2812.dynaml.utils.sumac.ThreeOrFour

class ValidationSuite extends FunSuite with Matchers {

  def parse(args: Map[String,String], msg: String)(builder: => FieldArgs) {
    val a = builder
    val exc = withClue(args){the[ArgException] thrownBy {a.parse(args)}}
    withClue(args){exc.getMessage should include(msg)}
  }

  test("@Required") {
    def parseInt(args: Map[String,String], msg: String) = {
      parse(args, msg){new IntRequiredArgs()}
    }

    parseInt(Map("a" -> "1"), "must specify a value for b")

    parseInt(Map("b" -> "1"), "must specify a value for a")
    //also an error if values are given, but they match the defaults
    parseInt(Map("a" -> "0", "b" -> "7"), "must specify a value for ")

    val intArgs = new IntRequiredArgs()
    intArgs.parse(Map("a" -> "1", "b" -> "0"))
    intArgs.a should be (1)
    intArgs.b should be (0)

    //make sure that the checks still apply when called programmatically (doesn't depend on strings at all)
    intArgs.a = 0
    the[ArgException] thrownBy {intArgs.runValidation()}


    def parseString(args: Map[String,String], msg: String) = {
      parse(args, msg){new StringRequiredArgs()}
    }

    parseString(Map("e" -> "a"), "must specify a value for f")
    parseString(Map("f" -> "hi"), "must specify a value for e")
    parseString(Map("e" -> "<null>", "f" -> "hi"), "must specify a value for e")
    parseString(Map("e" -> "blah", "f" -> "blah"), "must specify a value for f")

  }

  test("@Positive") {
    def parseP(args: Map[String,String], msg: String) {
      parse(args, msg){new PositiveArgs()}
    }

    parseP(Map("c" -> "1.0"), "must specify a positive value for a")
    parseP(Map("a" -> "3"), "must specify a positive value for c")
    parseP(Map("a" -> "3", "c" -> "-3.8"), "must specify a positive value for c")
    parseP(Map("a" -> "-3", "c" -> "3.8"), "must specify a positive value for a")
    parseP(Map("a" -> "0", "c" -> "1"), "must specify a positive value for a")

    val a = new PositiveArgs()
    a.parse(Map("a" -> "1", "c" -> "7.9"))
    a.a should be (1)
    a.c should be (7.9f)
  }


  test("@FileExists") {

    val tmpFile = File.createTempFile("file",".tmp")
    val tmpPath = tmpFile.getAbsolutePath
    tmpFile.deleteOnExit()

    def parseF(args: Map[String,String], msg: String) {
      parse(args, msg){new FileExistsArgs()}
    }

    parseF(Map("path" -> "fakeFile.tmp", "file" -> tmpPath), "must specify a file that exists for path, current value = fakeFile.tmp")
    parseF(Map("path" -> tmpPath, "file" -> "fakeFile.tmp"), "must specify a file that exists for file, current value = fakeFile.tmp")
    parseF(Map("path" -> null, "file" -> tmpPath), "must specify a valid file name for path")

    val a = new FileExistsArgs()
    a.parse(Map("path" -> tmpPath, "file" -> tmpPath))
    a.file should be(tmpFile)
    a.path should be(tmpPath)


  }


  test("@Range") {
    def parseR(args: Map[String,String], msg: String) {
      parse(args, msg) {new RangeArgs()}
    }

    val msgX = "must specify a value between 3.0 and 8.0 for x"
    parseR(Map("y" -> "-80"), msgX)
    parseR(Map("x"->"1", "y" -> "-80"), msgX)
    parseR(Map("x" -> "9", "y" -> "-80"), msgX)
    val msgY = "must specify a value between -83.0 and -72.0 for y"
    parseR(Map("x" -> "5"), msgY)
    parseR(Map("x" -> "5", "y" -> "5"), msgY)
    parseR(Map("x" -> "5", "y" -> "-90"), msgY)

    val a = new RangeArgs()
    a.parse(Map("x"->"4", "y" -> "-77"))
    a.x should be (4)
    a.y should be (-77)
  }

  test("user-defined") {
    //silly example of user-defined annotation validations
    parse(Map("x" -> "7"), "x must be 3 or 4"){new UserDefinedAnnotationArgs()}
    val a1 = new UserDefinedAnnotationArgs()
    a1.parse(Map("x" -> "3"))
    a1.x should be (3)

    //this arg class hasn't registered any validation w/ the annotation, so it is irrelevant
    val a2 = new UnregisteredAnnotationArgs()
    a2.parse(Map("x" -> "7"))
    a2.x should be (7)

    val a3 = new UserDefinedAnnotationUpdateArgs()
    a3.parse(Map("x" -> "17"))
    a3.x should be (3)

    a3.parse(Map("x" -> "4"))
    a3.x should be (4)
  }

  test("multi-annotation") {
    parse(Map("b"->"-4"), "must specify a positive value for b"){new MultiAnnotationArgs}
    parse(Map("b"->"7"), "must specify a value for b"){new MultiAnnotationArgs}

    val a = new MultiAnnotationArgs()
    a.parse(Map("b" -> "3"))
    a.b should be (3)
  }

  test("nested validation") {
    val b = new BarArgs()
    b.parse(Array[String]("--bar.foo", "hi"))
    b.bar.foo should be ("hi")

    val b2 = new BarArgs()
    val exc = the[ArgException] thrownBy  {b2.parse(Array[String]())}
    exc.getMessage should include("must specify a value for bar.foo")


    //make sure the args dont' get muddled at all if a nested arg has the same name
    val o = new OuterRequired()
    o.parse(Array("--x", "6", "--inner.x", "7"))
    o.x should be (6)
    o.inner.x should be (7)

    def t(s:String*): ArgException = {
      val a = new OuterRequired()
      the[ArgException] thrownBy {a.parse(s.toArray)} 
    }

    val exc1 = t()
    val exc2 = t("--x", "6")
    val exc3 = t("--inner.x", "7")
    val exc4 = t("--x", "1","--inner.x", "7")
    val exc5 = t("--x", "5","--inner.x", "567")
    exc1.getMessage should include ("must specify a value for")
    Seq(exc2,exc5).foreach{_.getMessage should include ("must specify a value for inner.x")}
    Seq(exc3,exc4).foreach{_.getMessage should include ("must specify a value for x")}

    val o2 = new OuterRequired()
    o2.parse(Array[String]("--x", "567","--inner.x", "1"))
    o2.x should be (567)
    o2.inner.x should be (1)
  }
}

class IntRequiredArgs extends FieldArgs {
  @Required
  var a: Int = _
  @Required
  var b: Int = 7
  var c = 19
}

class StringRequiredArgs extends FieldArgs {
  @Required
  var e: String = _
  @Required
  var f: String = "blah"
}

class PositiveArgs extends FieldArgs {
  @Positive
  var a: Int = _
  var b: Int = _
  @Positive
  var c: Float = _
  var d: Float = _
}

class RangeArgs extends FieldArgs {
  @Range(min=3,max=8)
  var x: Int = _
  @Range(min= -83, max= -72)
  var y: Float = _
}


class MultiAnnotationArgs extends FieldArgs {
  @Positive @Required
  var b = 7
}

class UserDefinedAnnotationArgs extends FieldArgs {
  @ThreeOrFour
  var x: Int = _

  registerAnnotationValidation(classOf[ThreeOrFour]){(_, value, _, name) =>
    if (value != 3 && value != 4) {
      throw new ArgException(name + " must be 3 or 4")
    }
  }
}

class UserDefinedAnnotationUpdateArgs extends FieldArgs {
  @ThreeOrFour
  var x: Int = _
  registerAnnotationValidationUpdate(classOf[ThreeOrFour]){(_, value, _, name, holder) =>
    if (value != 3 && value !=4) {
      holder.setValue(3)
    }
  }
}

class UnregisteredAnnotationArgs extends FieldArgs {
  @ThreeOrFour
  var x: Int = _
}
class FileExistsArgs extends FieldArgs {
  @FileExists
  var path: String = _

  @FileExists
  var file: File = _
}

class FooArgs extends FieldArgs {
  @Required
  var foo: String = _
}

class BarArgs extends FieldArgs {
  var bar = new FooArgs()
}

class OuterRequired extends FieldArgs {
  @Required
  var x = 1

  var inner: InnerRequired = _
}

class InnerRequired extends FieldArgs {
  @Required
  var x = 567
}
