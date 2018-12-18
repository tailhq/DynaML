package io.github.mandar2812.dynaml.utils.sumac

import org.scalatest.FunSuite
import org.scalatest.Matchers

import scala.concurrent.duration._
import scala.collection._
import java.io.File
import java.util.{Calendar, Date, TimeZone}
import java.text.SimpleDateFormat

//import io.github.mandar2812.dynaml.utils.sumac.MyEnum

import scala.reflect.ClassTag


class ParserTest extends FunSuite with Matchers {

  test("SimpleParser") {
    checkParseAndBack(StringParser, "ooga", "ooga")
    IntParser.parse("5") should be (5)
    DoubleParser.parse("5") should be (5.0)
    DoubleParser.parse("1e-10") should be (1e-10)
    BooleanParser.parse("false") should be (false)
    BooleanParser.parse("true") should be (true)
    DurationParser.parse("10.seconds") should be (10 seconds)
    checkParseAndBack(DurationParser, "10 seconds", 10 seconds)
    checkParseAndBack(DurationParser, "10 minutes", 10 minutes)
    FiniteDurationParser.parse("3.days") should be (3 days)
    checkParseAndBack(FiniteDurationParser, "3 days", 3 days)
  }


  test("valueAsString") {
    //just make sure it works on primitives
    class Foo extends FieldArgs {
      var x: Int = 5
    }
    val f = new Foo()
    f.getStringValues should be (Map("x" -> "5"))
  }

  def checkParseAndBack(p: SimpleParser[_], s: String, v: AnyRef) {
    p.parse(s) should be (v)
    p.valueAsString(v, null) should be (s)
  }

  def checkReparse(p: SimpleParser[_ <: AnyRef], s: String) {
    val v = p.parse(s)
    val s2 = p.valueAsString(v, null)
    val v2 = p.parse(s2)
    v2 should be (v)
  }

  test("FileParser") {
    val homeDir = System.getProperty("user.home")
    FileParser.parse("~/foo") should be (new java.io.File(homeDir, "foo"))
    try {
      val winHomeDir = "c:\\users\\foo"
      System.setProperty("user.home", winHomeDir)
      FileParser.parse("~/foo") should be (new java.io.File(winHomeDir, "foo"))
      checkReparse(FileParser, "~/foo")
    } finally {
      System.setProperty("user.home", homeDir)
    }
    val cwd = System.getProperty("user.dir")
    FileParser.parse("ooga").getAbsolutePath should be (new java.io.File(cwd, "ooga").getAbsolutePath)
    checkReparse(FileParser, "ooga")
    checkReparse(FileParser, "~/ooga")
  }

  def collectionCheck[A <: FieldArgs : ClassTag,R](args: A, builder: Seq[Duration] => R) {
    val in = "10 seconds, 15.seconds, 30 minutes"
    val out = "10 seconds,15 seconds,30 minutes"
    val exp = builder(Seq(10 seconds, 15 seconds, 30 minutes))
    args.parse(Array("--x", in))

    val act = args.getArgs("").find{_.getName == "x"}.get.getCurrentValue
    act should be (exp)
    args.getStringValues("x") should be (out)

  }

  test("ListParser") {
    //Note this doesn't work w/ primitive types now, b/c its based on java reflection

    //Is there is better way to get a handle on parameterized types????
    val field = classOf[ContainerA].getDeclaredField("boundaries")
    val parsed = ParseHelper.parseInto("a,b,cdef,g", field.getGenericType, "dummy")
    parsed should be (Some(ValueHolder(List("a", "b", "cdef", "g"), field.getGenericType)))

    class ListTestClass extends FieldArgs {
      var x: List[Duration] = _
    }

    collectionCheck[ListTestClass, List[Duration]](new ListTestClass(), List.apply[Duration] _)
  }

  test("OptionParser") {
    //Doesn't work with primitive types, same problem as ListParser?
    val stringOptType = classOf[ContainerOption].getDeclaredField("string").getGenericType
    val listOptType = classOf[ContainerOption].getDeclaredField("listOfString").getGenericType
    OptionParser.parse("foo",stringOptType , null) should be (Some("foo"): Option[String])
    OptionParser.parse(null, stringOptType, null) should be (None: Option[String])
    OptionParser.parse(Parser.nullString, stringOptType, null) should be (None: Option[String])

    OptionParser.parse("a,b,cdef,g", listOptType, null) should be (Some(List("a", "b", "cdef", "g")): Option[List[String]])

    OptionParser.valueAsString(Some("foo"), stringOptType) should be ("foo")
    OptionParser.valueAsString(Some(List("a", "b", "cdef")), listOptType) should be ("a,b,cdef")
    OptionParser.valueAsString(None, stringOptType) should be (Parser.nullString)
  }

  test("ParseHelper") {
    ParseHelper.parseInto("ooga", classOf[String], "dummy") should be (Some(ValueHolder("ooga", classOf[String])))
    ParseHelper.parseInto("5.6", classOf[Double], "dummy") should be (Some(ValueHolder(5.6, classOf[Double])))
    ParseHelper.parseInto("5.6", classOf[String], "dummy") should be (Some(ValueHolder("5.6", classOf[String])))
    ParseHelper.parseInto("abc", classOf[RandomUnknownClass], "dummy") should be (None)
  }

  test("array parser") {
    class A extends FieldArgs {
      var x: Array[Duration] = _
    }
    collectionCheck(new A(), Array.apply[Duration] _)
  }

  test("traversable parser") {
    class A extends FieldArgs {
      var x: Traversable[Duration] = _
    }
    collectionCheck(new A(), Traversable.apply[Duration] _)
  }

  test("seq parser") {
    //note that these are all the types in scala.collection, NOT the ones in scala.predef
    class A extends FieldArgs {
      var x: Seq[Duration] = _
    }
    val a = new A()
    collectionCheck(new A(), Seq.apply[Duration] _)
  }

  test("vector parser"){
    class B extends FieldArgs {
      var x: Vector[Duration] = _
    }
    val b = new B()
    collectionCheck(new B(), Vector.apply[Duration] _)
  }

  test("map parser") {
    class A extends FieldArgs {
      var x: Map[File,Duration] = _
    }

    val a = new A()
    a.parse(Array("--x", "/blah/ooga:10 seconds,/foo/bar:1 hour"))
    a.x should be (Map(
      new File("/blah/ooga") -> (10 seconds),
      new File("/foo/bar") -> (1 hour)
    ))
    a.getStringValues should be (Map("x" -> "/blah/ooga:10 seconds,/foo/bar:1 hour"))

    val ex = the[IllegalArgumentException] thrownBy  {a.parse(Array("--x", "adfadfdfa"))}
    ex.getCause.getMessage should include ("'adfadfdfa' cannot be parsed. Caused by: `:' expected but end of source found")
  }

  test("map of Seq parser") {
    class A extends FieldArgs {
      var complex: Map[String, Seq[String]] = _
    }

    val a = new A()

    a.parse(Array("--complex", "key1:'v1,v2,v3',key2:'v4,v5,v6'"))
    a.complex should be (Map(
      "key1" -> Seq("v1", "v2", "v3"),
      "key2" -> Seq("v4", "v5", "v6")
    ))
  }
  
  test("date parser") {
    def checkDateAndCalendar(parser:Parser[_], s:String, m: Int) {
      val d = parser.parse(s, classOf[Date], null).asInstanceOf[Date]
      d.getMonth() should be (m)
      val c = parser.parse(s, classOf[Calendar], null).asInstanceOf[Calendar]
      c.get(Calendar.MONTH) should be (m)

      //value as string should use unambiguous canonical form always
      parser.valueAsString(d, null) should be ("2013-11-12")
      parser.valueAsString(c, null) should be ("2013-11-12")
    }
    checkDateAndCalendar(USDateParser,"11/12/2013", 10)
    checkDateAndCalendar(StandardDateParser, "12/11/2013", 10)

    class A extends FieldArgs {
      registerParser(USDateParser)
      var x: Date = _
    }
    val a = new A()

    class B extends FieldArgs {
      registerParser(USDateParser)
      var x: Calendar = _
    }
    val b = new B()


    //using java time classes is a serious pain ...
    val tz = TimeZone.getTimeZone("UTC")
    val format = new SimpleDateFormat("yyyy-MM-dd")
    format.setTimeZone(tz)
    val d = format.parse("2013-12-26")

    Seq("2013-12-26", "2013/12/26", "12-26-2013", "12/26/2013").foreach{p =>
      withClue(p){
        a.parse(Array("--x", p))
        a.x should be (d)

        b.parse(Array("--x", p))
        b.x.getTimeInMillis should be (d.getTime)
        b.x.getTimeZone should be (tz)

      }
    }
  }

  test("enum parser") {
    class A extends FieldArgs {
      var x: MyEnum = _
    }
    val a = new A()

    a.parse(Array("--x", "Abigail"))
    a.x should be (MyEnum.Abigail)
    a.getStringValues should be (Map("x" -> "Abigail"))



    val ex = the[ArgException] thrownBy {a.parse(Array("--x", "foobar"))}
    ex.getMessage should include ("foobar is not in set of enum values: " + MyEnum.values.mkString(","))
  }

}

class RandomUnknownClass

class ContainerA extends FieldArgs {
  var title: String = _
  var count: Int = _
  var boundaries: List[String] = _
}

class ContainerOption(val string: Option[String], val listOfString: Option[List[String]])
