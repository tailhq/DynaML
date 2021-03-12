package io.github.tailhq.dynaml.utils.sumac

import types.{SelectInput,MultiSelectInput}
import org.scalatest.FunSuite
import org.scalatest.Matchers
import java.lang.reflect.Type
import java.io.{ObjectInputStream, ByteArrayInputStream, ObjectOutputStream, ByteArrayOutputStream}
import io.github.tailhq.dynaml.utils.sumac.validation.Required

/**
 *
 */

class FieldArgsTest extends FunSuite with Matchers {

  test("parseStrings") {
    val o = new StringHolder(null, null) with FieldArgs
    o.parse(Array("--name", "hello"))
    o.name should be ("hello")
    o.parse(Array("--comment", "blah di blah blah"))
    o.name should be ("hello")
    o.comment should be ("blah di blah blah")
    o.parse(Array("--name", "ooga", "--comment", "stuff"))
    o.name should be ("ooga")
    o.comment should be ("stuff")
  }

  test("parseCommandLine") {
    val o = new StringHolder(null, null) with FieldArgs
    o.parse("--name hello")
    o.name should be ("hello")
    o.parse(
      """--name \
        goodbye
      """)
    o.name should be ("goodbye")
  }

  test("parseMixed") {
    val o = new MixedTypes(null, 0) with FieldArgs

    o.parse(Array("--name", "foo", "--count", "17"))
    o.name should be ("foo")
    o.count should be (17)
    o.parse(Array("--count", "-5"))
    o.name should be ("foo")
    o.count should be (-5)
  }

  test("subclass parsing") {
    val o = new Child(false, null, 0) with FieldArgs

    o.parse(Array("--flag", "true", "--name", "bugaloo"))
    o.name should be ("bugaloo")
    o.flag should be (true)
  }

  test("parseOptions") {
    val o = new ArgsWithOptions()

    // check default behavior
    o.parse(Array[String]())
    o.optStringNone should be (None: Option[String])
    o.optStringSome should be (Some("ooga"))
    o.optListStringNone should be (None: Option[List[String]])
    o.optListStringSome should be (Some(List("abc", "def", "ghi")))

    o.parse(Array("--optStringSome", "foo", "--optListStringSome", "x,y,zzz"))
    o.optStringNone should be (None: Option[String])
    o.optStringSome should be (Some("foo"))
    o.optListStringNone should be (None: Option[List[String]])
    o.optListStringSome should be (Some(List("x", "y", "zzz")))
  }

  test("help message") {
    val o = new StringHolder(null, null) with FieldArgs
    val exc1 = the[ArgException] thrownBy  {o.parse(Array("--xyz", "hello"))}
    //the format is still ugly, but at least there is some info there
    "\\-\\-name\\s.*String".r.findFirstIn(exc1.getMessage()) should be ('defined)
    "\\-\\-comment\\s.*String".r.findFirstIn(exc1.getMessage()) should be ('defined)

    val o2 = new MixedTypes(null, 0) with FieldArgs
    val exc2 = the[ArgException] thrownBy {o2.parse(Array("--foo", "bar"))}
    "\\-\\-name\\s.*String".r findFirstIn(exc2.getMessage) should be ('defined)
    "\\-\\-count\\s.*[Ii]nt".r findFirstIn(exc2.getMessage) should be ('defined)  //java or scala types, I'll take either for now

    val exc3 = the[ArgException] thrownBy  {o2.parse(Array("--count", "ooga"))}
    //this message really should be much better.  (a) the number format exception should come first and (b) should indicate that it was while processing the "count" argument
    "\\-\\-name\\s.*String".r findFirstIn(exc3.getMessage) should be ('defined)
    "\\-\\-count\\s.*[Ii]nt".r findFirstIn(exc3.getMessage) should be ('defined)  //java or scala types, I'll take either for now
  }

  test("error msg on unknown types") {
    //Note that FieldArgs will simply ignore fields with types that it doesn't support
    // (contrast with FieldArgsExceptionOnUnparseable below)
    val o = new SpecialTypes("", null) with FieldArgs

    //no exception if we only pass the args it knows about
    o.parse(Array("--name", "blah"))
    o.name should be ("blah")

    //it will throw an exception if we pass in an argument for a field that it didn't know what to do with
    val exc = the[ArgException] thrownBy  {o.parse(Array("--funky", ""))}
    exc.getMessage should include ("unknown option funky")

    //on the other hand, FieldArgsExceptionOnUnparseable will throw an exception no matter what we do
    val o2 = new SpecialTypes("", null) with FieldArgsExceptionOnUnparseable
    val exc2 = the[ArgException] thrownBy  {o2.parse(Array("--name", "blah"))}
    exc2.getMessage should include ("type")
    exc2.getMessage should include ("MyFunkyType")
  }


  test("good error msg") {
    val o = new MixedTypes("", 0) with FieldArgs

    val exc1 = the[ArgException] thrownBy  {o.parse(Array("--count", "hi"))}
    //don't actually need the message to look *exactly* like this, but extremely useful for it to at least say what it was trying to parse
    exc1.getMessage should startWith ("""Error parsing "hi" into field "count" (type = int)""")
  }

  test("set args") {
    case class SetArgs(var set: Set[String]) extends FieldArgs
    val s = new SetArgs(null)
    s.parse(Array("--set", "a,b,c,def"))
    s.set should be (Set("a", "b", "c", "def"))
  }

  test("help") {
    val s = new IgnoredArgs()
    val exc = the[ArgException] thrownBy  {s.parse(Array("--help"))}
    """unknown option""".r findFirstIn (exc.getMessage) should be ('empty)
    """\-\-x\s.*[Ii]nt""".r findFirstIn(exc.getMessage) should be ('defined)
  }

  test("selectInput") {
    case class SelectInputArgs(var select: SelectInput[String] = SelectInput("a", "b", "c")) extends FieldArgs
    val s = new SelectInputArgs()
    val id = System.identityHashCode(s.select)
    s.parse(Array("--select", "b"))
    s.select.value should be (Some("b"))
    System.identityHashCode(s.select) should be (id)
    s.select.options should be (Set("a", "b", "c"))

    an[ArgException] should be thrownBy {s.parse(Array("--select", "q"))}
  }

  test("selectInput order") {
    import util.Random._
    val max = 1000
    val orderedChoices = shuffle(1.to(max).map(_.toString))
    case class SelectInputArgs(var select: SelectInput[String] = SelectInput(orderedChoices:_*)) extends FieldArgs
    val s = new SelectInputArgs()
    val id = System.identityHashCode(s.select)

    val index = nextInt(max).toString
    s.parse(Array("--select", index))
    s.select.value should be (Some(index))
    System.identityHashCode(s.select) should be (id)
    s.select.options.toList should be (orderedChoices)

    an [ArgException] should be thrownBy  {s.parse(Array("--select", "q"))}
  }

  test("multiSelectInput") {
    case class MultiSelectInputArgs(var multiSelect: MultiSelectInput[String] = MultiSelectInput("a", "b", "c")) extends FieldArgs
    val s = new MultiSelectInputArgs()
    val id = System.identityHashCode(s.multiSelect)
    s.parse(Array("--multiSelect", "b"))
    s.multiSelect.value should be (Set("b"))
    System.identityHashCode(s.multiSelect) should be (id)
    s.multiSelect.options should be (Set("a", "b", "c"))

    s.parse(Array("--multiSelect", "b,c"))
    s.multiSelect.value should be (Set("b", "c"))

    an [ArgException] should be thrownBy  {s.parse(Array("--multiSelect", "q"))}
    an[ArgException] should be thrownBy  {s.parse(Array("--multiSelect", "b,q"))}
    an[ArgException] should be thrownBy  {s.parse(Array("--multiSelect", "q,b"))}

  }

  test("exclude scala helper fields") {

    {
      val m = new MixedTypes(null, 0) with FieldArgs
      val names = m.parser.nameToHolder.keySet
      names should be (Set("name", "count"))
    }


    {
      val s = new SomeApp()
      val names = s.getArgHolder.parser.nameToHolder.keySet
      names should be (Set("x", "y"))
    }

  }



  test("annotations") {
    val c = new ClassWithSomeAnnotations() with FieldArgs
    c.parser.nameToHolder.values.foreach { f =>
      f.getName match {
        case "foo" =>
          f.getDescription should be ("foo")
        case "ooga" =>
          f.getDescription should be ("this is an integer argument")
        case "" =>
          assert(false, "use variable name if no name given in annotation")
        case "x" => assert(false, "use name from annotation instead of variable name")
        case "y" =>
          f.getDescription should be ("another integer argument")
        case "z" =>
          assert(false, "use name from annotation instead of variable name")
        case "wakka" =>
          f.getDescription should be ("wakka")
      }
    }

    c.parse(Array("--foo", "hi", "--ooga", "17", "--y", "181", "--wakka", "1.81"))
    c.foo should be ("hi")
    c.x should be (17)
    c.y should be (181)
    c.z should be (1.81)

    an[ArgException] should be thrownBy {c.parse(Array("--x", "17"))}
    an[ArgException] should be thrownBy  {c.parse(Array("--z", "1"))}
  }


  test("custom parsers") {
    val c = new ArgsWithCustomType()
    c.parse(Array("--x", "7", "--y", "hithere:345","--z", "oogabooga"))
    c.x should be (7)
    c.y should be (CustomType("hithere", 345))
    c.z should be ("oogabooga")
  }

  test("validation") {
    val c = new ArgsWithValidation()
    c.parse(Array("--x", "-5", "--z", "23.5"))
    c.x should be (0)
    c.y should be ("ooga")
    c.z should be (23.5)

    c.parse(Array("--x", "17", "--y", "blah", "--z", "134"))
    c.x should be (17)
    c.y should be ("blah")
    c.z should be (134)

    an[Exception] should be thrownBy {c.parse(Array("--z", "0"))}
  }

  test("private fields ignored") {
    val c = new ArgsWithPrivateFields()

    c.parse(Array("--x","7"))
    c.x should be (7)

    c.parser.nameToHolder should not contain key ("q")
  }

  test("vals ignored") {
    val c = new ArgsWithVals()

    c.parse(Array("--x", "19"))
    c.x should be (19)

    c.parser.nameToHolder should not contain key ("y")
  }

  test("respect ignore annotation") {
    val c = new IgnoredArgs()
    c.parse(Array("--x", "123"))
    c.x should be (123)

    c.parser.nameToHolder should not contain key ("y")
  }

  test("getStringValues") {
    val c = new IgnoredArgs()
    c.x = 35245
    c.getStringValues should be (Map("x"-> "35245"))

    val a = new ArgsWithCustomType()
    a.x = 5
    a.y = CustomType("blah", 17)
    a.getStringValues should be (Map(
      "x" -> "5",
      "y" -> "blah:17",
      "z" -> Parser.nullString
    ))
  }

  test("nested args") {
    val args = new NestedArgs()
    val origFirstSet = args.firstSet
    //basic parsing of top level args
    args.parse(Map("x" -> "3"))
    args.x should be (3)
    //inner args should not be null, and also should *not* take values from unqualified names
    args.firstSet should not be (null)
    args.secondSet should not be (null)
    args.firstSet.x should be (0)
    args.secondSet.x should be (0)
    (args.firstSet eq origFirstSet) should be (true)  //if a nested arg is not null, don't replace it with a new object


    //now lets actually parse some nested args
    args.parse(Map("x" -> "91", "firstSet.x" -> "-32", "secondSet.x" -> "11"))
    args.x should be (91)
    args.firstSet.x should be (-32)
    args.secondSet.x should be (11)
  }

  test("args are serializable") {
    val args = new SomeArgs()
    args.parse(Array("--x", "1", "--y", "test"))
    val out = new ByteArrayOutputStream()
    new ObjectOutputStream(out).writeObject(args)
    out.close()
    val in = new ByteArrayInputStream(out.toByteArray)
    val deserializedArgs = new ObjectInputStream(in).readObject().asInstanceOf[SomeArgs]
    in.close()
    deserializedArgs.x should be (1)
    deserializedArgs.y should be ("test")
  }

  test("getDefaultValues") {

    {
      //args w/ explicit defaults
      val args = new SomeArgs()
      args.x = 17
      args.y = "blah"

      val defaultArgs = args.getDefaultArgs.map{a => a.getName -> a}.toMap
      defaultArgs("x").getCurrentValue should be (0)
      defaultArgs("y").getCurrentValue should be ("hello")
    }

    {
      //args w/ java default values
      val args2 = new ArgsWithValidation()
      args2.x = 98
      args2.y = "ooga"
      args2.z = 3.9
      val defaultArgs2 = args2.getDefaultArgs.map{a => a.getName -> a}.toMap
      defaultArgs2("x").getCurrentValue should be (0)
      defaultArgs2("y").getCurrentValue should be (null)
      defaultArgs2("z").getCurrentValue should be (0.0)
    }
  }

}


case class StringHolder(var name: String, var comment: String)

case class MixedTypes(var name: String, var count: Int)

//is there an easier way to do this in scala?
class Child(var flag: Boolean, name: String, count: Int) extends MixedTypes(name, count)

case class SpecialTypes(var name: String, var funky: MyFunkyType)

case class MyFunkyType(var x: String)


class SomeApp extends ArgApp[SomeArgs] {
  def main(args: SomeArgs) {}
}

class SomeArgs extends FieldArgs {
  var x: Int = 0
  var y: String = "hello"
}



class ClassWithSomeAnnotations {
  var foo: String = _
  @Arg(name="ooga", description="this is an integer argument")
  var x: Int = _
  @Arg(description="another integer argument")
  var y: Int = _
  @Arg(name="wakka")
  var z: Double = _
}

case class CustomType(val name: String, val x: Int)

object CustomTypeParser extends Parser[CustomType] {
  def canParse(tpe:Type) = {
    ParseHelper.checkType(tpe, classOf[CustomType])
  }
  def parse(s: String, tpe: Type, currentVal: AnyRef) = {
    val parts = s.split(":")
    CustomType(parts(0), parts(1).toInt)
  }
  override def valueAsString(currentVal: AnyRef, tpe: Type) = {
    val ct = currentVal.asInstanceOf[CustomType]
    ct.name + ":" + ct.x
  }
}

class ArgsWithCustomType extends FieldArgs {
  registerParser(CustomTypeParser)
  var x: Int = _
  var y: CustomType = _
  var z: String = _
}


class ArgsWithValidation extends FieldArgs {
  var x: Int = _
  var y: String = _
  var z: Double = _
  addValidation{
    if (x < 0)
      x = 0
    if (y == null)
      y = "ooga"
    if (z < 5)
      throw new RuntimeException("z must be greater than 5 -- was " + z)
  }
}

class ArgsWithPrivateFields extends FieldArgs {
  var x: Int = _
  private var q: Int = _
}

class ArgsWithVals extends FieldArgs {
  var x: Int = _
  val y = 18
}

class IgnoredArgs extends FieldArgs {
  var x: Int = _
  @Ignore
  var y: Int = _
}


class NestedArgs extends FieldArgs {
  var firstSet = new SomeArgs()
  var secondSet: SomeArgs = _
  var x: Int = 7
  var z: Float = _
}

class ArgsWithOptions extends FieldArgs {
  var optStringNone: Option[String] = None
  var optStringSome: Option[String] = Some("ooga")
  var optListStringNone: Option[List[String]] = None
  var optListStringSome: Option[List[String]] = Some(List("abc", "def", "ghi"))
}
