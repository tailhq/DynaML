package io.github.mandar2812.dynaml.utils.sumac

import org.scalatest.FunSuite
import org.scalatest.Matchers
import java.io._

class ArgBuilderTest extends FunSuite with Matchers {

  val nullOutput = new PrintStream(new OutputStream {
    def write(p1: Int) {}
  })

  test("prompting") {
    val args = new BuilderTestArgs()
    val input = fixedInputStream(
      //no name
      "","",
      //count = 5
      "5",
      //no property file
      "",""
    )
    //I'm not actually checking what the prompts are here ...
    ArgBuilder.promptForArgs(args, input, nullOutput)
    args.count should be (5)
    args.name should be ("Here's a name")
    args.propertyFile should be (null)
  }

  val dir = new File("test_output/" + getClass.getSimpleName)
  dir.mkdirs()

  test("prompting & saving") {
    val args = new BuilderTestArgs()
    val propFile = new File(dir, "arg_builder_test_output.properties").getAbsolutePath
    val input = fixedInputStream(
      //empty string name
      "","\"\"",
      //count = 5
      "5",
      //property file
      propFile
    )
    ArgBuilder.promptForArgs(args, input, nullOutput)
    args.saveConfig()

    val args2 = new BuilderTestArgs()
    args2.propertyFile = new File(propFile)
    args2.parse(Array[String]())
    args2.name should be ("")
    args2.count should be (5)
    args2.propertyFile should be (new File(propFile))
  }

  def fixedInputStream(lines: String*): BufferedReader = {
    //THIS IS REALLY UGLY.  it only works b/c I know I only call readLine on the Buffered Reader
    val itr = lines.iterator
    new BufferedReader(new Reader() {
      def close() {}

      def read(p1: Array[Char], p2: Int, p3: Int) = 0
    }) {
      override def readLine(): String = if (itr.hasNext) itr.next else null
    }
  }
}

class BuilderTestArgs extends FieldArgs with PropertiesConfig {
  var name = "Here's a name"
  var count: Int = _
}
