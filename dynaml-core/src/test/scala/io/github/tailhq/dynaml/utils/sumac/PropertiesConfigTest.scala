package io.github.tailhq.dynaml.utils.sumac

import org.scalatest.FunSuite
import java.io.{PrintWriter, File}
import java.util.Properties
import org.scalatest.Matchers

class PropertiesConfigTest extends FunSuite with Matchers {

  val testOutDir = new File("test_output/" + getClass.getSimpleName)
  testOutDir.mkdirs()

  test("load properties") {
    val propFile = new File(testOutDir, "load_properties_test.properties")
    val p = new Properties()
    p.put("x", "98")
    p.put("blah", "ooga booga")
    val out = new PrintWriter(propFile)
    p.store(out,null)
    out.close()


    val args = new PropertyArgs()
    args.parse(Array("--propertyFile", propFile.getAbsolutePath))
    args.x should be (98)
    args.blah should be ("ooga booga")
  }

  test("roundtrip properties") {
    val propFile = new File(testOutDir, "roundtrip_properties_test.properties")
    val args = new PropertyArgs()
    args.x = 5
    args.wakka = 93.4f
    args.propertyFile = propFile
    args.saveConfig()

    val args2 = new PropertyArgs()
    args2.propertyFile = propFile
    args2.parse(Map[String,String]())
    args2.x should be (5)
    args2.wakka should be (93.4f)
    args2.blah should be (null)
  }


  class PropertyArgs extends FieldArgs with PropertiesConfig {
    var x: Int = _
    var blah: String = _
    var wakka: Float = _
  }

}


