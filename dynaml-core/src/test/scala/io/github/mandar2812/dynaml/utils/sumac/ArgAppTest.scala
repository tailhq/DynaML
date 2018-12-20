package io.github.mandar2812.dynaml.utils.sumac

import org.scalatest.FunSuite
import org.scalatest.Matchers

class ArgAppTest extends FunSuite with Matchers {

  test("getArgumentClass") {
    val m = new MyApp()
    m.getArgumentClass should be (classOf[MyArgs])

    val m2 = new MyNestedArgApp()
    m2.getArgumentClass should be (classOf[MyArgs])
  }

  test("main") {
    val m = new MyApp()
    m.main(Array("--a", "hello", "--b", "17"))

    val m2 = new MyNestedArgApp()
    m2.main(Array("--a", "byebye", "--b", "3"))
  }

}

class MyApp extends Dummy with ArgApp[MyArgs] with Matchers {
  def main(args: MyArgs) {
    args.a should be ("hello")
    args.b should be (17)
  }
}

trait Dummy


trait NestedArgMain extends ArgMain[MyArgs] {
  def blah(x: Int) = x + 5
}

class MyNestedArgApp extends NestedArgMain with Matchers {
  def main(args: MyArgs) {
    args.a should be ("byebye")
    args.b should be (3)
  }
}
