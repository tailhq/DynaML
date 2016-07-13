package io.github.mandar2812.dynaml.pipes

import org.scalatest.{FlatSpec, Matchers}

/**
  * Created by mandar on 13/7/16.
  */
class DataPipesSpec extends FlatSpec with Matchers {

  "A Data Pipe" should "have type consistency" in {
    val pipe = DataPipe((x: Int) => x+1)

    val num = pipe(1)

    assert(num == 2)
    assert(num match {case _: Int => true})
  }

  "DataPipes" should "be consistent with respect to composition" in {
    val pipe1 = DataPipe((x: Int) => x+1)
    val pipe2 = DataPipe((y: Int) => y/2.0)

    val p = pipe1 > pipe2

    val num = p(0)

    assert(num == 0.5)
    assert(num match {case _: Double => true})
  }

}
