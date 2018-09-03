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

  "Tuple Integer Encoding" should "create a valid bijective mapping from tuples to number systems" in {

    val binaryNumbers = List(2, 2)

    val numbers = 0 until 4

    val enc = TupleIntegerEncoder(binaryNumbers)

    assert(Seq(List(0, 0), List(1, 0), List(0, 1), List(1, 1)).map(enc(_)).zip(numbers).forall(ab => ab._1 == ab._2))

  }

}
