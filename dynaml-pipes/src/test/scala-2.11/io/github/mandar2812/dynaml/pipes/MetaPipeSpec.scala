package io.github.mandar2812.dynaml.pipes

import org.scalatest.{FlatSpec, Matchers}

class MetaPipeSpec extends FlatSpec with Matchers {

  "Meta pipes" should " extend the idea of curried functions, compose with other pipes" in {

    val mult_by_num = MetaPipe((n: Int) => (x: Int) => n*x)

    val add_one = DataPipe[Int, Int](_ + 1)


    val mult_by_num_add_one = mult_by_num >> add_one

    val linear_transform = MetaPipe21((m: Int, c: Int) => (x: Int) => m*x + c)

    val square_num = DataPipe[Int, Int](n => n*n)

    val linear_then_square = linear_transform >> square_num

    assert(mult_by_num(2)(1) == 2 && mult_by_num_add_one(3)(1) == 4)
    assert(linear_transform(2, 2)(2) == 6 && linear_then_square(2, 1)(2) == 25)

  }

}
