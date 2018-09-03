package io.github.mandar2812.dynaml.tensorflow.data

import io.github.mandar2812.dynaml.pipes.DataPipe2
import org.scalatest.{FlatSpec, Matchers}
import io.github.mandar2812.dynaml.tensorflow._

class DataSetSpec extends FlatSpec with Matchers {

  "Data Sets " should " execute map/flatMap/reduce/filter operations faithfully" in {

    val numbers = dtfdata.dataset(1 to 100)

    val even_numbers = numbers.filter(_ % 2 == 0)

    assert(even_numbers.size == 50 && even_numbers.data.forall(_ % 2 == 0))

    val doubleofnumbers = numbers.map((i: Int) => 2*i)

    assert(doubleofnumbers.data.forall(_ % 2 == 0) && numbers.zip(doubleofnumbers).data.forall(c => c._2 == c._1 * 2))

    val addPipe = DataPipe2[Int, Int, Int]((x, y) => x + y)

    assert(numbers.reduce[Int](addPipe) == numbers.size*(numbers.size + 1)/2)

  }

}
