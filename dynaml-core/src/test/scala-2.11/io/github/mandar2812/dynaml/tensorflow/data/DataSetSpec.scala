package io.github.mandar2812.dynaml.tensorflow.data

import io.github.mandar2812.dynaml.pipes.{DataPipe, DataPipe2}
import org.scalatest.{FlatSpec, Matchers}
import io.github.mandar2812.dynaml.tensorflow._

class DataSetSpec extends FlatSpec with Matchers {

  "Data Sets " should " execute canonical transformations faithfully" in {

    val max = 10

    val numbers = dtfdata.dataset(1 to max)

    val numbers_rev = dtfdata.dataset(max to 1)

    val even_numbers = numbers.filter(_ % 2 == 0)

    val ev_num       = numbers.filter(DataPipe[Int, Boolean](_ % 2 == 0))

    assert(even_numbers.size == max/2 && even_numbers.data.forall(_ % 2 == 0))
    assert(ev_num.size == max/2 && ev_num.data.forall(_ % 2 == 0))

    val (odd_numbers, odd_num) = (numbers.filterNot(_ % 2 == 0), numbers.filterNot(DataPipe[Int, Boolean](_ % 2 == 0)))

    assert(odd_numbers.size == max/2 && odd_numbers.data.forall(_ % 2 == 1))
    assert(odd_num.size == max/2 && odd_num.data.forall(_ % 2 == 1))

    val doubleofnumbers = numbers.map((i: Int) => 2*i)

    assert(doubleofnumbers.data.forall(_ % 2 == 0) && numbers.zip(doubleofnumbers).data.forall(c => c._2 == c._1 * 2))

    val addPipe = DataPipe2[Int, Int, Int]((x, y) => x + y)

    assert(numbers.reduce[Int](addPipe) == numbers.size*(numbers.size + 1)/2)

    assert(numbers.reduceLeft[Int](addPipe) == numbers.size*(numbers.size + 1)/2)

    val running_sum = numbers.scanLeft[Int](0)(addPipe)

    assert(running_sum.data.zipWithIndex.forall(c => c._1 == c._2*(c._2 + 1)/2))

    val run_sum = running_sum.data.toSeq

    assert(numbers.scan[Int](0)(addPipe).data.forall(run_sum.contains(_)))

    assert(numbers_rev.scanRight[Int](0)(addPipe).data.zip(max to 1).forall(c => c._1 == c._2*(c._2 + 1)/2))

    assert(numbers.grouped(2).data.forall(_.length == 2))

    assert(
      numbers.transform(DataPipe[Iterable[Int], Iterable[Int]](c => Iterable(c.sum))).data.head ==
        numbers.size*(numbers.size + 1)/2)

  }

}
