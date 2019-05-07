package io.github.mandar2812.dynaml.tensorflow.data

import io.github.mandar2812.dynaml.pipes.{DataPipe, DataPipe2}
import io.github.mandar2812.dynaml.DynaMLPipe._
import org.scalatest.{FlatSpec, Matchers}
import io.github.mandar2812.dynaml.tensorflow._
import org.platanios.tensorflow.api._

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

    val doubleofnumbers = numbers.map(DataPipe((i: Int) => 2*i))

    val tripleofnumbers = numbers.map(DataPipe((i: Int) => 3*i))

    assert(
      doubleofnumbers.data.forall((p: Int) => p % 2 == 0) &&
        numbers.map(DataPipe((i: Int) => 2*i)).data.forall(_ % 2 == 0) &&
        numbers.zip(doubleofnumbers).data.forall(c => c._2 == c._1 * 2) &&
        ZipDataSet(numbers, doubleofnumbers).data.forall(c => c._2 == c._1 * 2) &&
        numbers.zip(doubleofnumbers)
          .join(numbers.zip(tripleofnumbers))
          .data.forall(c => c._2._1 == 2*c._1 && c._2._2 == 3*c._1)
    )

    val addPipe = DataPipe2[Int, Int, Int]((x, y) => x + y)
    val addPipe2 = DataPipe[Iterable[Int], Int](_.sum)


    assert(
      numbers.reduce[Int](addPipe) == numbers.size*(numbers.size + 1)/2 &&
        numbers.reduce[Int](addPipe2) == numbers.size*(numbers.size + 1)/2)

    assert(numbers.transform(DataPipe((d: Iterable[Int]) => d.map(2 * _))).data.forall(_ % 2 == 0))

    assert(numbers.reduceLeft[Int](addPipe) == numbers.size*(numbers.size + 1)/2)

    val running_sum = numbers.scanLeft[Int](0)(addPipe)

    assert(running_sum.data.zipWithIndex.forall(c => c._1 == c._2*(c._2 + 1)/2))

    val run_sum = running_sum.data.toSeq

    assert(numbers.flatMap(DataPipe[Int, Iterable[Int]](i => 1 to i)).size == numbers.size*(numbers.size + 1)/2)
    assert(numbers.flatMap(DataPipe[Int, Iterable[Int]](1 to _)).size == numbers.size*(numbers.size + 1)/2)

    assert(numbers.scan[Int](0)(addPipe).data.forall(run_sum.contains(_)))

    assert(numbers_rev.scanRight[Int](0)(addPipe).data.zip(max to 1).forall(c => c._1 == c._2*(c._2 + 1)/2))

    assert(numbers.grouped(2).data.forall(_.length == 2))

    assert(
      numbers.transform(DataPipe[Iterable[Int], Iterable[Int]](c => Iterable(c.sum))).data.head ==
        numbers.size*(numbers.size + 1)/2)

    assert(odd_numbers.concatenate(even_numbers).size == max)

    val zip_numbers = numbers.zip(numbers)

    val (f, t) = zip_numbers.unzip

    assert(f.size == max && t.size == max)

    val supervisedDataSet = zip_numbers.to_supervised(identityPipe[(Int, Int)])

    assert(supervisedDataSet.features.data.zipWithIndex.forall(c => c._1 == c._2 + 1))
    assert(supervisedDataSet.targets.data.zipWithIndex.forall(c => c._1 == c._2 + 1))
    assert(dtfdata.supervised_dataset(numbers, numbers).unzip._2.size == numbers.size)


    val tr_te = supervisedDataSet.partition(DataPipe[(Int, Int), Boolean](c => c._1 % 2 == 0))

    assert(
      tr_te.training_dataset.data.forall(c => c._1 % 2 == 0) &&
        tr_te.test_dataset.data.forall(c => c._1 % 2 == 1))

    val tr_te2 = numbers.partition(DataPipe[Int, Boolean](_ % 2 == 0))

    assert(
      tr_te2.training_dataset.data.forall(_ % 2 == 0) &&
        tr_te2.test_dataset.data.forall(_ % 2 == 1))

    val coll_data = dtfdata.dataset.collect(Seq(numbers, numbers))

    assert(coll_data.data.zipWithIndex.forall(p => p._1 == Seq.fill(2)(p._2 + 1)))

    val coll_sup_data = dtfdata.supervised_dataset.collect(
      Seq(
        dtfdata.supervised_dataset(numbers, numbers),
        dtfdata.supervised_dataset(numbers, numbers))
    )

    assert(coll_sup_data.data.forall(p => p._1 == p._2))

  }

  "DynaML data sets" should " build Tensor Flow data sets properly" in {

    val max = 10

    val numbers = dtfdata.dataset(1 to max)

    val tf_data1 = numbers.build(
      DataPipe[Int, Tensor[Int]](i => Tensor(i).reshape(Shape(1))),
      INT32, Shape(1))

    /*val tf_data2 = numbers.build(
      DataPipe[Int, Output[Int]](i => Tensor[Int](i).toOutput),
      INT32,
      Shape(1)
    )*/

    val tf_data3 = numbers.build_buffered(
      2,
      DataPipe[Seq[Int], Tensor[Int]](x => dtf.tensor_i32(x.length)(x:_*)),
      INT32, Shape(1)
    )

    assert(
      tf_data1.outputDataTypes[INT32] == INT32 &&
        tf_data1.outputShapes[Shape] == Shape(1) &&
        tf_data1.createInitializableIterator[INT32, Shape]().next() != null)



    /*assert(
      tf_data2.outputDataTypes == INT32 &&
        tf_data2.outputShapes == Shape(1) &&
        tf_data2.createInitializableIterator().next() != null)*/

    assert(
      tf_data3.outputDataTypes[INT32] == INT32 &&
        tf_data3.outputShapes[Shape] == Shape() &&
        tf_data3.createInitializableIterator[INT32, Shape]().next() != null)

  }

}
