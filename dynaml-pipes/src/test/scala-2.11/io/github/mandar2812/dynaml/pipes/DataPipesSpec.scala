package io.github.mandar2812.dynaml.pipes

import org.scalatest.{FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer

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
    val pipe1 = DataPipe((x: Int) => x + 1)
    val pipe2 = DataPipe((y: Int) => y/2.0)
    val pipe3 = DataPipe((y: Int) => y * 2)

    val p = pipe1 > pipe2

    val bifurcation = DataPipe((x: Int) => (x + 1, x * 2))

    val other_bifurcation = pipe1 >-< pipe3

    val add_num = DataPipe2((x: Int, y: Int) => x + y)
    val add_tup = DataPipe((xy: (Int, Int)) => xy._1 + xy._2)


    val final_pipe = other_bifurcation > add_tup
    val final_pipe2 = bifurcation > add_num

    val trip = DataPipe((x: Int) => (x+1, x+2, x+3))
    val pipe_3 = DataPipe3((x: Int, y: Int, z: Int) => x + y + z)

    val proc3 = trip > pipe_3

    assert(proc3(1) == 9)

    val quad = DataPipe((x: Int) => (x+1, x+2, x+3, x+4))
    val pipe_4 = DataPipe4((x: Int, y: Int, z: Int, u: Int) => x + y + z + u)

    val proc4 = quad > pipe_4

    assert(proc4(1) == 14)

    assert(bifurcation(1) == (2, 2) && bifurcation(1) == other_bifurcation(1, 1) && final_pipe(1, 1) == final_pipe2(1))

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

  "Iterable and Stream Data Pipes" should " be created, composed and applied correctly" in {


    val data_max = 20
    val numbers = 1 to data_max

    val buffer = ArrayBuffer[Int]()

    val filterPipe = DataPipe[Int, Boolean](_ % 2 == 0)
    val filterFunc = (n: Int) => n % 2 == 0

    val mapPipe    = DataPipe[Int, Int](_ * 2)
    val mapFunc    = (n: Int) => n*2

    val sideEff    = DataPipe[Int]((i: Int) => buffer.append(Seq(i):_*))
    val sideEffFun = (n: Int) => buffer.append(n)

    val flatMap    = DataPipe[Int, Iterable[Int]](1 to _)
    val flatMapFun = (n: Int) => 1 to n

    val flatMapS    = DataPipe[Int, Stream[Int]](n => (1 to n).toStream)
    val flatMapFunS = (n: Int) => (1 to n).toStream

    assert(
      IterableDataPipe(filterPipe).run(numbers).forall(_ % 2 == 0) &&
        IterableDataPipe(filterFunc).run(numbers).forall(_ % 2 == 0))

    val st_pipe = IterableDataPipe(filterPipe) > StreamDataPipe.toStreamPipe[Int, Iterable[Int]]

    assert(st_pipe.run(numbers).forall(_ % 2 == 0))

    assert(
      IterablePartitionPipe(filterPipe).run(numbers)._2.forall(_ % 2 == 1) &&
        IterablePartitionPipe(filterFunc).run(numbers)._2.forall(_ % 2 == 1))

    assert(
      IterableDataPipe(mapPipe).run(numbers).forall(_ % 2 == 0) &&
        IterableDataPipe(mapFunc).run(numbers).forall(_ % 2 == 0))

    assert(IterableFlatMapPipe(flatMap).run(numbers).sum == (1 to data_max).map(i => i*(i + 1)/2).sum)
    assert(IterableFlatMapPipe(flatMapFun).run(numbers).sum == (1 to data_max).map(i => i*(i + 1)/2).sum)

    IterableDataPipe(sideEff).run(numbers)
    assert(buffer.forall(numbers.contains))

    buffer.clear()

    IterableDataPipe(sideEffFun).run(numbers)
    assert(buffer.forall(numbers.contains))

    buffer.clear()


    assert(
      StreamDataPipe(filterPipe).run(numbers.toStream).forall(_ % 2 == 0) &&
        StreamDataPipe(filterFunc).run(numbers.toStream).forall(_ % 2 == 0))

    val it_pipe = StreamDataPipe(filterPipe) > IterableDataPipe.toIterablePipe[Int, Stream[Int]]

    assert(it_pipe.run(numbers.toStream).forall(_ % 2 == 0))

    assert(
      StreamPartitionPipe(filterPipe).run(numbers.toStream)._2.forall(_ % 2 == 1) &&
        StreamPartitionPipe(filterFunc).run(numbers.toStream)._2.forall(_ % 2 == 1))

    assert(
      StreamDataPipe(mapPipe).run(numbers.toStream).forall(_ % 2 == 0) &&
        StreamDataPipe(mapFunc).run(numbers.toStream).forall(_ % 2 == 0))

    assert(StreamFlatMapPipe(flatMapS).run(numbers.toStream).sum == (1 to data_max).map(i => i*(i + 1)/2).sum)
    assert(StreamFlatMapPipe(flatMapFunS).run(numbers.toStream).sum == (1 to data_max).map(i => i*(i + 1)/2).sum)

    StreamDataPipe(sideEff).run(numbers.toStream)
    assert(buffer.forall(numbers.contains))

    buffer.clear()

    StreamDataPipe(sideEffFun).run(numbers.toStream)
    assert(buffer.forall(numbers.contains))

    buffer.clear()

  }

}
