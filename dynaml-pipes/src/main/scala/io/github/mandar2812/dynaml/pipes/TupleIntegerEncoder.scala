package io.github.mandar2812.dynaml.pipes

/**
  * @author mandar2812 date 23/10/2016.
  *
  * Encodes tuples of integers to a single integer value.
  * Ex. List(1,0,1,0) -> 1*1 + 4*1 = 5 if arities = List(2,2,2,2)
  */
case class TupleIntegerEncoder(arities: List[Int]) extends Encoder[List[Int], Int] {

  assert(arities.forall(_ >= 2), "The arities must be greater than 1")

  override def run(data: List[Int]): Int = {
    val dataZipped = data zip arities
    assert(
      dataZipped.forall(c => c._1 < c._2),
      "Elements must be from 0 to p-1 where p is aritiy for the element")
    TupleIntegerEncoder.encodeAcc(dataZipped, 1, 0)
  }

  val i: DataPipe[Int, List[Int]] =
    DataPipe((n: Int) => TupleIntegerEncoder.decodeAcc(
      (n, arities.tail), arities.head,
      List(), arities.length))
}

object TupleIntegerEncoder {
  def encodeAcc(l: List[(Int, Int)], productAcc: Int, acc: Int): Int = l match {
    case List() => acc
    case e::tail => encodeAcc(tail, productAcc*e._2, acc+e._1*productAcc)
  }

  def decodeAcc(n: (Int, List[Int]), modAcc: Int, acc: List[Int], dim: Int): List[Int] = n match {
    case (v, List()) => acc++List(v)
    case (0, l) => acc++List.fill[Int](dim-acc.length)(0)
    case (v, h::tail) => decodeAcc((v/modAcc, tail), h, acc++List(v%modAcc), dim)
  }

}