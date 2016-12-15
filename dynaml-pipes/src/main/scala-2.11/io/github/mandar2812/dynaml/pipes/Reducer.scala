package io.github.mandar2812.dynaml.pipes

/**
  * @author mandar date: 15/12/2016.
  *
  * Represents any function which takes
  * a collection of [[Double]] and returns
  * a [[Double]].
  */
sealed trait Reducer extends ReducerPipe[Double]

object SumReducer extends Reducer {
  override def run(data: Array[Double]): Double = data.sum
}

object ProductReducer extends Reducer{
  override def run(data: Array[Double]): Double = data.product
}