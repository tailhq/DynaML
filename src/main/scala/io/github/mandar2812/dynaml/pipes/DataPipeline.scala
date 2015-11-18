package io.github.mandar2812.dynaml.pipes

/**
  * @author mandar2812 on 18/11/15.
  *
  * Top level trait representing an
  * abstract pipe that defines a transformation
  * between two data types, i.e. [[Source]] and [[Destination]]
  */
trait DataPipeline[Source, Destination] {
  def run(data: Source): Destination

  /**
    * Represents the composition of two
    * pipes, resulting in a third pipe
    * Schematically represented as:
    *
    * [[Source]] -> [[Destination]] :: [[Destination]] -> [[Further]] ==
    * [[Source]] -> [[Further]]
    *
    * */
  def ::[Further, NextPipe <: DataPipeline[Destination, Further]](that: NextPipe):
  DataPipeline[Source, Further] = {
    val runFunc = (d: Source) => that.run(this.run(d))
    new DataPipeline[Source, Further] {
      def run(data: Source) = runFunc(data)
    }
  }
}

object DataPipeline {
  def apply[S,D](func: (S) => D): DataPipeline[S, D] = {
    new DataPipeline[S,D] {
      def run(data: S) = func(data)
    }
  }
}
