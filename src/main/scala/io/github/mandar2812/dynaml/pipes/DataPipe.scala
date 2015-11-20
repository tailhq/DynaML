package io.github.mandar2812.dynaml.pipes

/**
  * @author mandar2812 on 18/11/15.
  *
  * Top level trait representing an
  * abstract pipe that defines a transformation
  * between two data types, i.e. [[Source]] and [[Destination]]
  */
trait DataPipe[Source, Destination] {
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
  def >[Further](that: DataPipe[Destination, Further]):
  DataPipe[Source, Further] = {
    val runFunc = (d: Source) => that.run(this.run(d))
    DataPipe(runFunc)
  }
}

trait ParallelPipe[Source1, Result1, Source2, Result2]
  extends DataPipe[(Source1, Source2), (Result1, Result2)] {

}

trait BifurcationPipe[Source, Result1, Result2]
  extends DataPipe[Source, (Result1, Result2)] {


}

trait SideEffectPipe[I] extends DataPipe[I, Unit] {

}

object DataPipe {
  def apply[S,D](func: (S) => D): DataPipe[S, D] = {
    new DataPipe[S,D] {
      def run(data: S) = func(data)
    }
  }

  def apply[S1, D1, S2, D2](pipe1: DataPipe[S1, D1], pipe2: DataPipe[S2, D2]):
  ParallelPipe[S1, D1, S2, D2] = {
    new ParallelPipe[S1, D1, S2, D2] {
      def run(data: (S1, S2)) = (pipe1.run(data._1), pipe2.run(data._2))
    }
  }

  def apply[S, D1, D2](func: (S) => (D1, D2)): BifurcationPipe[S, D1, D2] = {
    new BifurcationPipe[S, D1, D2] {
      def run(data: S) = func(data)
    }
  }

  def apply[S](func: (S) => Unit): SideEffectPipe[S] = {
    new SideEffectPipe[S] {
      def run(data: S) = func(data)
    }
  }
}

