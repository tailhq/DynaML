package io.github.mandar2812.dynaml.pipes

/**
  * Data Pipe wrapper of a curried function of order 1
  * @author mandar2812 date: 21/02/2017.
  *
  * */
trait MetaPipe[Source, Intermediate, Destination] extends
  DataPipe[Source, DataPipe[Intermediate, Destination]] {

  self =>

  /**
    * This is a modified form of the [[>]] operator in [[DataPipe]]
    * The pipe composition is applied to the resultant [[DataPipe]]
    * produced by the [[MetaPipe.run]] and the argument.
    *
    * @tparam Further The type of the result of the parameter.
    * @param next The data pipe to join
    * */
  def >>[Further](next: DataPipe[Destination, Further])
  : MetaPipe[Source, Intermediate, Further] =
    new MetaPipe[Source, Intermediate, Further] {
      override def run(data: Source) = self.run(data) > next
    }

}

object MetaPipe {

  def apply[Source, Intermediate, Destination](
    f: Source => Intermediate => Destination)
  : MetaPipe[Source, Intermediate, Destination] =
    new MetaPipe[Source, Intermediate, Destination] {
      override def run(data: Source) = DataPipe(f(data))
    }
}

/**
  * Wraps a curried function of order 1 which takes 2 arguments
  * and returns a [[Function1]].
  * @author mandar2812 date: 21/02/2017
  *
  * */
trait MetaPipe21[A, B, C, D] extends DataPipe2[A, B, DataPipe[C, D]] {

  self =>

  def >>[E](next: DataPipe[D, E]): MetaPipe21[A, B, C, E] = new MetaPipe21[A, B, C, E] {
    override def run(data1: A, data2: B) = self.run(data1, data2) > next
  }
}

object MetaPipe21 {
  def apply[A, B, C, D](f: (A, B) => C => D): MetaPipe21[A, B, C, D] =
    new MetaPipe21[A, B, C, D]{
      override def run(data1: A, data2: B) = DataPipe(f(data1, data2))
    }
}

/**
  * Wraps a curried function of order 1 which takes 1 arguments
  * and returns a [[Function2]].
  * @author mandar2812 date: 21/02/2017
  *
  * */
trait MetaPipe12[A, B, C, D] extends DataPipe[A, DataPipe2[B, C, D]] {
  self =>

  def >>[E](next: DataPipe[D, E]): MetaPipe12[A, B, C, E] = new MetaPipe12[A, B, C, E] {
    override def run(data: A) = self.run(data) > next
  }

}

object MetaPipe12 {
  def apply[A, B, C, D](f: A => (B, C) => D): MetaPipe12[A, B, C, D] =
    new MetaPipe12[A, B, C, D] {
      override def run(data: A) = DataPipe2(f(data))
    }
}