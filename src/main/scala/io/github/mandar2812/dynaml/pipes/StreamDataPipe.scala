package io.github.mandar2812.dynaml.pipes

/**
  * @author mandar2812 on 17/11/15.
  *
  * Represents an abstract stream data pipeline.
  * @tparam I The type of a single source record
  * @tparam J The result type of a single record.
  *
  */
trait StreamDataPipe[I, J, K] extends DataPipe[Stream[I], K]{

  /**
    * The functions that compose the
    * pipe operations.
    *
    * */
  val pipe: (I) => J

  /**
    * The function which writes
    * to the destination
    * */

  override def run(data: Stream[I]): K
}

trait StreamMapPipe[I, J] extends StreamDataPipe[I, J, Stream[J]] {
  override def run(data: Stream[I]): Stream[J] = data.map(pipe)
}

trait StreamFilterPipe[I] extends StreamDataPipe[I, Boolean, Stream[I]] {
  override def run(data: Stream[I]): Stream[I] = data.filter(pipe)
}

trait StreamSideEffectPipe[I] extends StreamDataPipe[I, Unit, Unit] {
  override def run(data: Stream[I]): Unit = data.foreach(pipe)
}

object StreamDataPipe {
  
}