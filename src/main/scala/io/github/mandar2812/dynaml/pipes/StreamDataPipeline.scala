package io.github.mandar2812.dynaml.pipes

import scala.collection.mutable.{MutableList => ML}

/**
  * @author mandar2812 on 17/11/15.
  *
  * Represents an abstract stream data pipeline.
  * @tparam S The data source, i.e. URL, File etc
  * @tparam I The intermediate type of a single record.
  * @tparam D The destination of the pipeline,
  *           D = Data structure if the result is to be held in memory
  *           D = Unit if the result is a side effect like writing to disk or
  *           a database instance.
  *
  * It assumes.
  * 1) Data Source which can be converted to a [[Stream]]
  * of records of type [[I]]
  *
  */
trait StreamDataPipeline[S, I, D] extends DataPipeline[S, D]{

  /**
    * Initialization Function
    *
    * */
  val initialize: (S) => Stream[I]

  /**
    * The functions that compose the
    * pipe operations.
    *
    * */
  val pipe: ML[(I) => I]

  /**
    * The function which writes
    * to the destination
    * */
  val write: (Stream[I]) => D

  override def run(data: S): D = {
    write(initialize(data).map(record => {
      pipe.foldLeft(record)((rec, func) => func(rec))
    }))
  }
}

/**
  * Represents an abstract pipeline which converts a data
  * source into a [[Stream]]
  * */
trait DataToStreamPipeline[Source, I] extends StreamDataPipeline[Source, I, Stream[I]] {
  val write = identity[Stream[I]] _
  val func: (I) => I
  override val pipe = ML(func)

}

trait StreamToResultPipeline[I, D] extends StreamDataPipeline[Stream[I], I, D] {
  override val initialize = identity[Stream[I]] _
  override val pipe = ML(identity[I] _)
}
