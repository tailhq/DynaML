package io.github.mandar2812.dynaml.pipes

/**
  * Created by mandar on 18/11/15.
  */
trait DataPipeline[Source, Destination] {
  def run(data: Source): Destination
  def ::[Further, NextPipe <: DataPipeline[Destination, Further]](that: NextPipe):
  DataPipeline[Source, Further] = {
    val runFunc = (d: Source) => that.run(this.run(d))
    new DataPipeline[Source, Further] {
      def run(data: Source) = runFunc(data)
    }
  }
}
