package io.github.mandar2812.dynaml.models

import io.github.mandar2812.dynaml.pipes.DataPipe

/**
  * @author mandar2812 on 09/09/16.
  */
trait ModelTestPipe[Source, Destination,
Data, Features, Label,
M <: Model[Data, Features, Label]]
  extends DataPipe[(M, Source), Destination] {

}
