package io.github.tailhq.dynaml.modelpipe

import io.github.tailhq.dynaml.models.Model
import io.github.tailhq.dynaml.pipes.DataPipe

/**
  * @author tailhq on 09/09/16.
  */
trait ModelTestPipe[Source, Destination,
Data, Features, Label,
M <: Model[Data, Features, Label]]
  extends DataPipe[(M, Source), Destination] {

}
