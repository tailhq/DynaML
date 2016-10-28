package io.github.mandar2812.dynaml.probability

import io.github.mandar2812.dynaml.pipes.DataPipe

/**
  * Created by mandar on 24/09/2016.
  */
class MeasurableFunction[
Domain1, Domain2](baseRV: RandomVariable[Domain1])(func: DataPipe[Domain1, Domain2])
  extends RandomVariable[Domain2] {
  override val sample: DataPipe[Unit, Domain2] = baseRV.sample > func
}

object MeasurableFunction {
  def apply[Domain1, Domain2](baseRV: RandomVariable[Domain1])(func: DataPipe[Domain1, Domain2])
  : MeasurableFunction[Domain1, Domain2] = new MeasurableFunction[Domain1, Domain2](baseRV)(func)

  def apply[Domain1, Domain2](baseRV: RandomVariable[Domain1])(func: Domain1 => Domain2)
  : MeasurableFunction[Domain1, Domain2] = new MeasurableFunction[Domain1, Domain2](baseRV)(DataPipe(func))

}