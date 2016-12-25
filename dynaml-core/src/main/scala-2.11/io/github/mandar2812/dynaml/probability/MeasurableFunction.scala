package io.github.mandar2812.dynaml.probability

import io.github.mandar2812.dynaml.pipes.DataPipe

/**
  * Created by mandar2812 on 24/09/2016.
  */
class MeasurableFunction[
Domain1, Domain2](baseRV: RandomVariable[Domain1])(func: DataPipe[Domain1, Domain2])
  extends RandomVariable[Domain2] {
  override val sample: DataPipe[Unit, Domain2] = baseRV.sample > func

  def _baseRandomVar = baseRV

  //def asProbabilityModel = new ProbabilityModel()
}

object MeasurableFunction {

  def apply[Domain1, @specialized Domain2](baseRV: RandomVariable[Domain1])(func: Domain1 => Domain2)
  : MeasurableFunction[Domain1, Domain2] = new MeasurableFunction(baseRV)(DataPipe(func))

}

class RealValuedMeasurableFunction[Domain1](baseRV: RandomVariable[Domain1])(func: DataPipe[Domain1, Double])
  extends MeasurableFunction[Domain1, Double](baseRV)(func) with ContinuousRandomVariable[Double]

object RealValuedMeasurableFunction {

  def apply[Domain1](baseRV: RandomVariable[Domain1])(func: (Domain1) => Double)
  : RealValuedMeasurableFunction[Domain1] = new RealValuedMeasurableFunction(baseRV)(DataPipe(func))
}