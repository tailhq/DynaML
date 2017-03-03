package io.github.mandar2812.dynaml.modelpipe

import io.github.mandar2812.dynaml.kernels.LocalScalarKernel
import io.github.mandar2812.dynaml.models.sgp.AbstractSkewGPModel
import io.github.mandar2812.dynaml.pipes.{DataPipe, DataPipe4}

import scala.reflect.ClassTag

/**
  * @author mandar2812 date: 02/03/2017.
  *
  * A [[DataPipe4]] which returns an instance of
  * [[AbstractSkewGPModel]].
  *
  * @tparam Data The type of the training data
  * @tparam IndexSet The type of input features.
  *
  * @param cov Covariance function of the process
  * @param n Noise covariance.
  * @param transform An implicit parameter which can convert
  *                  the training data from type [[Data]] to [[Seq]]
  */
class ExtendedSkewGPPipe[Data, IndexSet: ClassTag](
  cov: LocalScalarKernel[IndexSet],
  n: LocalScalarKernel[IndexSet])(
  implicit transform: DataPipe[Data, Seq[(IndexSet, Double)]]) extends
  DataPipe4[DataPipe[IndexSet, Double], Double, Double, Data, AbstractSkewGPModel[Data, IndexSet]] {

  /**
    * Return a [[AbstractSkewGPModel]] instance
    *
    * @param data1 The mean function as a [[DataPipe]]
    * @param data2 The skewness parameter
    * @param data3 The cutoff parameter
    * @param data4 The training data set of type [[Data]]
    * */
  override def run(data1: DataPipe[IndexSet, Double], data2: Double, data3: Double, data4: Data) =
    AbstractSkewGPModel(cov, n, data1, data2, data3)(data4)
}
