package io.github.tailhq.dynaml.modelpipe

import io.github.tailhq.dynaml.DynaMLPipe
import io.github.tailhq.dynaml.kernels.LocalScalarKernel
import io.github.tailhq.dynaml.models.sgp.ESGPModel
import io.github.tailhq.dynaml.pipes.{DataPipe, DataPipe4}

import scala.reflect.ClassTag

/**
  * @author tailhq date: 02/03/2017.
  *
  * A [[DataPipe4]] which returns an instance of [[ESGPModel]].
  *
  * @tparam Data The type of the training data
  * @tparam IndexSet The type of input features.
  * @param cov Covariance function of the process
  * @param n Noise covariance.
  * @param transform An implicit parameter which can convert
  *                  the training data from type [[Data]] to [[Seq]]
  */
class ESGPPipe4[Data, IndexSet: ClassTag](
  cov: LocalScalarKernel[IndexSet],
  n: LocalScalarKernel[IndexSet])(
  implicit transform: DataPipe[Data, Seq[(IndexSet, Double)]]) extends
  DataPipe4[DataPipe[IndexSet, Double], Double, Double, Data, ESGPModel[Data, IndexSet]] {

  /**
    * @return an [[ESGPModel]] instance
    *
    * @param trend The mean function as a [[DataPipe]]
    * @param skewness The skewness parameter
    * @param cutoff The cutoff parameter
    * @param training_data The training data set of type [[Data]]
    * */
  override def run(
    trend: DataPipe[IndexSet, Double],
    skewness: Double, cutoff: Double,
    training_data: Data) =
    ESGPModel(
      cov, n, trend,
      skewness, cutoff)(
      training_data)
}


object ESGPPipe4 {

  def apply[IndexSet: ClassTag](
    cov: LocalScalarKernel[IndexSet],
    n: LocalScalarKernel[IndexSet]): ESGPPipe4[Seq[(IndexSet, Double)], IndexSet] = {

    implicit val tr = DynaMLPipe.identityPipe[Seq[(IndexSet, Double)]]
    new ESGPPipe4[Seq[(IndexSet, Double)], IndexSet](cov, n)
  }
}