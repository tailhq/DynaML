package io.github.mandar2812.dynaml.models.stp

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.kernels.LocalScalarKernel
import org.apache.log4j.Logger

/**
  * Created by mandar on 27/08/16.
  */
class MOStudentTRegression[I](
  mu: Double, cov: LocalScalarKernel[(I, Int)],
  n: LocalScalarKernel[(I, Int)],
  data: Stream[(I, DenseVector[Double])],
  num: Int, numOutputs: Int) extends
  AbstractSTPRegressionModel[Stream[(I, DenseVector[Double])], (I, Int)](
    mu, cov, n, data, num*numOutputs) {

  private val logger = Logger.getLogger(this.getClass)

  val noutputs = numOutputs

  /**
    * Convert from the underlying data structure to
    * Seq[(I, Y)] where I is the index set of the GP
    * and Y is the value/label type.
    **/
  override def dataAsSeq(data: Stream[(I, DenseVector[Double])]): Seq[((I, Int), Double)] =
  data.map((patternAndLabel) =>
    patternAndLabel._2.mapPairs((i, label) =>
      ((patternAndLabel._1, i), label)
    ).toArray.toSeq).reduceLeft((s1, s2) => s1 ++ s2)
}

