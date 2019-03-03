package io.github.mandar2812.dynaml.models.stp

import io.github.mandar2812.dynaml.kernels.LocalScalarKernel
import org.apache.log4j.Logger

/**
  * Created by mandar on 27/08/16.
  */

class MTStudentTRegression[I](
  mu: Double,
  cov: LocalScalarKernel[(I, Int)],
  n: LocalScalarKernel[(I, Int)],
  data: Seq[Stream[(I, Double)]],
  num: Int, numOutputs: Int) extends
  AbstractSTPRegressionModel[Seq[Stream[(I, Double)]], (I, Int)](
    mu, cov, n, data, num*numOutputs) {

  assert(
    data.length == numOutputs,
    "Number of outputs in data should match numOutputs constructor variable"
  )

  private val logger = Logger.getLogger(this.getClass)

  val noutputs = numOutputs

  /**
    * Convert from the underlying data structure to
    * Seq[(I, Y)] where I is the index set of the GP
    * and Y is the value/label type.
    **/
  override def dataAsSeq(data: Seq[Stream[(I, Double)]]): Seq[((I, Int), Double)] =
  data.zipWithIndex.map((patternSet) =>
    patternSet._1.map(patternAndLabel => ((patternAndLabel._1, patternSet  ._2), patternAndLabel._2))
  ).reduceLeft((s1, s2) => s1 ++ s2)
}

