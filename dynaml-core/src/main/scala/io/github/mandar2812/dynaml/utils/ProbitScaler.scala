package io.github.mandar2812.dynaml.utils

import breeze.stats.distributions._
import breeze.linalg.{DenseMatrix, DenseVector, cholesky, inv}
import io.github.mandar2812.dynaml.pipes.{ReversibleScaler, Scaler}

object ProbitScaler extends ReversibleScaler[DenseVector[Double]] {

    private val std_gaussian = Gaussian(0d, 1d)

    override def run(data: DenseVector[Double]): DenseVector[Double] = data.map(std_gaussian.cdf)

    override val i: Scaler[DenseVector[Double]] = Scaler((p: DenseVector[Double]) => p.map(std_gaussian.inverseCdf))

}
