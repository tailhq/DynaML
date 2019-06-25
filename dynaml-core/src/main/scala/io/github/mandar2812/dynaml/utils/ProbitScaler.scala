package io.github.mandar2812.dynaml.utils

import breeze.stats.distributions._
import breeze.linalg.{DenseMatrix, DenseVector, cholesky, inv}
import io.github.mandar2812.dynaml.pipes.{ReversibleScaler, Scaler}
import org.platanios.tensorflow.api.core.types.{IsReal, TF, IsFloatOrDouble}
import org.platanios.tensorflow.api.{tfi, Tensor}

object ProbitScaler extends ReversibleScaler[DenseVector[Double]] {

    private val std_gaussian = Gaussian(0d, 1d)

    override def run(data: DenseVector[Double]): DenseVector[Double] = data.map(std_gaussian.cdf)

    override val i: Scaler[DenseVector[Double]] = Scaler((p: DenseVector[Double]) => p.map(std_gaussian.inverseCdf))

}

//@TODO: Implement the erf inverse.
/* class ProbitScalerTF[T: TF: IsFloatOrDouble] extends ReversibleScaler[Tensor[T]] {
    
    override def run(data: Tensor[T]): Tensor[T] = 
    tfi.erf(data.divide(Tensor(math.sqrt(2.0f)).castTo[T]))
        .add(Tensor(1.0f).castTo[T])
        .multiply(Tensor(0.5f).castTo[T])
        .castTo[T]


    override val i: Scaler[Tensor[T]] = ???
} */