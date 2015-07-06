package org.kuleuven.esat.optimization

import breeze.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.kuleuven.esat.svm.LSSVMSparkModel

import scala.util.Random

/**
 * @author mandar2812
 */
class ConjugateGradientSpark extends RegularizedOptimizer[Int, DenseVector[Double],
  DenseVector[Double], Double, RDD[LabeledPoint]]{

  def getRegParam = this.regParam

  /**
   * Find the optimum value of the parameters using
   * Gradient Descent.
   *
   * @param nPoints The number of data points
   * @param initialP The initial value of the parameters
   *                 as a [[DenseVector]]
   * @param ParamOutEdges An [[java.lang.Iterable]] object
   *                      having all of the out edges of the
   *                      parameter node
   *
   * @return The value of the parameters as a [[DenseVector]]
   *
   *
   * */
  override def optimize(nPoints: Long,
                        ParamOutEdges: RDD[LabeledPoint],
                        initialP: DenseVector[Double]): DenseVector[Double] = {
    val (a,b) = LSSVMSparkModel.getFeatureMatrix(nPoints, ParamOutEdges,
      initialP, this.miniBatchFraction, this.regParam)

    ConjugateGradient.runCG(a, b, initialP, 0.0001, this.numIterations)
  }
}

object ConjugateGradientSpark {
  /**
   * Solves for x in A.x = b (where A is symmetric +ve semi-definite)
   * iteratively using the Conjugate Gradient
   * algorithm.
   * */
  def runCG(A: DenseMatrix[Double],
            b: DenseVector[Double],
            x: DenseVector[Double],
            epsilon: Double,
            MAX_ITERATIONS: Int): DenseVector[Double] = {
    val residual = b - (A*x)
    val p = residual
    var count = 1.0
    var alpha = math.pow(norm(residual, 2), 2)/(p.t * (A*p))
    var beta = 0.0
    while(norm(residual, 2) >= epsilon && count <= MAX_ITERATIONS) {
      //update x
      axpy(alpha, p, x)
      //before updating residual, calculate norm (required for beta)
      val de = math.pow(norm(residual, 2), 2)
      //update residual
      axpy(-1.0*alpha, A*p, residual)
      //calculate beta
      beta = math.pow(norm(residual, 2), 2)/de
      //update p
      p :*= beta
      axpy(1.0, residual, p)
      //update alpha
      alpha = math.pow(norm(residual, 2), 2)/(p.t * (A*p))
      count += 1
    }
    x
  }
}
