package org.kuleuven.esat.optimization

import breeze.linalg._
import org.kuleuven.esat.graphUtils.CausalEdge

/**
 * @author mandar2812
 */
class ConjugateGradient extends RegularizedOptimizer[Int, DenseVector[Double],
  DenseVector[Double], Double, Iterable[CausalEdge]]{

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
                        ParamOutEdges: Iterable[CausalEdge],
                        initialP: DenseVector[Double]): DenseVector[Double] = {

    val dims = initialP.length
    //Cast as problem of form A.w = b
    //A = Phi^T . Phi + I_dims*regParam
    //b = Phi^T . Y
    val (a,b): (DenseMatrix[Double], DenseVector[Double]) = ParamOutEdges.map((edge) => {
      val phi = DenseVector(edge.getPoint().getFeatureMap())
      val label = edge.getLabel().getValue()
      val phiY: DenseVector[Double] = phi * label
      (phi*phi.t, phiY)
    }).reduce((couple1, couple2) => {
      (couple1._1+couple2._1, couple1._2+couple2._2)
    })
    val A = a + (DenseMatrix.eye[Double](dims)*regParam)

    ConjugateGradient.runCG(A, b, initialP, 0.0001, this.numIterations)
  }
}

object ConjugateGradient {
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
