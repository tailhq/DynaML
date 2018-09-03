package io.github.mandar2812.dynaml.optimization

import breeze.linalg.{DenseMatrix, DenseVector, norm}
import org.scalatest.{FlatSpec, Matchers}

/**
  * Created by mandar on 5/7/16.
  */
class ConjugateGradientSpec extends FlatSpec with Matchers {

  "Conjugate Gradient " should "be able to solve linear systems "+
    "of the form A.x = b, where A is symmetric positive definite. " in {

    val A = DenseMatrix((1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 4.0))
    val b = DenseVector(2.0, 4.0, 8.0)

    val x = DenseVector(2.0, 2.0, 2.0)

    val epsilon = 1E-6

    val xnew = ConjugateGradient.runCG(A, b, DenseVector(1.0, 1.0, 1.0), epsilon, MAX_ITERATIONS = 3)

    assert(norm(xnew-x) <= epsilon)
  }

}
