package io.github.mandar2812.dynaml.utils

import breeze.linalg.{DenseMatrix, trace}
import io.github.mandar2812.dynaml.utils
import org.scalatest.{FlatSpec, Matchers}

class UtilsSpec extends FlatSpec with Matchers {

  "log1pexp" should " compute correctly" in assert(utils.log1pExp(0d) == math.log(2))

  "diagonal function" should " obtain the diagonals of matrices correctly" in {

    val m = DenseMatrix.eye[Double](2)

    val errMat = utils.diagonal(m) - m

    assert(trace(errMat.t*errMat) == 0.0)



  }
}
