package io.github.mandar2812.dynaml.utils

import breeze.linalg._
import breeze.stats.distributions.{Gaussian, LogNormal}
import io.github.mandar2812.dynaml.algebra.PartitionedMatrixOps._
import io.github.mandar2812.dynaml.algebra.{PartitionedMatrix, btrace}
import io.github.mandar2812.dynaml.analysis.implicits._
import io.github.mandar2812.dynaml.utils
import org.scalatest.{FlatSpec, Matchers}
import spire.algebra._
import spire.implicits._

class UtilsSpec extends FlatSpec with Matchers {

  "log1pexp" should " compute correctly" in assert(utils.log1pExp(0d) == math.log(2))

  "diagonal function" should " obtain the diagonals of matrices correctly" in {

    val m = DenseMatrix.eye[Double](2)

    val errMat = utils.diagonal(m) - m

    assert(trace(errMat.t*errMat) == 0.0)


    val blocks = Stream.tabulate(2, 2)((i, j) =>
      if(i == j) ((i.toLong, j.toLong), m)
      else ((i.toLong, j.toLong), DenseMatrix.zeros[Double](2, 2))
    ).flatten

    val pm = PartitionedMatrix(blocks, numrows = 4L, numcols = 4L)

    val errMat2 = utils.diagonal(pm) - pm

    assert(btrace(errMat2.t*errMat2) == 0.0)

  }

  "CSV Reader" should " return an non-empty iterator over csv lines" in
    assert(utils.getCSVReader("data/delve.csv", ',').iterator.hasNext)

  "Non empty text files " should "be readable" in assert(utils.textFileToStream("data/delve.csv").nonEmpty)

  "Sumamry/Order statistics" should " compute correctly" in {

    val data = List(DenseVector(0d), DenseVector(1d))

    val (m, s) = utils.getStats(data)

    assert(m(0) == 0.5 && s(0) == 0.5)

    val (m1, s1) = utils.getStatsMult(data)

    assert(m1(0) == 0.5 && s1(0, 0) == 0.5)

    val (min, max) = utils.getMinMax(data)

    assert(min(0) == 0d && max(0) == 1d)

  }

  "Quick select" should " find the kth smallest element in a collection" in
    assert(utils.quickselect(Stream(2d, 3d, 4d, 1d), 2) == 2d)

  "Median implementation" should " compute correctly" in
    assert(utils.median(Stream(4d, 3d, 2d, 5d, 1d)) == 3d)

  "Prior Map distributions" should " be generated from Map objects of continuous distributions" in {

    val p = Map("a" -> Gaussian(0d, 1d), "b" -> LogNormal(0d, 1d))

    assert(utils.getPriorMapDistr(p).draw().keys.toSeq == Seq("a", "b"))

  }

  "Chebyshev, Legendre, Hermite, Harmonic & factorial functions" should " compute" in {

    assert(utils.hermite(0, 100d) == 1d && utils.hermite(1, 13.45) == 13.45 && utils.hermite(2, 2) == 3d)

    assert(
      utils.chebyshev(0, 100d, 1) == 1d &&
        utils.chebyshev(1, 13.45, 1) == 13.45 &&
        utils.chebyshev(2, 2, 2) == 15d)

    assert(utils.legendre(0, 100d) == 1d && utils.legendre(1, 13.45) == 13.45 && utils.legendre(2, 2) == 5.5)

    assert(utils.H(2.5) == 1.5)

    assert(utils.factorial(5) == 120)

  }

  "Numeric ranges " should " be left inclusive only" in
    assert(utils.range[Double](0d, 1d, 2) == Stream(0d, 0.5))

  "Haar DWT matrix " should "compute correctly" in {
    assert(utils.haarMatrix(2) == DenseMatrix((1d, 1d), (1d, -1d)))

    assert(
      utils.haarMatrix(4) ==
        DenseMatrix(
          (1d, 1d, 1d, 1d),
          (1d, 1d, -1d, -1d),
          (1d, -1d, 0d, 0d),
          (0d, 0d, 1d, -1d))
    )
  }

  "Product Fields" should " extend from their components" in {

    val f = Field[Double]

    val pf = utils.productField(f, f)

    val (x, y) = ((1d, 2d), (1d, 2d))

    assert(
      pf.plus(x, y) == (2d, 4d) &&
        pf.minus(x, y) == (0d, 0d) &&
        pf.times(x, y) == (1d, 4d) &&
        pf.div(x, y) == (1d, 1d) &&
        pf.negate(x) == (-1d, -2d) &&
        pf.zero == (f.zero, f.zero) &&
        pf.one == (f.one, f.one) &&
        pf.mod(x, y) == (0d, 0d) &&
        pf.quot(x, y) == (1d, 1d) &&
        pf.gcd(x, y) == (1d, 1d) &&
        pf.lcm(x, y) == (1d, 4d))


  }

  "String replace function" should " work as expected" in
    assert(utils.replace("abab")("blah")("abab blah abab") == "blah blah blah")

  "isSquareMatrix and isSymmetricMatrix methods" should " yield correct results" in {

    val res1 = try {
      utils.isSquareMatrix(DenseMatrix.zeros[Double](2, 3))
      None
    } catch {
      case m: MatrixNotSquareException => Some(m)
    }


    val res2 = try {
      utils.isSymmetricMatrix(DenseMatrix((1d, 3d), (2d, 4d)))
      None
    } catch {
      case m: MatrixNotSymmetricException => Some(m)
    }



    assert(res1.isDefined)
    assert(res2.isDefined)


  }


}
