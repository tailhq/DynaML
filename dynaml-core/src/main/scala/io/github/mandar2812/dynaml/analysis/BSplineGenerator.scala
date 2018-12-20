package io.github.mandar2812.dynaml.analysis

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.pipes.{DataPipe, MetaPipe, MetaPipe21}


/**
  * <h3>(B)ASIC SPLINES</h3>
  *
  * B-Spline generator as described in
  * <a href="http://ftp.cs.wisc.edu/Approx/bsplbasic.pdf">B(asic)-Spline Basics</a> by Carl de Boor.
  *
  * B<sub>i,k</sub>(t) :=
  * &omega;<sub>i,k</sub>(t) B<sub>i,k-1</sub>(t) + (1 - &omega;<sub>i+1,k</sub>(t)) B<sub>i+1,k-1</sub>
  * @author mandar date 28/06/2017.
  * */
abstract class BSplineGenerator extends MetaPipe21[Int, Int, Double, Double] {

  /**
    * The knot sequence, strictly non-decreasing
    * */
  val knot: (Int) => Double

  /**
    * &omega;<sub>i,k</sub>(t) := 0, if t<sub>i</sub> = t<sub>i+k-1</sub>
    *                          otherwise, (t - t<sub>i</sub>)/(t<sub>i+k-1</sub> - t<sub>i</sub>)
    * */
  def omega(i: Int, k: Int)(x: Double): Double = {
    val (t_i, t_ik) = (knot(i), knot(i+k-1))
    if(t_i != t_ik) (x-t_i)/(t_ik-t_i) else 0d
  }

  protected def bsplinerec(x: Double, k: Int, i: Int, acc: Seq[Double]): Double = acc match {
    case Seq() => 1d
    case Seq(a) => a
    case _ => bsplinerec(
      x, k+1, i, acc.sliding(2).toSeq.zipWithIndex.map(couple => {
        val (accum, index) = couple
        val (first, second) = (accum.head, accum.last)
        omega(index+i, k+1)(x)*first + (1d - omega(index+i+1, k+1)(x))*second
      }))
  }

  override def run(segment: Int, order: Int) = DataPipe(
    (t: Double) => bsplinerec(
      t, 1, segment,
      Seq.tabulate[Double](order)(l =>
        if(t >= knot(segment+l) && t < knot(segment+l+1)) 1d
        else 0d)
    )
  )
}

/**
  * The [[BSplineGenerator]] companion object has convenience methods for
  * creating arbitrary b-spline generators from knot sequences.
  * */
object BSplineGenerator {

  /**
    * Create a b-spline generator from a supplied sequence of knots.
    * @param ks A sequence of knots
    * @param isSorted Set to true if the sequence is already sorted (defaults to false).
    * @return a [[BSplineGenerator]] instance.
    * */
  def apply(ks: Seq[Double], isSorted: Boolean = false): BSplineGenerator = {
    val sorted_knots = if (isSorted) ks else ks.sorted

    new BSplineGenerator {
      override val knot = (i: Int) => {
        if(i >= 0 && i < sorted_knots.length) sorted_knots(i)
        else if(i < 0) sorted_knots.head
        else sorted_knots.last
      }
    }
  }

  /**
    * Create a b-spline generator from a
    * bi-infinite sequence of knots (specified as a [[DataPipe]])
    *
    * @param knotFunction A [[DataPipe]] from [[Int]] to [[Double]] which outputs
    *                     the knot for any index.
    * @return a [[BSplineGenerator]] instance.
    * */
  def apply(knotFunction: DataPipe[Int, Double]): BSplineGenerator = new BSplineGenerator {
    override val knot = knotFunction.run _
  }

}

/**
  * Cardinal B-Splines have knots as the sequence of integers.
  * */
object CardinalBSplineGenerator extends BSplineGenerator {
  override val knot = (i: Int) => i.toDouble
}

/**
  * Bernstein B-Splines have knots as (...000111...)
  * */
object BernsteinSplineGenerator extends BSplineGenerator {
  override val knot = (i: Int) => if(i <= 0) 0d else 1d
}

/**
  * <h3>Generalised Piecewise Polynomial Splines</h3>
  *
  * Represents linear combinations of b-splines.
  *
  * */
abstract class SplineGenerator extends MetaPipe[Int, Double, Double] {

  /**
    * Specifies which knots have non-zero weights
    * */
  val knot_indices: Array[Int]

  /**
    * Weights each basic spline
    * */
  val coefficients: DenseVector[Double]

  /**
    * Used to calculate the basic splines
    * */
  val bSplineGenerator: BSplineGenerator

  override def run(order: Int) = DataPipe((x: Double) => {
    val splines = DenseVector(knot_indices.map(i => bSplineGenerator(i, order)(x)))
    coefficients dot splines
  })
}

object SplineGenerator {

  /**
    * Creates a spline generator.
    * */
  def apply(
    generator: BSplineGenerator)(
    indices: Seq[Int], coeff: DenseVector[Double]) =
    new SplineGenerator {

      override val bSplineGenerator = generator

      override val knot_indices = indices.toArray

      override val coefficients = coeff
    }

  def apply(knots: Seq[Double], coefficients: DenseVector[Double]): SplineGenerator =
    apply(BSplineGenerator(knots))(knots.indices, coefficients)

}
