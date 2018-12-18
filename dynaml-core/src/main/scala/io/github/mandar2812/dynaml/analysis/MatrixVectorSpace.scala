package io.github.mandar2812.dynaml.analysis

import breeze.linalg.{DenseMatrix, trace}
import spire.algebra.{Field, InnerProductSpace}
import spire.implicits._

/**
  * Defines an inner product space over m &times; n matrices.
  * @author mandar2812 date 21/06/2017.
  * */
case class MatrixVectorSpace(rows: Int, cols: Int) extends
  InnerProductSpace[DenseMatrix[Double], Double] {

  override def dot(v: DenseMatrix[Double], w: DenseMatrix[Double]) = trace(v.t*w)

  override implicit def scalar = Field[Double]

  override def timesl(r: Double, v: DenseMatrix[Double]) = v*r

  override def negate(x: DenseMatrix[Double]) = x*(-1d)

  override def zero = DenseMatrix.zeros[Double](rows, cols)

  override def plus(x: DenseMatrix[Double], y: DenseMatrix[Double]) = x+y
}