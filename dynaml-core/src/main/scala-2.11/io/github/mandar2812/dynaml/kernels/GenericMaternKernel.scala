package io.github.mandar2812.dynaml.kernels

import breeze.linalg.DenseMatrix
import breeze.numerics.{exp, lgamma, sqrt}
import spire.algebra.{Field, InnerProductSpace}

import io.github.mandar2812.dynaml.utils._

/**
  * @author mandar2812 date: 27/01/2017.
  *
  * Implementation of the half integer Matern
  * covariance function
  */
class GenericMaternKernel[T](private var l: Double, p: Int = 2)(
  implicit ev: Field[T] with InnerProductSpace[T, Double])
  extends StationaryKernel[T, Double, DenseMatrix[Double]]
    with LocalScalarKernel[T] with Serializable { self =>

  override val hyper_parameters = List("p", "l")

  blocked_hyper_parameters = List("p")

  state = Map("p" -> p.toDouble, "l" -> l)

  override def setHyperParameters(h: Map[String, Double]) = {
    super.setHyperParameters(h)
    if(h contains "p")
      state += ("p" -> math.floor(math.abs(h("p"))))
    this
  }

  override def eval(x: T) = {
    val r = math.sqrt(ev.dot(x, x))
    val nu = state("p") + 0.5
    val lengthscale = state("l")
    val order = state("p").toInt

    val leadingTerm = exp(-sqrt(2*nu)*r/lengthscale)*exp(lgamma(order + 1) - lgamma(2*order + 1))

    val sumTerm = (0 to order).map(i => {
      math.pow(sqrt(8*nu)*r/lengthscale, order-i)*(factorial(order+i)/(factorial(i)*factorial(order-i)))
    }).sum

    leadingTerm*sumTerm
  }

  override def gradient(x: T, y: T) = {

    val diff = ev.minus(x, y)
    val r = math.sqrt(ev.dot(diff, diff))

    val nu = state("p") + 0.5
    val lengthscale = state("l")
    val order = state("p").toInt

    val leadingTerm = exp(-sqrt(2*nu)*r/lengthscale)*exp(lgamma(order + 1) - lgamma(2*order + 1))

    val sumTerm = (0 to order).map(i => {
      math.pow(sqrt(8*nu)*r/lengthscale, order-i)*(factorial(order+i)/(factorial(i)*factorial(order-i)))
    }).sum

    Map("l" -> {

      val diffLead = leadingTerm * sqrt(2*nu) * r/math.pow(lengthscale, 2.0)
      val diffSum = (0 to order).map(i => {
        -1*(order-i) *
          math.pow(sqrt(8*nu)* r/lengthscale, order-i) *
          (factorial(order+i)/(factorial(i)*factorial(order-i)))/lengthscale
      }).sum

      diffLead*sumTerm + diffSum*leadingTerm
    })
  }
}

/**
  * @author mandar2812 date: 27/01/2017.
  *
  * Implementation of the half integer Matern-ARD
  * covariance function
  */
abstract class GenericMaternARDKernel[T](private var l: T, private var p: Int = 3)(
  implicit ev: Field[T] with InnerProductSpace[T, Double])
  extends StationaryKernel[T, Double, DenseMatrix[Double]]
    with LocalScalarKernel[T] with Serializable { self =>

  blocked_hyper_parameters = List("p")

}

