package io.github.mandar2812.dynaml.kernels

import breeze.linalg.DenseMatrix
import breeze.numerics.{exp, lgamma, sqrt}
import spire.algebra.{Field, InnerProductSpace}

import io.github.mandar2812.dynaml.utils._

/**
  * Implementation of the half integer Matern
  * covariance function
  *
  * @author mandar2812 date: 27/01/2017.
  *
  *
  * */
class GenericMaternKernel[T](private var l: Double, p: Int = 2)(
  implicit evInner: InnerProductSpace[T, Double], evField: Field[T])
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

  override def evalAt(config: Map[String, Double])(x: T) = {
    val r = math.sqrt(evInner.dot(x, x))
    val nu = config("p") + 0.5
    val lengthscale = config("l")
    val order = config("p").toInt

    val leadingTerm = exp(-sqrt(2*nu)*r/lengthscale)*exp(lgamma(order + 1) - lgamma(2*order + 1))

    val sumTerm = (0 to order).map(i => {
      math.pow(sqrt(8*nu)*r/lengthscale, order-i)*(factorial(order+i)/(factorial(i)*factorial(order-i)))
    }).sum

    leadingTerm*sumTerm
  }

  override def gradientAt(config: Map[String, Double])(x: T, y: T) = {

    val diff = evField.minus(x, y)
    val r = math.sqrt(evInner.dot(diff, diff))

    val nu = config("p") + 0.5
    val lengthscale = config("l")
    val order = config("p").toInt

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
  * Implementation of the half integer Matern-ARD
  * covariance function
  *
  * @author mandar2812 date: 27/01/2017.
  *
  *
  * */
abstract class GenericMaternARDKernel[T](private var l: T, private var p: Int = 3)(
  implicit evInner: InnerProductSpace[T, Double], evField: Field[T])
  extends StationaryKernel[T, Double, DenseMatrix[Double]]
    with LocalScalarKernel[T] with Serializable { self =>

  blocked_hyper_parameters = List("p")

}

