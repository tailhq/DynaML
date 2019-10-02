/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
* */
package io.github.mandar2812.dynaml

import breeze.linalg._
import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.probability.RandomVariable
import io.github.mandar2812.dynaml.utils._


/**
  * Utilities for computational real analysis
  * 
  * @author mandar2812 date 2017/08/15
  * */
package object analysis {

  /**
    * Generates a fourier series feature mapping.
    * */
  object FourierBasisGenerator extends DataPipe2[Double, Int, Basis[Double]] {

    override def run(omega: Double, components: Int): Basis[Double] =
      Basis((x: Double) => {
        if(components % 2 == 0) {
          DenseVector(
            (Seq(1d) ++
              (-components/2 to components/2)
              .filterNot(_ == 0)
              .map(i =>
                if(i < 0) math.cos(i*omega*x)
                else math.sin(i*omega*x))
            ).toArray
          )
        } else {
          DenseVector(
            (Seq(1d) ++
              (-(components/2).toInt - 1 to components/2)
              .filterNot(_ == 0)
              .map(i =>
                if(i < 0) math.cos(i*omega*x)
                else math.sin(i*omega*x))
            ).toArray
          )
        }
      })

  }

  /**
    * Generates a polynomial feature mapping upto a specified degree.
    * */
  object PolynomialBasisGenerator extends DataPipe[Int, Basis[Double]] {

    override def run(degree: Int): Basis[Double] = {

      Basis((x: Double) => DenseVector((0 to degree).map(d => math.pow(x, d)).toArray))
    }
  }

  /**
    * Generate a basis of Cardinal Cubic B-Splines 
    * */
  object CubicSplineGenerator extends DataPipe[Seq[Int], Basis[Double]] {

    override def run(knotIndices: Seq[Int]): Basis[Double] =
      Basis((x: Double) => DenseVector(Array(1d) ++ knotIndices.toArray.map(k => CardinalBSplineGenerator(k, 3)(x))))

  }

  /**
    * Generate a basis of Bernstein Polynomials
    * */
  object BernsteinSplineSeriesGenerator extends DataPipe[Seq[Int], Basis[Double]] {

    override def run(knotIndices: Seq[Int]): Basis[Double] =
      Basis((x: Double) => DenseVector(Array(1d) ++ knotIndices.toArray.map(k => BernsteinSplineGenerator(k, 3)(x))))
  }

  /**
    * Generate a basis of Chebyshev functions
    * */
  object ChebyshevBasisGenerator extends DataPipe2[Int, Int, Basis[Double]] {

    override def run(maxdegree: Int, kind: Int): Basis[Double] = {
      require(kind == 1 || kind == 2, "Chebyshev functions are either of the first or second kind")
      Basis(
        (x: Double) =>
        DenseVector(
          (0 to maxdegree).map(d => chebyshev(d,x,kind)).toArray
        )
      )
    }
  }

  /**
    * Generate a hermite polynomial basis.
    *
    * */
  object HermiteBasisGenerator extends DataPipe[Int, Basis[Double]] {

    override def run(maxdegree: Int): Basis[Double] =
      Basis(
        (x: Double) =>
        DenseVector(
          (0 to maxdegree).map(d => hermite(d, x)).toArray
        )
      )
  }

  /**
    * Generate a legendre polynomial basis
    * */
  object LegendreBasisGenerator extends DataPipe[Int, Basis[Double]] {
    override def run(maxdegree: Int): Basis[Double] =
      Basis(
        (x: Double) =>
          DenseVector(
            (0 to maxdegree).map(d => legendre(d, x)).toArray
          )
      )
  }

  sealed class QuadratureRule(
    val nodes: Seq[Double],
    val weights: Seq[Double]) {


    protected def scale_nodes_and_weights(lower: Double, upper: Double): (Seq[Double], Seq[Double]) = {
      val sc_nodes = nodes.map(n => {
        val mid_point = (lower + upper) / 2d

        val mid_diff  = (upper - lower) / 2d

        mid_point + mid_diff*n
      })

      val sc_weights = weights.map(_*(upper - lower)/2d)

      (sc_nodes, sc_weights)
    }

    def scale(lower: Double, upper: Double): QuadratureRule = {

      val (sc_nodes, sc_weights) = scale_nodes_and_weights(lower, upper)

      new QuadratureRule(nodes = sc_nodes, weights = sc_weights)
    }

    def integrate(f: Double => Double): Double = {

      weights.zip(nodes.map(f)).map(c => c._2*c._1).sum
    }
  }

  case class MonteCarloQuadrature(override val nodes: Seq[Double]) extends
    QuadratureRule(nodes, Seq.fill(nodes.length)(1d/nodes.length)) {

    self =>

    override def scale(lower: Double, upper: Double): MonteCarloQuadrature = {

      val q = super.scale(lower, upper)

      self.copy(nodes = q.nodes)
    }

  }

  val monte_carlo_quadrature: MetaPipe[RandomVariable[Double], Int, MonteCarloQuadrature] =
    MetaPipe((rv: RandomVariable[Double]) => (n: Int) => MonteCarloQuadrature(rv.iid(n).draw.toSeq))

  case class GaussianQuadrature(
    override val nodes: Seq[Double],
    override val weights: Seq[Double]) extends
    QuadratureRule(nodes, weights) {

    self =>

    override def scale(lower: Double, upper: Double): GaussianQuadrature = {

      val q = super.scale(lower, upper)

      self.copy(nodes = q.nodes, weights = q.weights)
    }

  }


  val twoPointGaussLegendre = GaussianQuadrature(
    Seq(-0.5773502692d, 0.5773502692d),
    Seq( 1d,            1d)
  )

  val threePointGaussLegendre = GaussianQuadrature(
    Seq(-0.7745966692, 0d,           0.7745966692),
    Seq( 0.5555555556, 0.8888888888, 0.5555555556)
  )


  val fourPointGaussLegendre = GaussianQuadrature(
    Seq(-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116),
    Seq( 0.3478548451,  0.6521451549, 0.6521451549, 0.3478548451)
  )

  val fivePointGaussLegendre = GaussianQuadrature(
    Seq(-0.9061798459, -0.5384693101, 0d,           0.5384693101, 0.9061798459),
    Seq( 0.2369268851,  0.4786286705, 0.5688888888, 0.4786286705, 0.2369268851)
  )

  val sixPointGaussLegendre = GaussianQuadrature(
    Seq(-0.9324695142, -0.6612093865, -0.2386191861, 0.2386191861, 0.6612093865, 0.9324695142),
    Seq( 0.1713244924,  0.3607615730,  0.4679139346, 0.4679139346, 0.3607615730, 0.1713244924)
  )

  val sevenPointGaussLegendre = GaussianQuadrature(
    Seq(-0.9491079123, -0.7415311856, -0.4058451514, 0d,           0.4058451514, 0.7415311856, 0.9491079123),
    Seq( 0.1294849662,  0.2797053915,  0.3818300505, 0.4179591837, 0.3818300505, 0.2797053915, 0.1294849662)
  )

  val eightPointGaussLegendre = GaussianQuadrature(
    Seq(
      -0.9602898565, -0.7966664774, -0.5255324099, -0.1834346425, 0.1834346425, 0.5255324099, 0.7966664774, 0.9602898565
    ),
    Seq(
      0.1012285363,  0.2223810345,  0.3137066459,  0.3626837834, 0.3626837834, 0.3137066459, 0.2223810345, 0.1012285363
    )
  )

}
