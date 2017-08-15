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
import io.github.mandar2812.dynaml.utils._
import io.github.mandar2812.dynaml.analysis._


/**
  * Utilities for computational real analysis
  * 
  * @author mandar2812 date 2017/08/15
  * */
package object analysis {

  /**
    * Generates a fourier series feature mapping.
    * */
  object FourierSeriesGenerator extends MetaPipe21[Double, Int, Double, DenseVector[Double]] {

    override def run(omega: Double, components: Int): DataPipe[Double, DenseVector[Double]] = {

      DataPipe((x: Double) => {
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

  }

  /**
    * Generates a polynomial feature mapping upto a specified degree.
    * */
  object PolynomialSeriesGenerator extends MetaPipe[Int, Double, DenseVector[Double]] {

    override def run(degree: Int): DataPipe[Double, DenseVector[Double]] = {

      DataPipe((x: Double) => DenseVector((0 to degree).map(d => math.pow(x, d)).toArray))
    }
  }

  /**
    * Generate a basis of Cardinal Cubic B-Splines 
    * */
  object CubicSplineGenerator extends MetaPipe[Seq[Int], Double, DenseVector[Double]] {

    override def run(knotIndices: Seq[Int]): DataPipe[Double, DenseVector[Double]] =
      DataPipe((x: Double) => DenseVector(Array(1d) ++ knotIndices.toArray.map(k => CardinalBSplineGenerator(k, 3)(x))))

  }

  /**
    * Generate a basis of Chebyshev functions
    * */
  object ChebyshevSeriesGenerator extends MetaPipe21[Int, Int, Double, DenseVector[Double]] {

    override def run(maxdegree: Int, kind: Int): DataPipe[Double, DenseVector[Double]] = {
      require(kind == 1 || kind == 2, "Chebyshev functions are either of the first or second kind")
      DataPipe((x: Double) => DenseVector((0 to maxdegree).map(d => if(d == 0) 1d else chebyshev(d,x,kind)).toArray))
    }
  }

}
