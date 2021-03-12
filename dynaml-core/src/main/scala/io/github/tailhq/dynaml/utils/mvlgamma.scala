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
package io.github.tailhq.dynaml.utils

import breeze.generic.UFunc
import breeze.numerics.{lgamma, log}

/**
  * @author mandar date: 06/02/2017.
  *
  * Multivariate Gamma function.
  *
  */
object mvlgamma extends UFunc {


  def mvlgammaRec(n: Int, a: Double, gammaAcc: Double): Double = n match {
    case 1 => lgamma(a)+gammaAcc
    case _ => mvlgammaRec(n-1, a, gammaAcc + 0.5*(n-1)*log(math.Pi) + lgamma(a + 0.5*(1-n)))
  }

  implicit object ImplDouble extends Impl2[Int, Double, Double] {
    override def apply(v: Int, v2: Double) = mvlgammaRec(v, v2, 0.0)
  }

}
