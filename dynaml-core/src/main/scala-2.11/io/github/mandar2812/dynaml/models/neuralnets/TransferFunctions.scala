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
package io.github.mandar2812.dynaml.models.neuralnets

import breeze.linalg.DenseVector
import breeze.numerics.{cosh, sigmoid, sinh, tanh}
import io.github.mandar2812.dynaml.pipes.Scaler
import org.apache.spark.annotation.Experimental

/**
 * @author mandar2812
 *
 * Object implementing neural network transfer functions.
 */

object TransferFunctions {

  val SIGMOID = "logsig"

  val TANH = "tansig"

  val LIN = "linear"

  val RECLIN = "reclinear"

  /**
   * Hyperbolic tangent function
   * */
  val tansig = (x:Double) => tanh(x)

  /**
    * First derivative of the hyperbolic tangent function
    * */
  val Dtansig = (x: Double) => tansig(x)/(sinh(x)*cosh(x))

  /**
   * Sigmoid/Logistic function
   *
   * */
  val logsig = (x:Double) => sigmoid(x)

  /**
    * First derivative of the sigmoid function
    * */
  val Dlogsig = (x:Double) => sigmoid(x)*(1-sigmoid(x))

  /**
   * Linear/Identity Function
   * */
  val lin = (x: Double) => x

  /**
    * First derivative of the linear function
    * */
  val Dlin = (_: Double) => 1.0

  /**
    * Rectified linear Function
    * */
  val recLin = (x: Double) => math.max(0.0, x)

  /**
    * First derivative of the rectified linear function
    * */
  val DrecLin = (x: Double) => if(x < 0 ) 0.0 else 1.0

  /**
   * Function which returns
   * the appropriate activation
   * for a given string
   * */
  def getActivation(s: String): (Double) => Double =
    s.toLowerCase match {
      case "sigmoid" => logsig
      case "logsig" => logsig
      case "tansig" => tansig
      case "linear" => lin
      case "reclinear" => recLin
    }

  /**
    * Function which returns
    * the appropriate derivative
    * activation for a given string
    * */
  def getDiffActivation(s: String): (Double) => Double =
    s.toLowerCase match {
      case "sigmoid" => Dlogsig
      case "logsig" => Dlogsig
      case "tansig" => Dtansig
      case "linear" => Dlin
      case "reclinear" => DrecLin
    }
}

trait Activation[I] extends Scaler[I] {
  val grad: Scaler[I]
}

object Activation {

  def apply[I](f: (I) => I, gr: (I) => I): Activation[I] = new Activation[I] {
    override val grad = Scaler(gr)

    override def run(data: I) = f(data)
  }

  def apply[I](f: Scaler[I], gr: Scaler[I]): Activation[I] = apply(f.run _, gr.run _)
}

object Sigmoid extends Activation[Double] {

  override val grad = Scaler(TransferFunctions.Dlogsig)

  override def run(data: Double) = TransferFunctions.logsig(data)
}

object VectorSigmoid extends Activation[DenseVector[Double]] {

  override val grad = Scaler((x: DenseVector[Double]) => x.map(TransferFunctions.Dlogsig))

  override def run(data: DenseVector[Double]) = data.map(TransferFunctions.logsig)
}

object VectorTansig extends Activation[DenseVector[Double]] {

  override val grad = Scaler((x: DenseVector[Double]) => x.map(TransferFunctions.Dtansig))

  override def run(data: DenseVector[Double]) = data.map(TransferFunctions.tansig)
}

object VectorLinear extends Activation[DenseVector[Double]] {

  override val grad = Scaler((x: DenseVector[Double]) => x.map(TransferFunctions.Dlin))

  override def run(data: DenseVector[Double]) = data.map(TransferFunctions.lin)
}

object VectorRecLin extends Activation[DenseVector[Double]] {

  override val grad = Scaler((x: DenseVector[Double]) => x.map(TransferFunctions.DrecLin))

  override def run(data: DenseVector[Double]) = data.map(TransferFunctions.recLin)
}

@Experimental
class VectorWavelet(motherWavelet: (Double) => Double, motherWaveletGr: (Double) => Double)
  extends Activation[DenseVector[Double]] {

  override val grad = Scaler((x: DenseVector[Double]) => x.map(motherWaveletGr))

  override def run(data: DenseVector[Double]) = data.map(motherWavelet)
}

