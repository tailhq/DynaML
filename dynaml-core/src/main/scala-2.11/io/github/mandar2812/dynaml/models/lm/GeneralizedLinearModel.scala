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
package io.github.mandar2812.dynaml.models.lm

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.evaluation.Metrics
import io.github.mandar2812.dynaml.models.LinearModel
import io.github.mandar2812.dynaml.optimization.GloballyOptimizable

import scala.util.Random

/**
  * Created by mandar on 4/4/16.
  */
abstract class GeneralizedLinearModel[T](data: Stream[(DenseVector[Double], Double)],
                             numPoints: Int,
                             map: (DenseVector[Double]) => DenseVector[Double] =
                             identity[DenseVector[Double]] _)
  extends LinearModel[Stream[(DenseVector[Double], Double)],
    DenseVector[Double], DenseVector[Double], Double, T]
    with GloballyOptimizable {

  override protected val g = data

  val task: String

  val h: (Double) => Double = identity _

  featureMap = map

  def dimensions = featureMap(data.head._1).length

  override def initParams(): DenseVector[Double] =
    DenseVector.ones[Double](dimensions + 1)


  override protected var params: DenseVector[Double] = initParams()

  override def clearParameters(): Unit = {
    params = initParams()
  }

  def prepareData(d: Stream[(DenseVector[Double], Double)]): T


  /**
    * Learn the parameters
    * of the model which
    * are in a node of the
    * graph.
    *
    **/
  override def learn(): Unit = {
    params = optimizer.optimize(numPoints,
      prepareData(g), initParams())
  }

  /**
    * Predict the value of the
    * target variable given a
    * point.
    *
    **/
  override def predict(point: DenseVector[Double]): Double =
    h(params dot DenseVector(featureMap(point).toArray ++ Array(1.0)))

  override protected var hyper_parameters: List[String] =
    List("regularization")

  /**
    * Calculates the energy of the configuration,
    * in most global optimization algorithms
    * we aim to find an approximate value of
    * the hyper-parameters such that this function
    * is minimized.
    *
    * @param h       The value of the hyper-parameters in the configuration space
    * @param options Optional parameters about configuration
    * @return Configuration Energy E(h)
    **/
  override def energy(h: Map[String, Double],
                      options: Map[String, String]): Double = {

    setState(h)
    val folds: Int = options("folds").toInt
    val shuffle = Random.shuffle((1L to numPoints).toList)

    val avg_metrics: DenseVector[Double] = (1 to folds).map { a =>
      //For the ath fold
      //partition the data
      //ceil(a-1*npoints/folds) -- ceil(a*npoints/folds)
      //as test and the rest as training
      val test = shuffle.slice((a - 1) * numPoints / folds, a * numPoints / folds)
      val (trainingData, testData) = g.zipWithIndex.partition((c) => !test.contains(c._2))
      val tempParams = optimizer.optimize(numPoints,
        prepareData(trainingData.map(_._1)),
        initParams())

      val scoresAndLabels = testData.map(_._1).map(p =>
        (this.h(tempParams dot DenseVector(featureMap(p._1).toArray ++ Array(1.0))), p._2))

      val metrics = Metrics("classification")(
        scoresAndLabels.toList,
        testData.length,
        logFlag = true)
      val res: DenseVector[Double] = metrics.kpi() / folds.toDouble
      res
    }.reduce(_+_)
    //Perform n-fold cross validation

    task match {
      case "regression" => avg_metrics(1)
      case "classification" => 1 - avg_metrics(2)
    }
  }

  override protected var current_state: Map[String, Double] = Map("regularization" -> 0.001)

  /**
    * Set the model "state" which
    * contains values of its hyper-parameters
    * with respect to the covariance and noise
    * kernels.
    * */
  def setState(s: Map[String, Double]): this.type ={
    this.setRegParam(s("regularization"))
    current_state = Map("regularization" -> s("regularization"))
    this
  }
}

object GeneralizedLinearModel {
  def apply[T](data: Stream[(DenseVector[Double], Double)],
               task: String = "regression",
               map: (DenseVector[Double]) => DenseVector[Double] =
               identity[DenseVector[Double]] _,
               modeltype: String = "") = task match {
    case "regression" => new RegularizedGLM(data, data.length, map).asInstanceOf[GeneralizedLinearModel[T]]
    case "classification" => task match {
      case "probit" => new ProbitGLM(data, data.length, map).asInstanceOf[GeneralizedLinearModel[T]]
      case _ => new LogisticGLM(data, data.length, map).asInstanceOf[GeneralizedLinearModel[T]]
    }
  }
}
