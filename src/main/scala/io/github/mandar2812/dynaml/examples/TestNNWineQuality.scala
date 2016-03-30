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
package io.github.mandar2812.dynaml.examples

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.evaluation.BinaryClassificationMetrics
import io.github.mandar2812.dynaml.models.neuralnets.{FFNeuralGraph, FeedForwardNetwork}
import io.github.mandar2812.dynaml.pipes.{DataPipe, DynaMLPipe, StreamDataPipe}

/**
  * Created by mandar on 11/1/16.
  */
object TestNNWineQuality {
  def apply (hidden: Int = 2, nCounts:List[Int] = List(), acts:List[String],
             training: Int = 100, test: Int = 1000,
             columns: List[Int] = List(11,0,1,2,3,4,5,6,7,8,9,10),
             stepSize: Double = 0.01, maxIt: Int = 30, mini: Double = 1.0,
             alpha: Double = 0.5, regularization: Double = 0.5,
             wineType: String = "red"): Unit = {

    //Load wine quality data into a stream
    //Extract the time and Dst values
    //separate data into training and test
    //pipe training data to model and then generate test predictions
    //create RegressionMetrics instance and produce plots

    val modelTrainTest =
      (trainTest: ((Stream[(DenseVector[Double], Double)],
        Stream[(DenseVector[Double], Double)]),
        (DenseVector[Double], DenseVector[Double]))) => {

        val gr = FFNeuralGraph(trainTest._1._1.head._1.length, 1, hidden,
          acts ++ List("tansig"), nCounts)

        val transform = DataPipe(
          (d: Stream[(DenseVector[Double], Double)]) =>
            d.map(el => (el._1, DenseVector(el._2)))
        )

        val model = new FeedForwardNetwork[
          Stream[(DenseVector[Double], Double)]
          ](trainTest._1._1, gr, transform)

        model.setLearningRate(stepSize)
          .setMaxIterations(maxIt)
          .setBatchFraction(mini)
          .setMomentum(alpha)
          .setRegParam(regularization)
          .learn()

        val res = model.test(trainTest._1._2)

        val scoresAndLabelsPipe =
          DataPipe(
            (res: Seq[(DenseVector[Double], DenseVector[Double])]) =>
              res.map(i => (i._1(0), i._2(0))).toList) >
            DataPipe((list: List[(Double, Double)]) =>
              list.map{l => (l._1*trainTest._2._2(-1) + trainTest._2._1(-1),
                l._2*trainTest._2._2(-1) + trainTest._2._1(-1))})

        val scoresAndLabels = scoresAndLabelsPipe.run(res)

        val metrics = new BinaryClassificationMetrics(
          scoresAndLabels,
          scoresAndLabels.length)

        metrics.setName(wineType+" wine quality")
        metrics.print()
        metrics.generatePlots()
      }

    val preProcessPipe = DynaMLPipe.fileToStream >
      DynaMLPipe.dropHead >
      DynaMLPipe.replace(";", ",") >
      DynaMLPipe.extractTrainingFeatures(columns, Map()) >
      DynaMLPipe.splitFeaturesAndTargets >
      StreamDataPipe((pattern:(DenseVector[Double], Double)) =>
        if(pattern._2 <= 6.0) (pattern._1, -1.0) else (pattern._1, 1.0))

    val trainTestPipe = DataPipe(preProcessPipe, preProcessPipe) >
      DynaMLPipe.splitTrainingTest(training, test) >
      DynaMLPipe.trainTestGaussianStandardization >
      DataPipe(modelTrainTest)

    trainTestPipe run
      ("data/winequality-"+wineType+".csv",
        "data/winequality-"+wineType+".csv")

  }

}
