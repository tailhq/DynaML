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

import breeze.linalg.{DenseVector => BDV}
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.graph.FFNeuralGraph
import io.github.mandar2812.dynaml.modelpipe.GLMPipe
import io.github.mandar2812.dynaml.models.lm.GeneralizedLinearModel
import io.github.mandar2812.dynaml.models.neuralnets.{
  Activation,
  FeedForwardNetwork,
  GenericFFNeuralNet,
  NeuralStackFactory
}
import io.github.mandar2812.dynaml.pipes.{DataPipe, _}
import io.github.mandar2812.dynaml.utils
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.optimization.FFBackProp
import io.github.mandar2812.dynaml.utils.GaussianScaler

/**
  * Created by mandar on 11/1/16.
  */
object TestNNDelve {

  implicit val transform: DataPipe[Stream[(BDV[Double], Double)], Stream[
    (BDV[Double], BDV[Double])
  ]] =
    DataPipe(
      (d: Stream[(BDV[Double], Double)]) => d.map(el => (el._1, BDV(el._2)))
    )

  def apply(
    hidden: Int = 2,
    nCounts: List[Int] = List(),
    acts: List[String],
    training: Int = 100,
    test: Int = 1000,
    columns: List[Int] = List(10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    stepSize: Double = 0.01,
    maxIt: Int = 30,
    mini: Double = 1.0,
    alpha: Double = 0.5,
    regularization: Double = 0.5
  ): Unit = {

    //Load Housing data into a stream
    //Extract the time and Dst values
    //separate data into training and test
    //pipe training data to model and then generate test predictions
    //create RegressionMetrics instance and produce plots

    val extractTrainingFeatures = (l: Stream[String]) =>
      utils.extractColumns(l, ",", columns, Map())

    val normalizeData =
      (trainTest: (
        Iterable[(BDV[Double], Double)],
        Iterable[(BDV[Double], Double)]
      )) => {

        val (mean, variance) = utils.getStats(
          trainTest._1.map(tup => BDV(tup._1.toArray ++ Array(tup._2))).toList
        )

        val stdDev: BDV[Double] =
          variance.map(
            v => math.sqrt(v / (trainTest._1.toSeq.length.toDouble - 1.0))
          )

        val normalizationFunc = (point: (BDV[Double], Double)) => {
          val extendedpoint = BDV(point._1.toArray ++ Array(point._2))

          val normPoint = (extendedpoint - mean) :/ stdDev
          val length    = normPoint.length
          (normPoint(0 until length), normPoint(-1))
        }

        (
          (
            trainTest._1.map(normalizationFunc),
            trainTest._2.map(normalizationFunc)
          ),
          (mean, stdDev)
        )
      }

    val modelTrainTest =
      (trainTest: (
        (Iterable[(BDV[Double], Double)], Iterable[(BDV[Double], Double)]),
        (BDV[Double], BDV[Double])
      )) => {

        val gr = FFNeuralGraph(
          trainTest._1._1.head._1.length,
          1,
          hidden,
          acts,
          nCounts
        )

        val model = new FeedForwardNetwork[Stream[(BDV[Double], Double)]](
          trainTest._1._1.toStream,
          gr
        )

        model
          .setLearningRate(stepSize)
          .setMaxIterations(maxIt)
          .setBatchFraction(mini)
          .setMomentum(alpha)
          .setRegParam(regularization)
          .learn()

        val res = model.test(trainTest._1._2.toStream)
        val scoresAndLabelsPipe =
          DataPipe(
            (res: Seq[(BDV[Double], BDV[Double])]) =>
              res.map(i => (i._1(0), i._2(0))).toList
          ) > DataPipe(
            (list: List[(Double, Double)]) =>
              list.map { l =>
                (
                  l._1 * trainTest._2._2(-1) + trainTest._2._1(-1),
                  l._2 * trainTest._2._2(-1) + trainTest._2._1(-1)
                )
              }
          )

        val scoresAndLabels = scoresAndLabelsPipe.run(res)

        val metrics =
          new RegressionMetrics(scoresAndLabels, scoresAndLabels.length)

        metrics.print()
        metrics.generatePlots()
      }

    val preProcessPipe = DataPipe(utils.textFileToStream _) >
      DataPipe(extractTrainingFeatures) >
      IterableDataPipe((line: String) => {
        val split = line.split(",")
        (BDV(split.tail.map(_.toDouble)), split.head.toDouble)
      })

    val trainTestPipe = DataPipe(preProcessPipe, preProcessPipe) >
      DataPipe(
        (data: (
          Iterable[(BDV[Double], Double)],
          Iterable[(BDV[Double], Double)]
        )) => {
          (data._1.take(training), data._2.takeRight(test))
        }
      ) >
      DataPipe(normalizeData) >
      DataPipe(modelTrainTest)

    trainTestPipe.run(("data/delve.csv", "data/delve.csv"))

  }

}

object TestNeuralStackDelve {

  def apply(
    nCounts: List[Int],
    acts: List[Activation[BDV[Double]]],
    training: Int = 100,
    test: Int = 1000,
    columns: List[Int] = List(10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    stepSize: Double = 0.01,
    maxIt: Int = 30,
    mini: Double = 1.0,
    alpha: Double = 0.5,
    regularization: Double = 0.5
  ) = {

    type Features = BDV[Double]
    type Data     = Iterable[(Features, Features)]
    type Scales   = (GaussianScaler, GaussianScaler)

    val readData = fileToStream >
      extractTrainingFeatures(columns, Map()) >
      splitFeaturesAndTargets >
      IterableDataPipe(
        (pattern: (BDV[Double], Double)) => (pattern._1, BDV(pattern._2))
      )

    val preProcessPipe = duplicate(readData) >
      splitTrainingTest(training, test) >
      gaussianScalingTrainTest

    val modelTrainTestPipe = DataPipe((dataAndScales: (Data, Data, Scales)) => {
      /*
       * First create the elements needed for
       * the neural architecture:
       *
       * 1. A Stack Factory
       * 2. A random weight generator
       * 3. A learning procedure
       * */

      val (trainingData, testData, scales) = dataAndScales

      /*
       * The stack factory will create the architecture to the specification
       * every time it is given the layer parameters.
       * */
      val stackFactory = NeuralStackFactory(nCounts)(acts)

      val weightsInitializer = GenericFFNeuralNet.getWeightInitializer(nCounts)

      val backPropOptimizer =
        new FFBackProp(stackFactory)
          .setNumIterations(maxIt)
          .setRegParam(regularization)
          .setStepSize(stepSize)
          .setMiniBatchFraction(mini)
          .momentum_(alpha)

      val ff_neural_net = GenericFFNeuralNet(
        backPropOptimizer,
        trainingData,
        DataPipe[Data, Stream[(Features, Features)]](_.toStream),
        weightsInitializer
      )

      /*
       * Train the model
       * */
      ff_neural_net.learn()

      /*
       * Generate predictions for the test data
       * and evaluate performance.
       * */
      val predictionsAndTargets = (scales._2 * scales._2)
        .i(testData.map(p => (ff_neural_net.predict(p._1), p._2)))

      (
        ff_neural_net,
        new RegressionMetrics(
          predictionsAndTargets.map(c => (c._1(0), c._2(0))).toList,
          predictionsAndTargets.toSeq.length
        )
      )

    })

    val dataFlow = preProcessPipe > modelTrainTestPipe

    val dataFile = dataDir + "/delve.csv"
    dataFlow((dataFile, dataFile))
  }

}

object TestGLMDelve {
  def apply(
    training: Int = 100,
    test: Int = 1000,
    columns: List[Int] = List(10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    stepSize: Double = 0.01,
    maxIt: Int = 30,
    mini: Double = 1.0,
    alpha: Double = 0.5,
    regularization: Double = 0.5
  ) = {

    val modelpipe = new GLMPipe(
      (tt: (
        (Iterable[(BDV[Double], Double)], Iterable[(BDV[Double], Double)]),
        (BDV[Double], BDV[Double])
      )) => tt._1._1.toStream
    ) >
      trainParametricModel[
        Stream[(BDV[Double], Double)],
        BDV[Double],
        BDV[Double],
        Double,
        Stream[(BDV[Double], Double)],
        GeneralizedLinearModel[Stream[(BDV[Double], Double)]]
      ](regularization, stepSize, maxIt, mini)

    val testPipe = DataPipe(
      (modelAndData: (
        GeneralizedLinearModel[Stream[(BDV[Double], Double)]],
        (Iterable[(BDV[Double], Double)], BDV[Double], BDV[Double])
      )) => {

        val pipe1 = StreamDataPipe((couple: (BDV[Double], Double)) => {
          (modelAndData._1.predict(couple._1), couple._2)
        })
        val means   = modelAndData._2._2
        val stdDevs = modelAndData._2._3

        val scoresAndLabelsPipe = pipe1 >
          StreamDataPipe(
            (s: (Double, Double)) =>
              (
                (s._1 * stdDevs(-1)) + means(-1),
                (s._2 * stdDevs(-1)) + means(-1)
              )
          )

        val scoresAndLabels =
          scoresAndLabelsPipe.run(modelAndData._2._1.toStream).toList

        val metrics =
          new RegressionMetrics(scoresAndLabels, scoresAndLabels.length)

        metrics.setName("Fried Delve")
        metrics.print()
        metrics.generatePlots()

      }
    )

    val preProcessPipe = fileToStream >
      extractTrainingFeatures(columns, Map()) >
      splitFeaturesAndTargets

    val trainTestPipe = DataPipe(preProcessPipe, preProcessPipe) >
      DataPipe(
        (data: (
          Iterable[(BDV[Double], Double)],
          Iterable[(BDV[Double], Double)]
        )) => {
          (data._1.take(training), data._2.takeRight(test))
        }
      ) >
      trainTestGaussianStandardization >
      BifurcationPipe(
        modelpipe,
        DataPipe(
          (tt: (
            (Iterable[(BDV[Double], Double)], Iterable[(BDV[Double], Double)]),
            (BDV[Double], BDV[Double])
          )) => (tt._1._2, tt._2._1, tt._2._2)
        )
      ) >
      testPipe

    trainTestPipe run ("data/delve.csv", "data/delve.csv")

  }
}
