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
package io.github.tailhq.dynaml.examples

import breeze.linalg.{DenseMatrix, DenseVector => BDV}
import io.github.tailhq.dynaml.DynaMLPipe
import io.github.tailhq.dynaml.evaluation.BinaryClassificationMetrics
import io.github.tailhq.dynaml.modelpipe.GLMPipe
import io.github.tailhq.dynaml.models.lm.GeneralizedLinearModel
import io.github.tailhq.dynaml.pipes._

object TestAdultLogistic {

  def apply(
    training: Int = 1000,
    columns: List[Int] = List(6, 0, 1, 2, 3, 4, 5),
    stepSize: Double = 0.01,
    maxIt: Int = 30,
    mini: Double = 1.0,
    regularization: Double = 0.5,
    modelType: String = "logistic"
  ) = {

    val modelpipe = new GLMPipe(
      (tt: (
        (Iterable[(BDV[Double], Double)], Iterable[(BDV[Double], Double)]),
        (BDV[Double], BDV[Double])
      )) => tt._1._1.toStream,
      task = "classification",
      modelType = modelType
    ) > DynaMLPipe.trainParametricModel[
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
        Iterable[(BDV[Double], Double)]
      )) => {

        val pipe1 = StreamDataPipe((couple: (BDV[Double], Double)) => {
          (modelAndData._1.predict(couple._1), couple._2)
        })

        val scoresAndLabelsPipe = pipe1
        val scoresAndLabels =
          scoresAndLabelsPipe.run(modelAndData._2.toStream).toList

        val metrics = new BinaryClassificationMetrics(
          scoresAndLabels,
          scoresAndLabels.length,
          logisticFlag = true
        )

        metrics.setName("Adult Income")
        metrics.print()
        metrics.generatePlots()

      }
    )

    val preProcessPipe = DynaMLPipe.fileToStream >
      DynaMLPipe.extractTrainingFeatures(columns, Map()) >
      DynaMLPipe.splitFeaturesAndTargets

    val scaleFeatures = IterableDataPipe(
      (pattern: (BDV[Double], Double)) =>
        (pattern._1, math.max(pattern._2, 0.0))
    )

    val procTraining = preProcessPipe >
      DataPipe((data: Iterable[(BDV[Double], Double)]) => data.take(training)) >
      scaleFeatures

    val procTest = preProcessPipe > scaleFeatures

    val trainTestPipe = DataPipe(procTraining, procTest) >
      DynaMLPipe.featuresGaussianStandardization >
      BifurcationPipe(
        modelpipe,
        DataPipe(
          (tt: (
            (Iterable[(BDV[Double], Double)], Iterable[(BDV[Double], Double)]),
            (BDV[Double], BDV[Double])
          )) => tt._1._2
        )
      ) >
      testPipe

    trainTestPipe(("data/adult.csv", "data/adulttest.csv"))

  }

}
