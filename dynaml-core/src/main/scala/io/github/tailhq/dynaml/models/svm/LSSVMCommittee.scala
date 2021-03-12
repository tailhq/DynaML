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
package io.github.tailhq.dynaml.models.svm

import breeze.linalg.DenseVector
import io.github.tailhq.dynaml.modelpipe.DLSSVMPipe
import io.github.tailhq.dynaml.models.ensemble.CommitteeModel
import io.github.tailhq.dynaml.optimization._
import org.apache.spark.rdd.RDD

/**
  * @author tailhq date 3/6/16.
  */
class LSSVMCommittee(num: Long,
                     data: RDD[(DenseVector[Double], Double)],
                     pipes: DLSSVMPipe[RDD[(DenseVector[Double], Double)]]*) extends
  CommitteeModel[RDD[(DenseVector[Double], Double)],
    Stream[(DenseVector[Double], Double)],
    DLSSVM, DLSSVMPipe[RDD[(DenseVector[Double], Double)]]] (num, data, pipes:_*){

  override protected val optimizer: RegularizedOptimizer[
    DenseVector[Double],
    DenseVector[Double], Double,
    RDD[(DenseVector[Double], Double)]] = new RDDCommitteeSolver

  var modelTuners: List[ModelTuner[DLSSVM, DLSSVM]] =
    baseNetworks.map(m => new GridSearch[DLSSVM](m).setGridSize(10).setStepSize(0.1))

  override def learn(): Unit = {
    //First tune and learn the base SVM models
    (baseNetworks zip modelTuners).foreach(modelCouple => {
      val (_, conf) = modelCouple._2.optimize(modelCouple._1.getState, Map())
      modelCouple._1.setState(conf)
      modelCouple._1.learn()
    })
    //Now learn the committee weights
    val fMap = featureMap
    params = optimizer.optimize(num_points,
      g.map(patternCouple => (fMap(patternCouple._1), patternCouple._2)),
      initParams())
  }

}
