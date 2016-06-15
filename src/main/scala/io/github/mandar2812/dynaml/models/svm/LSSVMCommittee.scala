package io.github.mandar2812.dynaml.models.svm

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.models.DLSSVMPipe
import io.github.mandar2812.dynaml.models.ensemble.CommitteeModel
import io.github.mandar2812.dynaml.optimization.{GlobalOptimizer, GridSearch, RDDCommitteeSolver, RegularizedOptimizer}
import org.apache.spark.rdd.RDD

/**
  * Created by mandar on 3/6/16.
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

  var modelTuners: List[GlobalOptimizer[DLSSVM]] =
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
