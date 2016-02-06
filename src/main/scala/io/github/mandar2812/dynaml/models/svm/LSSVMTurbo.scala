package io.github.mandar2812.dynaml.models.svm

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
 * FS-LSSVM Turbo:
 * This class extends the basic functionality of the LSSVMSparkModel
 * to enable the training and evaluation of multiple model instances
 * in a single run through the training set.
 */
abstract class LSSVMTurbo(data: RDD[LabeledPoint], task: String)
  extends LSSVMSparkModel(data: RDD[LabeledPoint], task: String)
  with Serializable {

  //No need to use the processed_g variable
  //as it allows us to cahche only a particular
  //feature map applied to the data.
  this.processed_g = null

  def evaluateGrid(grid: Seq[Map[String, Double]]): Unit = {}


}

/*object LSSVMTurbo {

  def apply(implicit config: Map[String, String], sc: SparkContext): LSSVMTurbo = {
    val (data, task) = LSSVMSparkModel.generateRDD(config, sc)
    new LSSVMTurbo(data, task)
  }
}*/
