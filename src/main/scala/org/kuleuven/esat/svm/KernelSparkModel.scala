package org.kuleuven.esat.svm

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
 * Implementation of the Fixed Size
 * Kernel based LS SVM
 *
 * Fixed Size implies that the model
 * chooses a subset of the original
 * data to calculate a low rank approximation
 * to the kernel matrix.
 *
 * Feature Extraction is done in the primal
 * space using the Nystrom approximation.
 *
 * @author mandar2812
 */
class KernelSparkModel(data: RDD[LabeledPoint], task: String)
  extends LSSVMSparkModel(data: RDD[LabeledPoint], task: String) {


}

object KernelSparkModel {
  def apply(data: RDD[LabeledPoint], task: String): KernelSparkModel = new KernelSparkModel(data, task)
}
