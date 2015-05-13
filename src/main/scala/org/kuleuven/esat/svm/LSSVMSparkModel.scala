package org.kuleuven.esat.svm

import breeze.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.kuleuven.esat.evaluation.Metrics
import org.kuleuven.esat.graphicalModels.LinearModel
import org.kuleuven.esat.optimization.{GradientDescentSpark, Optimizer}

/**
 * Implementation of the Least Squares SVM
 * using Apache Spark RDDs
 */
abstract class LSSVMSparkModel(data: RDD[(Long, LabeledPoint)]) extends
LinearModel[RDD[(Long, LabeledPoint)], Int, Int, DenseVector[Double],
  DenseVector[Double], Double, RDD[LabeledPoint]]{

  /**
   * Predict the value of the
   * target variable given a
   * point.
   *
   **/
  override def predict(point: DenseVector[Double]): Double = 0.0

  /**
   * Learn the parameters
   * of the model which
   * are in a node of the
   * graph.
   *
   **/
  override def learn(): Unit = {}

  override protected val optimizer: Optimizer[Int,
    DenseVector[Double], DenseVector[Double], Double,
    RDD[LabeledPoint]] = new GradientDescentSpark(null, null)

  override protected var params: DenseVector[Double] = _

  override protected val g: RDD[(Long, LabeledPoint)] = data
}
