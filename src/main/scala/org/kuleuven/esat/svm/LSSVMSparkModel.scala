package org.kuleuven.esat.svm

import breeze.linalg.DenseVector
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.kuleuven.esat.evaluation.Metrics
import org.kuleuven.esat.graphicalModels.{GaussianLinearModel, LinearModel}
import org.kuleuven.esat.optimization.{SquaredL2Updater, LeastSquaresSVMGradient, GradientDescentSpark, Optimizer}
import org.apache.spark.mllib.linalg.Vector

/**
 * Implementation of the Least Squares SVM
 * using Apache Spark RDDs
 */
class LSSVMSparkModel(data: RDD[(Long, LabeledPoint)], task: String) extends
LinearModel[RDD[(Long, LabeledPoint)], Int, Int, DenseVector[Double],
  DenseVector[Double], Double, RDD[LabeledPoint]]{

  override protected val optimizer: Optimizer[Int,
    DenseVector[Double], DenseVector[Double], Double,
    RDD[LabeledPoint]] = new GradientDescentSpark(
    new LeastSquaresSVMGradient(),
    new SquaredL2Updater())


  override protected var params: DenseVector[Double] = DenseVector.ones(featuredims+1)

  override protected val g: RDD[(Long, LabeledPoint)] = data

  protected var featuredims: Int = data.first()._2.features.size

  protected val _nPoints = data.count()

  protected val _task = task

  protected val _data = data
  /**
   * Predict the value of the
   * target variable given a
   * point.
   *
   **/
  override def predict(point: DenseVector[Double]): Double =
    LSSVMSparkModel.predictBDV(params)(point)(_task)

  /**
   * Learn the parameters
   * of the model which
   * are in a node of the
   * graph.
   *
   **/
  override def learn(): Unit = {
    params = this.optimizer.optimize(_nPoints, params, this.data.map(_._2))
  }

  override def clearParameters: Unit = {
    params = DenseVector.ones[Double](featuredims+1)
  }
  
  override def evaluate(config: Map[String, String]) = {
    val sc = _data.context
    val (file, delim, head, _) = GaussianLinearModel.readConfig(config)
    val test_data = sc.textFile(file).map(line => line split delim)
      .filter(vector => vector.head forall Character.isDigit)
      .map(_.map(_.toDouble)).map(vector => LabeledPoint(vector(vector.length-1), 
      Vectors.dense(vector.slice(0, vector.length-1))))
    
    val results = test_data.map(point =>
      LSSVMSparkModel.predict(params)(point)(_task))
      .collect()
    Metrics(_task)(results.toList, results.length)
  }

}

object LSSVMSparkModel {
    
  def predict(params: DenseVector[Double])
             (point: LabeledPoint)
             (implicit _task: String): (Double, Double) = {
    val margin: Double = predictSparkVector(params)(point.features)(_task)
    val loss = point.label - margin
    (margin, math.pow(loss, 2))
  }

  def predictBDV(params: DenseVector[Double])
             (point: DenseVector[Double])
             (implicit _task: String): Double = {
    val margin: Double = params dot DenseVector.vertcat(
      point,
      DenseVector(1.0))
    _task match {
      case "classification" => math.tanh(margin)
      case "regression" => margin

    }
  }

  def predictSparkVector(params: DenseVector[Double])
                (point: Vector)
                (implicit _task: String): Double =
  predictBDV(params)(DenseVector(point.toArray))(_task)
}
