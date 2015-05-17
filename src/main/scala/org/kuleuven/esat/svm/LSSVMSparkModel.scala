package org.kuleuven.esat.svm

import breeze.linalg.DenseVector
import breeze.numerics.sqrt
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.kuleuven.esat.evaluation.Metrics
import org.kuleuven.esat.graphicalModels.{GaussianLinearModel, LinearModel}
import org.kuleuven.esat.kernels.Kernel
import org.kuleuven.esat.optimization._
import org.apache.spark.mllib.linalg.Vector

/**
 * Implementation of the Least Squares SVM
 * using Apache Spark RDDs
 */
class LSSVMSparkModel(data: RDD[LabeledPoint], task: String) extends
LinearModel[RDD[(Long, LabeledPoint)], Int, Int, DenseVector[Double],
  DenseVector[Double], Double, RDD[LabeledPoint]] with Serializable {

  override protected val optimizer = LSSVMSparkModel.getOptimizer(task)

  override protected val g = LSSVMSparkModel.indexedRDD(data)

  protected val _nPoints = data.count()

  protected val _task = task

  protected var featuredims: Int = g.first()._2.features.size

  override protected var params: DenseVector[Double] = DenseVector.ones(featuredims+1)

  val colStats = Statistics.colStats(g.map(_._2.features))

  def dimensions = featuredims

  def npoints = _nPoints

  var featureMap: DenseVector[Double] => DenseVector[Double]
  = identity

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
    val meanb = g.context.broadcast(DenseVector(colStats.mean.toArray))
    val varianceb = g.context.broadcast(DenseVector(colStats.variance.toArray))
    params = this.optimizer.optimize(_nPoints, params,
      this.g.map(point => {
        val vec = DenseVector(point._2.features.toArray)
        val ans = vec - meanb.value
        ans :/= sqrt(varianceb.value)
        new LabeledPoint(
          point._2.label,
          Vectors.dense(DenseVector.vertcat(
            featureMap(ans),
            DenseVector(1.0))
            .toArray)
        )
      })
    )
  }

  override def clearParameters: Unit = {
    params = DenseVector.ones[Double](featuredims+1)
  }

  def setRegParam(l: Double): this.type = {
    this.optimizer.setRegParam(l)
    this
  }
  
  override def evaluate(config: Map[String, String]) = {
    val sc = g.context
    val (file, delim, head, _) = GaussianLinearModel.readConfig(config)
    val csv = sc.textFile(file).map(line => line split delim)
      .map(_.map(_.toDouble)).map(vector => LabeledPoint(vector(vector.length-1), 
      Vectors.dense(vector.slice(0, vector.length-1))))

    val test_data = head match {
      case true =>
        csv.mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }
      case false =>
        csv
    }

    val mapPoint = (p: LabeledPoint) => new LabeledPoint(p.label,
        Vectors.dense(featureMap(DenseVector(p.features.toArray)).toArray))

    val predictLabel = LSSVMSparkModel.predict(params)(_task) _

    val results = test_data.map(point =>
      predictLabel(mapPoint(point)))
      .collect()
    Metrics(_task)(results.toList, results.length)
  }

}

object LSSVMSparkModel {

  def apply(implicit config: Map[String, String], sc: SparkContext): LSSVMSparkModel = {
    val (file, delim, head, task) = GaussianLinearModel.readConfig(config)
    val csv = sc.textFile(file).map(line => line split delim)
      .map(_.map(_.toDouble)).map(vector => {
      val label = vector(vector.length-1)
      vector(vector.length-1) = 1.0
      LabeledPoint(label, Vectors.dense(vector.slice(0, vector.length - 1)))
    }).cache()

    val data = head match {
      case true =>
        csv.mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }
      case false =>
        csv
    }

    new LSSVMSparkModel(data, task)
  }

  /**
   * Returns an indexed [[RDD]] from a non indexed [[RDD]] of [[T]]
   *
   * @param data : An [[RDD]] of [[T]]
   *
   * @return An (Int, T) Key-Value RDD indexed
   *         from 0 to data.count() - 1
   * */
  def indexedRDD[T](data: RDD[T]): RDD[(Long, T)] =
    data.zipWithIndex().map((p) => (p._2, p._1))

  def predict(params: DenseVector[Double])
             (_task: String)
             (point: LabeledPoint): (Double, Double) = {
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

  /**
   * Factory method to create the appropriate
   * optimization object required for the Gaussian
   * model
   * */
  def getOptimizer(task: String): GradientDescentSpark = task match {
    case "classification" => new GradientDescentSpark(
      new LeastSquaresSVMGradient(),
      new SquaredL2Updater())

    case "regression" => new GradientDescentSpark(
      new LeastSquaresGradient(),
      new SquaredL2Updater())
  }

  def kernelize(model: LSSVMSparkModel)(kernel: Kernel = null): KernelSparkModel = {
    //First, calculate a maximum entropy subset
    new KernelSparkModel(model.g.map{_._2}, model._task)
  }
}
