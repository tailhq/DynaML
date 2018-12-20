package io.github.mandar2812.dynaml.models.svm

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.sqrt
import com.github.tototoshi.csv.CSVReader
import com.tinkerpop.frames.FramedGraphFactory
import io.github.mandar2812.dynaml.utils.MinMaxAccumulator
import org.apache.spark.{Accumulator, SparkContext}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import io.github.mandar2812.dynaml.evaluation.{Metrics, MetricsSpark}
import io.github.mandar2812.dynaml.optimization._
import org.apache.log4j.Logger
import org.apache.spark.mllib.linalg.Vector

import scala.util.Random

/**
 * Implementation of the Least Squares SVM
 * using Apache Spark RDDs
 */
class LSSVMSparkModel(data: RDD[LabeledPoint], task: String)
  extends KernelSparkModel(data, task) with Serializable {

  override protected val optimizer = LSSVMSparkModel.getOptimizer(task)

  override protected var params: DenseVector[Double] = DenseVector.ones(featuredims+1)

  private var featureMatricesCache: (DenseMatrix[Double], DenseVector[Double]) = (null, null)

  def dimensions = featuredims

  /**
   * Predict the value of the
   * target variable given a
   * point.
   *
   **/
  override def predict(point: DenseVector[Double]): Double =
    LSSVMSparkModel.predictBDV(params)(task)(point)

  /**
   * Learn the parameters
   * of the model which
   * are in a node of the
   * graph.
   *
   **/
  override def learn(): Unit = {
    val featureMapb = g.context.broadcast(featureMap)
    val meanb = g.context.broadcast(DenseVector(colStats.mean.toArray))
    val varianceb = g.context.broadcast(DenseVector(colStats.variance.toArray))
    val trainingData = this.g.map(point => {
      val vec = DenseVector(point._2.features.toArray)
      val ans = vec - meanb.value
      ans :/= sqrt(varianceb.value)
      new LabeledPoint(
        point._2.label,
        Vectors.dense(DenseVector.vertcat(
          featureMapb.value(ans),
          DenseVector(1.0))
          .toArray)
      )
    })
    params = this.optimizer.optimize(nPoints, trainingData, params)

  }

  override def clearParameters: Unit = {
    params = DenseVector.ones[Double](featuredims+1)
  }

  override def setRegParam(l: Double): this.type = {
    this.optimizer.setRegParam(l)
    this
  }

  override def getRegParam = this.optimizer.getRegParam
  
  override def evaluate(config: Map[String, String]) = {
    val sc = g.context
    val (file, delim, head, _) = LSSVMModel.readConfig(config)
    val csv = sc.textFile(file).map(line => line split delim)
      .map(_.map(_.toDouble)).map(vector => LabeledPoint(vector(vector.length-1), 
      Vectors.dense(vector.slice(0, vector.length-1))))

    val test_data = head match {
      case true =>
        csv.mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }
      case false =>
        csv
    }

    val paramsb = sc.broadcast(params)
    val minmaxacc = sc.accumulator(DenseVector(Double.MaxValue, Double.MinValue),
      "Min Max Score acc")(MinMaxAccumulator)

    val featureMapbroadcast = sc.broadcast(featureMap)

    val meanb = g.context.broadcast(DenseVector(colStats.mean.toArray))
    val varianceb = g.context.broadcast(DenseVector(colStats.variance.toArray))


    val results = test_data.map(point => {
      val vec = DenseVector(point.features.toArray)
      val ans = vec - meanb.value
      ans :/= sqrt(varianceb.value)

      val sco: Double = paramsb.value dot
        DenseVector.vertcat(featureMapbroadcast.value(ans), DenseVector(1.0))

      minmaxacc += DenseVector(sco, sco)

      (sco, point.label)
    })
    val minmax = minmaxacc.value
    MetricsSpark(task)(results, results.count(), (minmax(0), minmax(1)))
  }

  def GetStatistics(): Unit = {
    println("Feature Statistics: \n")
    println("Mean: "+this.colStats.mean)
    println("Variance: \n"+this.colStats.variance)
  }

  def unpersist: Unit = {
    this.processed_g.unpersist()
    this.g.context.stop()
  }

  override def evaluateFold(params: DenseVector[Double])
                           (test_data_set: RDD[LabeledPoint])
                           (task: String): Metrics[Double] = {
    val sc = test_data_set.context
    var index: Accumulator[Long] = sc.accumulator(1)

    val paramsb = sc.broadcast(params)
    val minmaxacc = sc.accumulator(DenseVector(Double.MaxValue, Double.MinValue),
      "Min Max Score acc")(MinMaxAccumulator)
    val scoresAndLabels = test_data_set.map((e) => {
      index += 1
      val sco: Double = paramsb.value dot DenseVector(e.features.toArray)
      minmaxacc += DenseVector(sco, sco)
      (sco, e.label)
    })
    val minmax = minmaxacc.value
    MetricsSpark(task)(scoresAndLabels, index.value, (minmax(0), minmax(1)))
  }

  override def crossvalidate(folds: Int, reg: Double,
                             optionalStateFlag: Boolean = false): (Double, Double, Double) = {
    //Create the folds as lists of integers
    //which index the data points
    /*this.optimizer.setRegParam(reg).setNumIterations(2)
      .setStepSize(0.001).setMiniBatchFraction(1.0)*/
    val shuffle = Random.shuffle((1L to this.npoints).toList)

    if(!optionalStateFlag || featureMatricesCache == (null, null)) {
      featureMatricesCache = LSSVMSparkModel.getFeatureMatrix(npoints, processed_g.map(_._2),
        this.initParams(), 1.0, reg)

      val smoother:DenseMatrix[Double] = DenseMatrix.eye[Double](effectivedims)/reg
      smoother(-1,-1) = 0.0
      featureMatricesCache._1 :+= smoother
    } else {
      val smoother_old:DenseMatrix[Double] = DenseMatrix.eye[Double](effectivedims)/current_state("RegParam")
      smoother_old(-1,-1) = 0.0
      val smoother_new:DenseMatrix[Double] = DenseMatrix.eye[Double](effectivedims)/reg
      smoother_new(-1,-1) = 0.0

      featureMatricesCache._1 :-= smoother_old
      featureMatricesCache._1 :+= smoother_new
    }

    val avg_metrics: DenseVector[Double] = (1 to folds).map{a =>
      //For the ath fold
      //partition the data
      //ceil(a-1*npoints/folds) -- ceil(a*npoints/folds)
      //as test and the rest as training
      val test = shuffle.slice((a-1)*this.nPoints.toInt/folds, a*this.nPoints.toInt/folds)
      val test_data = processed_g.filter((keyValue) =>
        test.contains(keyValue._1)).map(_._2).cache()

      val (a_folda, b_folda) = LSSVMSparkModel.getFeatureMatrix(npoints,
        test_data,
        this.initParams(),
        1.0, reg)

      val featureMatrix_a = featureMatricesCache._1 - a_folda
      val bias = featureMatricesCache._2 - b_folda
      val tempparams = ConjugateGradientSpark.runCG(featureMatrix_a,
        bias, this.initParams(), 0.001, 35)
      val metrics = this.evaluateFold(tempparams)(test_data)(this.task)
      val res: DenseVector[Double] = metrics.kpi() / folds.toDouble
      res
    }.reduce(_+_)

    this.processed_g.unpersist(blocking = true)
    (avg_metrics(0),
      avg_metrics(1),
      avg_metrics(2))
  }

}

object LSSVMSparkModel {

  def scaleAttributes(mean: Vector,
                      variance: Vector)(x: Vector)
  : Vector = {
    val ans = DenseVector(x.toArray) - DenseVector(mean.toArray)
    ans :/= sqrt(DenseVector(variance.toArray))
    Vectors.dense(ans.toArray)
  }

  def apply(implicit config: Map[String, String], sc: SparkContext): LSSVMSparkModel = {
    val (file, delim, head, task) = LSSVMModel.readConfig(config)
    val minPartitions = if(config.contains("parallelism") &&
      config.contains("executors") && config.contains("factor"))
      config("factor").toInt * config("parallelism").toInt * config("executors").toInt
    else 2

    val csv = sc.textFile(file, minPartitions).map(line => line split delim)
      .map(_.map(_.toDouble)).map(vector => {
      val label = vector(vector.length-1)
      vector(vector.length-1) = 1.0
      LabeledPoint(label, Vectors.dense(vector.slice(0, vector.length - 1)))
    })

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

  def scoreSparkVector(params: DenseVector[Double])
                      (point: Vector): Double = {
    params dot DenseVector.vertcat(
      DenseVector(point.toArray),
      DenseVector(1.0))
  }

  def predictBDV(params: DenseVector[Double])
                (_task: String)
                (point: DenseVector[Double]): Double = {
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
  predictBDV(params)(_task)(DenseVector(point.toArray))

  /**
   * Factory method to create the appropriate
   * optimization object required for the LSSVM
   * model
   * */
  def getOptimizer(task: String): ConjugateGradientSpark = new ConjugateGradientSpark

  def getFeatureMatrix(nPoints: Long,
                       ParamOutEdges: RDD[LabeledPoint],
                       initialP: DenseVector[Double],
                       frac: Double, regParam: Double) = {
    val dims = initialP.length
    //Cast as problem of form A.w = b
    //A = Phi^T . Phi + I_dims*regParam
    //b = Phi^T . Y
    val (a,b): (DenseMatrix[Double], DenseVector[Double]) =
      ParamOutEdges.filter((_) => Random.nextDouble() <= frac)
        .mapPartitions((edges) => {
        Seq(edges.map((edge) => {
          val phi = DenseVector(edge.features.toArray)
          val label = edge.label
          val phiY: DenseVector[Double] = phi * label
          (phi*phi.t, phiY)
        }).reduce((couple1, couple2) => {
          (couple1._1+couple2._1, couple1._2+couple2._2)
        })).toIterator
      }).reduce((couple1, couple2) => {
        (couple1._1+couple2._1, couple1._2+couple2._2)
      })

    (a,b)
  }



}

object LSSVMModel {
  val manager: FramedGraphFactory = new FramedGraphFactory
  val logger = Logger.getLogger(this.getClass)

  /**
    * Factory function to rescale attributes
    * given a vector of means and the Cholesky
    * factorization of the inverse variance matrix
    *
    * */
  def scaleAttributes(mean: DenseVector[Double],
                      sigmaInverse: DenseMatrix[Double])(x: DenseVector[Double])
  : DenseVector[Double] = sigmaInverse * (x - mean)

  /**
    * Factory method to create the appropriate
    * optimization object required for the Gaussian
    * model
    * */
  def getOptimizer(task: String): ConjugateGradient = new ConjugateGradient

  def readCSV(reader: CSVReader, head: Boolean):
  (Iterable[(DenseVector[Double], Double)], Int) = {
    val stream = reader.toStream.toIterable
    val dim = stream.head.length

    def lines = if(head) {
      stream.drop(1)
    } else {
      stream
    }

    (lines.map{parseLine}, dim)
  }

  def parseLine = {line : List[String] =>
    //Parse line and extract features
    val yv = line.apply(line.length - 1).toDouble
    val xv: DenseVector[Double] =
      DenseVector(line.slice(0, line.length - 1).map{x => x.toDouble}.toArray)

    (xv, yv)
  }

  def readConfig(config: Map[String, String]): (String, Char, Boolean, String) = {

    assert(config.isDefinedAt("file"), "File name must be Defined!")
    val file: String = config("file")

    val delim: Char = if(config.isDefinedAt("delim")) {
      config("delim").toCharArray()(0)
    } else {
      ','
    }

    val head: Boolean = if(config.isDefinedAt("head")) {
      config("head") match {
        case "true" => true
        case "True" => true
        case "false" => false
        case "False" => false
      }
    } else {
      true
    }

    val task: String = if(config.isDefinedAt("task")) config("task") else ""

    (file, delim, head, task)
  }

}
