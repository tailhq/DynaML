package org.kuleuven.esat.optimization

import breeze.linalg.DenseVector
import org.apache.log4j.{Logger, Priority}
import org.apache.spark.AccumulatorParam
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
 * Implementation of SGD on
 * spark RDD
 */
class GradientDescentSpark (private val gradient: Gradient,
                            private val updater: Updater)
extends Optimizer[Int, DenseVector[Double],
  DenseVector[Double], Double, RDD[LabeledPoint]]{

  private var regParam: Double = 0.001

  /**
   * Set the regularization parameter. Default 0.0.
   */
  def setRegParam(regParam: Double): this.type = {
    this.regParam = regParam
    this
  }

  /**
   * Solve the convex optimization problem.
   */
  override def optimize(nPoints: Long,
                        initialP: DenseVector[Double],
                        ParamOutEdges: RDD[LabeledPoint])
  : DenseVector[Double] =
    GradientDescentSpark.runBatchSGD(
    nPoints,
    this.regParam,
    this.numIterations,
    this.updater,
    this.gradient,
    this.stepSize,
    initialP,
    ParamOutEdges,
    this.miniBatchFraction
  )
}

object GradientDescentSpark {

  private val logger = Logger.getLogger(this.getClass)

  def runBatchSGD(
                   nPoints: Long,
                   regParam: Double,
                   numIterations: Int,
                   updater: Updater,
                   gradient: Gradient,
                   stepSize: Double,
                   initial: DenseVector[Double],
                   POutEdges: RDD[LabeledPoint],
                   miniBatchFraction: Double): DenseVector[Double] = {
    var count = 1
    var oldW: DenseVector[Double] = initial
    var newW = oldW
    val sc = POutEdges.context
    val gradb = sc.broadcast(gradient)

    logger.log(Priority.INFO, "Training model using SGD")
    while(count <= numIterations) {
      val cumGradient =
        sc.accumulator(DenseVector.zeros[Double](initial.length))(new VectorAccumulator())
      val wb = sc.broadcast(oldW)
      POutEdges sample(withReplacement = false, fraction = miniBatchFraction) foreach
        ((ed) => {
          val features = DenseVector(ed.features.toArray)
          val label = ed.label
          val (g, _) = gradb.value.compute(features, label, wb.value)
          cumGradient += g
        })
      newW = updater.compute(oldW, cumGradient.value / nPoints.toDouble,
        stepSize, count, regParam)._1
      oldW = newW
      count += 1
    }
    newW
  }
}

class VectorAccumulator extends AccumulatorParam[DenseVector[Double]] {
  override def addInPlace(r1: DenseVector[Double],
                          r2: DenseVector[Double]): DenseVector[Double] = r1 + r2

  override def zero(initialValue: DenseVector[Double]): DenseVector[Double] =
    DenseVector.zeros(initialValue.length)
}