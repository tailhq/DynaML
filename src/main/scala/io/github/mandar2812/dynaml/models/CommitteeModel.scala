package io.github.mandar2812.dynaml.models

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.pipes.ModelPipe
import io.github.mandar2812.dynaml.models.neuralnets.FeedForwardNetwork
import io.github.mandar2812.dynaml.models.gp.GPRegression

/**
  *
  * w1*y1(x) + w2*y2(x) + ... + wb*yb(x)
  * Defines the basic skeleton of a "meta-model" or
  * a model of models.
  *
  * A set of base models are trained on sub-sampled versions
  * of the training data set and finally a predictor of the form.
  *
  * y(x) = f(y1(x), y2(x), ..., yb(x))
  *
  * Where f is some combination function and
  * b is the number of base models used.
  *
  * @tparam D The type of the data structure containing the
  *           training data set.
  *
  * @tparam D1 The type of data structure containing the data
  *            of the base models.
  *
  * @tparam BaseModel The type of model used as base model
  *                   for the meta model.
  *                   example: [[FeedForwardNetwork]], [[GPRegression]], etc
  *
  * @tparam Pipe A sub-type of [[ModelPipe]] which yields a [[BaseModel]]
  *              with [[D1]] as the base data structure given a
  *              data structure of type [[D]]
  *
  * @param num The number of training data points.
  *
  * @param data The actual training data
  *
  * @param networks A sequence of [[Pipe]] objects yielding [[BaseModel]]
  * */
abstract class MetaModel[
D, D1,
BaseModel <: Model[D1, DenseVector[Double], Double],
Pipe <: ModelPipe[D, D1, DenseVector[Double], Double, BaseModel]
](num: Int, data: D, networks: Pipe*)
  extends Model[D, DenseVector[Double], Double] {

  override protected val g = data

  val baseNetworks: List[BaseModel] =
    networks.toList.map(net => net.run(g))

}



/**
  * Defines an abstract implementation of a "committee-model".
  *
  * A predictor of the form
  *
  * y(x) = w1*y1(x) + w2*y2(x) + ... + wb*yb(x)
  *
  * is learned, where `(y1(x), y2(x), ..., yb(x))` a set of base models
  * are trained on sub-sampled versions of the training data set and `b`
  * is the number of base models used.
  *
  * @tparam D The type of the data structure containing the
  *           training data set.
  *
  * @tparam D1 The type of data structure containing the data
  *            of the base models.
  *
  * @tparam BaseModel The type of model used as base model
  *                   for the meta model.
  *                   example: [[FeedForwardNetwork]], [[GPRegression]], etc
  *
  * @tparam Pipe A sub-type of [[ModelPipe]] which yields a [[BaseModel]]
  *              with [[D1]] as the base data structure given a
  *              data structure of type [[D]]
  *
  * @param num The number of training data points.
  *
  * @param data The actual training data
  *
  * @param networks A sequence of [[Pipe]] objects yielding [[BaseModel]]
  * */
abstract class CommitteeModel[
D, D1,
BaseModel <: Model[D1, DenseVector[Double], Double],
Pipe <: ModelPipe[D, D1, DenseVector[Double], Double, BaseModel]
](num: Int, data: D, networks: Pipe*) extends
MetaModel[D,D1,BaseModel,Pipe](num, data, networks:_*) with
LinearModel[D, Int, Int, DenseVector[Double], DenseVector[Double],
  Double, D] {

  val num_points = num


  /**
    * Predict the value of the
    * target variable given a
    * point.
    *
    **/
  override def predict(point: DenseVector[Double]): Double =
    params dot featureMap(point)

  override def clearParameters(): Unit =
    DenseVector.fill[Double](baseNetworks.length)(1.0)

  override def initParams(): DenseVector[Double] =
    DenseVector.fill[Double](baseNetworks.length)(1.0)

  /**
    * Learn the parameters
    * of the model which
    * are in a node of the
    * graph.
    *
    **/
  override def learn(): Unit = {

    params = optimizer.optimize(num_points, g, initParams())
  }

  override protected var params: DenseVector[Double] =
    DenseVector.fill[Double](baseNetworks.length)(1.0)

  featureMap = (pattern) =>
    DenseVector(baseNetworks.map(net =>
      net.predict(pattern)).toArray)


}
