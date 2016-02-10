package io.github.mandar2812.dynaml.models

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.optimization.{CommitteeModelSolver, RegularizedOptimizer, BackPropogation}
import io.github.mandar2812.dynaml.pipes.{ModelPipe, DataPipe}

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
  * Created by mandar on 9/2/16.
  */
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
