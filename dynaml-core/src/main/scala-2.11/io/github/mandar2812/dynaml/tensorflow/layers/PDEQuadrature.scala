package io.github.mandar2812.dynaml.tensorflow.layers

import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Layer
import org.platanios.tensorflow.api._
import _root_.io.github.mandar2812.dynaml.tensorflow.dynamics._

/**
  * <h3>Quadrature for PDE Systems</h3>
  *
  * Computes the weighted error of an function approximation
  * with respect to some governing dynamics i.e. Partial Differential Operator.
  *
  * @param name String identifier for this loss function
  * @param f A sequence of residual functions, obtained as a result of
  *          applying some PDE operators (see [[DifferentialOperator]]).
  * @param quadrature_nodes A set of nodal points in the domain of the system on which
  *                         the error should be computed.
  * @param weights Weights associated with each quadrature node.
  * @param loss_weightage Weights to multiply each of the loss terms computed for every
  *                       element of [[f]].
  *
  * */
case class PDEQuadrature[D <: DataType](
  override val name: String,
  f: Seq[Layer[Output, Output]],
  quadrature_nodes: Tensor[D],
  weights: Tensor[D],
  loss_weightage: Tensor[D]) extends
  Layer[Output, Output](name) {

  require(quadrature_nodes.shape(0) == weights.shape(0) && weights.rank == 1)
  require(loss_weightage.rank == 0 || (loss_weightage.rank == 1 && loss_weightage.shape(0) == f.length))


  override val layerType: String = s"QuadratureLoss[${f.map(_.layerType)}]"

  override def forwardWithoutContext(input: Output)(implicit mode: Mode): Output = {

    val (q_nodes, q_weights, importance) = (
      tf.constant(quadrature_nodes, quadrature_nodes.dataType, quadrature_nodes.shape, "quadrature_nodes"),
      tf.constant(weights, weights.dataType, weights.shape, "quadrature_nodal_weights"),
      tf.constant(loss_weightage, loss_weightage.dataType, loss_weightage.shape, "colocation_error_weight")
    )

    val quadrature_loss =
      tf.stack(
        f.map(q => {
          val output = q.forward(q_nodes)

          val rank_output = output.rank
          val reduce_axes =
            if(rank_output > 2) Tensor(1 until rank_output)
            else if(rank_output == 2) Tensor(1)
            else null

          if(reduce_axes == null) {
            output.square.multiply(q_weights).sum()
          } else {
            output.square.sum(reduce_axes).multiply(q_weights).sum()
          }
        }), axis = -1)
        .multiply(importance)
        .sum()

    input.add(quadrature_loss)
  }
}
