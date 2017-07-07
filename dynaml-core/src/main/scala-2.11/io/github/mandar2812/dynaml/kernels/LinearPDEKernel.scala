package io.github.mandar2812.dynaml.kernels

/**
  * Kernels for linear PDE operator equations.
  * @author mandar2812 date 07/07/2017.
  * */
abstract class LinearPDEKernel[I] extends LocalScalarKernel[(I, Double)] {

  type exIndexSet = (I, Double)

  val baseKernel: LocalScalarKernel[(I, Double)]

  def invOperatorKernel: (exIndexSet, exIndexSet) => Double

  val baseID = "base::"+baseKernel.toString.split("\\.").last

  val operator_hyper_parameters: List[String]

  protected var operator_state: Map[String, Double]

  override val hyper_parameters =
    baseKernel.hyper_parameters.map(h => baseID+"/"+h) ++
      operator_hyper_parameters

  override def setHyperParameters(h: Map[String, Double]) = {
    assert(
      effective_hyper_parameters.forall(h.contains),
      "All hyper parameters must be contained in the arguments")

    //Set the base kernel hyper-parameters

    val base_kernel_state = h.filterKeys(_.contains(baseID)).map(c => (c._1.replace(baseID, ""), c._2))

    baseKernel.setHyperParameters(base_kernel_state)

    effective_hyper_parameters.foreach((key) => {
      state += (key -> h(key))
    })

    this
  }

  override def block(h: String*) = {
    super.block(h:_*)
    //Block parameters of base kernel
    baseKernel.block(h.filter(_.contains(baseID)).map(c => c.replace(baseID, "")):_*)
  }
}
