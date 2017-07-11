package io.github.mandar2812.dynaml.kernels

/**
  * Kernels for linear PDE operator equations.
  * @author mandar2812 date 07/07/2017.
  * */
trait LinearPDEKernel[I] extends LocalScalarKernel[(I, Double)] {

  val baseKernel: LocalScalarKernel[(I, Double)]

  type exIndexSet = (I, Double)

  def invOperatorKernel: (exIndexSet, exIndexSet) => Double

  val baseID: String

  def _operator_hyper_parameters: List[String]

  override def setHyperParameters(h: Map[String, Double]) = {
    assert(
      effective_hyper_parameters.forall(h.contains),
      "All hyper parameters must be contained in the arguments")

    //Set the base kernel hyper-parameters

    val base_kernel_state = h.filterKeys(_.contains(baseID)).map(c => (c._1.replace(baseID, "").tail, c._2))

    baseKernel.setHyperParameters(base_kernel_state)

    effective_hyper_parameters.foreach((key) => {
      state += (key -> h(key))
    })

    this
  }

  override def block(h: String*) = {
    super.block(h:_*)
    //Block parameters of base kernel
    baseKernel.block(h.filter(_.contains(baseID)).map(c => c.replace(baseID, "").tail):_*)
  }
}
