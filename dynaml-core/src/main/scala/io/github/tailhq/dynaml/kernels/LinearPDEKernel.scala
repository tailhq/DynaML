/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
* */
package io.github.tailhq.dynaml.kernels

/**
  * Kernels for linear PDE operator equations.
  * @author tailhq date 07/07/2017.
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
