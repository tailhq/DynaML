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
package io.github.mandar2812.dynaml.tensorflow.utils

import org.platanios.tensorflow.api._

/**
  * <h3>DynaML Tensorflow Utilities Package</h3>
  *
  * Contains miscellaneous utility functions for DynaML's
  * tensorflow handle/package.
  *
  * */
object dtfutils {

  /**
    * Convert a float tensor to a Sequence.
    * */
  def toDoubleSeq(t: Tensor): Iterator[Double] = {
    val datatype = t.dataType.toString()
    t.entriesIterator.map(x =>
      if(datatype == "FLOAT64") x.asInstanceOf[Double]
      else x.asInstanceOf[Float].toDouble)
  }


  def get_ffstack_properties(
    neuron_counts: Seq[Int],
    ff_index: Int): (Seq[Shape], Seq[String], Seq[String]) = {

    val layer_parameter_names = (ff_index until ff_index + neuron_counts.length - 1).map(i => "Linear_"+i+"/Weights")
    val layer_shapes          = neuron_counts.sliding(2).toSeq.map(c => Shape(c.head, c.last))
    val layer_datatypes       = Seq.fill(layer_shapes.length)("FLOAT64")


    (layer_shapes, layer_parameter_names, layer_datatypes)
  }


}

