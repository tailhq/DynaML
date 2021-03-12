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
package io.github.tailhq.dynaml

import io.github.tailhq.dynaml.tensorflow.api.Api
import io.github.tailhq.dynaml.tensorflow.data.DataApi
import io.github.tailhq.dynaml.tensorflow.utils.Utils

/**
  * <h3>DynaML Tensorflow Package</h3>
  *
  * A collection of functions, transformations and
  * miscellaneous objects to help working with tensorflow
  * primitives and models.
  *
  * @author tailhq date: 23/11/2017
  * */
package object tensorflow {

  /**
    * <h4>DynaML Tensorflow Pointer</h4>
    * The [[dtf]] object is the entry point
    * for tensor related operations,
    * for more details see [[Api]].
    * */
  val dtf: Api.type         = Api

  /**
    * <h4>DynaML Neural Net Building Blocks</h4>
    *
    * The [[dtflearn]] object contains components
    * that can be used to create custom neural architectures,
    * from basic building blocks, for more details see [[Learn]].
    * */
  val dtflearn: Learn.type  = Learn

  /**
    * <h4>DynaML Tensorflow Pipes</h4>
    *
    * The [[dtfpipe]] object contains work flows/pipelines to simplify working
    * with tensorflow data sets and models, for more details see [[Pipe]].
    * */
  val dtfpipe: Pipe.type    = Pipe


  /**
    * <h4>DynaML Data Set API</h4>
    *
    * The [[dtfdata]] object contains functions for creating
    * and working with Data Sets. See [[DataApi]] for more
    * docs.
    * */
  val dtfdata: DataApi.type = DataApi

  /**
    * <h3>DynaML Tensorflow Utilities Package</h3>
    *
    * Contains miscellaneous utility functions for DynaML's
    * tensorflow handle/package. See [[Utils]] for more details.
    *
    * */
  val dtfutils: Utils.type = Utils

  object pde extends tensorflow.dynamics.DynamicsAPI

}
