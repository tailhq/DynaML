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
package io.github.mandar2812.dynaml

/**
  * <h3>DynaML Tensorflow Package</h3>
  *
  * A collection of functions, transformations and
  * miscellaneous objects to help working with tensorflow
  * primitives and models.
  *
  * @author mandar2812 date: 23/11/2017
  * */
package object tensorflow {

  /**
    * <h4>DynaML Tensorflow Pointer</h4>
    * The [[dtf]] object is the entry point
    * for tensor related operations.
    * */
  val dtf      = Api

  /**
    * <h4>DynaML Neural Net Building Blocks</h4>
    *
    * The [[dtflearn]] object contains components
    * that can be used to create custom neural architectures,
    * from basic building blocks.
    * */
  val dtflearn = Learn

  /**
    * <h4>DynaML Tensorflow Pipes</h4>
    *
    * The [[dtfpipe]] contains work flows/pipelines to simplify working
    * with tensorflow data sets and models.
    * */
  val dtfpipe  = Pipe


  val data     = Data

  /**
    * <h3>DynaML Tensorflow Utilities Package</h3>
    *
    * Contains miscellaneous utility functions for DynaML's
    * tensorflow handle/package.
    *
    * */
  val dtfutils = Utils

}
