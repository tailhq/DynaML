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
package io.github.mandar2812.dynaml.tensorflow

import io.github.mandar2812.dynaml.pipes._
import org.platanios.tensorflow.api._


package object implicits {

  implicit val tensorTup2Split: MetaPipe12[(Tensor, Tensor), Int, Int, (Tensor, Tensor)] = MetaPipe12(
    (workingData: (Tensor, Tensor)) => (index_start: Int, index_end: Int) => (
      workingData._1(index_start::index_end + 1, ---),
      workingData._2(index_start::index_end + 1, ---)
    )
  )

  implicit val concatTensorTup2Splits: DataPipe[Iterable[(Tensor, Tensor)], (Tensor, Tensor)] = DataPipe(
    (ds: Iterable[(Tensor, Tensor)]) => {

      val separated_splits = ds.unzip

      (
        tfi.concatenate(separated_splits._1.toSeq, axis = 0),
        tfi.concatenate(separated_splits._2.toSeq, axis = 0)
      )

    })

}
