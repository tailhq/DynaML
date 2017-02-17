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
package io.github.mandar2812.dynaml.pipes

/**
  * @author mandar date: 17/02/2017.
  *
  * Data Pipes representing functions of 4 arguments
  */
trait DataPipe4[-Source1, -Source2, -Source3, -Source4, +Result] extends Serializable {
  self =>

  def run(data1: Source1, data2: Source2, data3: Source3, data4: Source4): Result

  def apply(data1: Source1, data2: Source2, data3: Source3, data4: Source4): Result = run(data1, data2, data3, data4)

  def >[Result2](otherPipe: DataPipe[Result, Result2]): DataPipe4[Source1, Source2, Source3, Source4, Result2] =
    DataPipe4((d1: Source1, d2:Source2, d3: Source3, d4: Source4) => otherPipe.run(self.run(d1, d2, d3, d4)))

}

object DataPipe4 {
  def apply[Source1, Source2, Source3, Source4, Result](func4: (Source1, Source2, Source3, Source4) => Result)
  : DataPipe4[Source1, Source2, Source3, Source4, Result] =
    new DataPipe4[Source1, Source2, Source3, Source4, Result] {
      override def run(data1: Source1, data2: Source2, data3: Source3, data4: Source4) =
        func4(data1, data2, data3, data4)
    }
}