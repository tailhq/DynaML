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
  * @author mandar date: 16/02/2017.
  *
  * Data Pipes representing functions of 2 arguments
  */
trait DataPipe2[-Source1, -Source2, Result] extends
  DataPipeConvertible[(Source1, Source2), Result] with
  Serializable {

  self =>

  def run(data1: Source1, data2: Source2): Result

  def apply(data1: Source1, data2: Source2): Result = run(data1, data2)

  def >[Result2](otherPipe: DataPipe[Result, Result2]): DataPipe2[Source1, Source2, Result2] =
    DataPipe2((d1: Source1, d2:Source2) => otherPipe.run(self.run(d1, d2)))

  override def toPipe: ((Source1, Source2)) => Result = (x: (Source1, Source2)) => self.run(x._1, x._2)
}

object DataPipe2 {
  def apply[Source1, Source2, Result](func2: (Source1, Source2) => Result): DataPipe2[Source1, Source2, Result] =
    new DataPipe2[Source1, Source2, Result] {
      override def run(data1: Source1, data2: Source2) = func2(data1, data2)
    }
}








