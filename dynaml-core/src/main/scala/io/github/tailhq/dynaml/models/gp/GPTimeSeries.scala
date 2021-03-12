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
package io.github.tailhq.dynaml.models.gp

import com.github.tototoshi.csv.CSVReader
import io.github.tailhq.dynaml.kernels.LocalScalarKernel

/**
  * @author tailhq datum 16/11/15.
  *
  * Gaussian Process Time Series Model
  *
  * y(t) = f(t) + e
  * f(t) ~ GP(0, cov(t, t'))
  * e|f(t) ~ N(0, noise(t, t'))
  */
class GPTimeSeries(cov: LocalScalarKernel[Double],
                   n: LocalScalarKernel[Double],
                   trainingdata: Seq[(Double, Double)])
  extends AbstractGPRegressionModel[Seq[(Double, Double)],Double](cov, n, trainingdata,
    trainingdata.length){

  /**
    * Convert from the underlying data structure to
    * Seq[(I, Y)] where I is the index set of the GP
    * and Y is the value/label type.
    **/
  override def dataAsSeq(data: Seq[(Double, Double)]) = data
}

object GPTimeSeries {
  def buildFromFile(file : CSVReader, columns: List[Int],
                    missingValues: List[Double], length: Int,
                    processFunc: Seq[String] => (Double, Double),
                    fraction: Double = 0.75):
  (Seq[(Double, Double)], Seq[(Double, Double)]) = {
    assert(columns.length == missingValues.length || missingValues.isEmpty,
      "Missing values must be of equal number to features")
    val stream = file.iterator
    val trLength = (length*fraction).toInt
    val trainingdata = stream.take(trLength).map(processFunc).toSeq
    val testData = stream.toSeq.map(processFunc)
    (trainingdata, testData)
  }


}