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
package io.github.tailhq.dynaml.examples

import com.cibo.scalastan._ 
import spire.implicits._
import io.github.tailhq.dynaml.probability._
import io.github.tailhq.dynaml.pipes._

object StanLinearModel {

    def apply(): StanResults = {
        val x = GaussianRV(0d, 1d) 
        val y = DataPipe((x: Double) => (GaussianRV(0d, 0.5d) + 2.5*x) - 1.5*x*x) 

        val xs = x.iid(500).draw.toSeq 
        val ys: Seq[Double] = xs.map(x => y(x).draw) 

        object MyModel extends StanModel {
            val n = data(int(lower = 0))
            val x = data(vector(n))
            val y = data(vector(n))
        
            val b = parameter(real())
            val m = parameter(real())
            val sigma = parameter(real(lower = 0.0))
        
            sigma ~ stan.cauchy(0, 1)
            y ~ stan.normal(m * x + b*x*:*x, sigma)
          }

          MyModel
           .withData(MyModel.x, xs)
           .withData(MyModel.y, ys)
           .run(chains = 5)

    }
}