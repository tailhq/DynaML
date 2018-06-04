!!! summary
    The [`dynaml.graphics`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/#io.github.mandar2812.dynaml.graphics.package) 
    package is a new addition to the API since the v1.5.3 release. It aims to provide more unified access point to producing
    visualizations.
    
    
## 3D Plots

Support for 3d visualisations is provided by the `dynaml.graphics.plot3d`, under the hood the `plot3d` object calls 
the Jzy3d library to produce interactive 3d plots.

Producing 3d charts involves similar procedure for each chart kind, use `plot3d.draw()` to generate a chart object,
and use `plot3d.show()` to display it in a GUI.


### Surface Plots

#### From defined functions

```scala
import io.github.mandar2812.dynaml.graphics._

val mexican_hat = 
  (x: Double, y: Double) => 
    (1.0/math.Pi)*(1.0 - 0.5*(x*x + y*y))*math.exp(-0.5*(x*x + y*y))
    

val mexican_hat_chart = plot3d.draw(mexican_hat)



```

![mexican](/images/plot3d.jpeg)

#### From a set of points 

### Histograms