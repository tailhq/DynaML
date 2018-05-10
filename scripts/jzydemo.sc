import org.jzy3d.analysis.AbstractAnalysis
import org.jzy3d.analysis.AnalysisLauncher
import org.jzy3d.chart.factories.AWTChartComponentFactory
import org.jzy3d.colors.Color
import org.jzy3d.colors.ColorMapper
import org.jzy3d.colors.colormaps.ColorMapRainbow
import org.jzy3d.plot3d.builder.{Builder, Mapper}
import org.jzy3d.plot3d.builder.concrete.OrthonormalGrid
import org.jzy3d.plot3d.primitives.Shape
import org.jzy3d.plot3d.rendering.canvas.Quality
import org.jzy3d.maths
import org.jzy3d.plot3d.builder.concrete.WaterfallTessellator

import _root_.io.github.mandar2812.dynaml.repl.Router.main

@main
def main(func: (Double, Double) => Double, chart_type: String, wireFrame: Boolean = true) = chart_type match {
  case "surface"      => AnalysisLauncher.open(new SurfaceDemo(func, wireFrame))
  case "waterfall"    => AnalysisLauncher.open(new WaterfallDemo(func))
  case "tessellation" => AnalysisLauncher.open(new DelauneyDemo(func))
  case _              => AnalysisLauncher.open(new SurfaceDemo(func, wireFrame))
}


class SurfaceDemo(function: (Double, Double) => Double, displayWireFrame: Boolean = true) extends AbstractAnalysis {
  override def init(): Unit = { // Define a function to plot
    val mapper = new Mapper() {
      def f(x: Double, y: Double) = function(x, y)
    }
    // Define range and precision for the function to plot
    val range = new maths.Range(-3, 3)
    val steps = 80
    // Create the object to represent the function over the given range.
    val surface: Shape = Builder.buildOrthonormal(
      new OrthonormalGrid(range, steps, range, steps),
      mapper
    )

    surface.setColorMapper(
      new ColorMapper(
        new ColorMapRainbow,
        surface.getBounds.getZmin,
        surface.getBounds.getZmax,
        new Color(1, 1, 1, .5f))
    )

    surface.setFaceDisplayed(true)
    surface.setWireframeDisplayed(displayWireFrame)
    // Create a chart
    chart = AWTChartComponentFactory.chart(Quality.Advanced, getCanvasType)
    chart.getScene.getGraph.add(surface)
  }
}

class DelauneyDemo(function: (Double, Double) => Double) extends AbstractAnalysis {
  override def init(): Unit = { // Define a function to plot
    val mapper = new Mapper() {
      def f(x: Double, y: Double) = function(x, y)
    }
    // Define range and precision for the function to plot
    val range = new maths.Range(-3, 3)
    val steps = 80

    val grid = new OrthonormalGrid(range, steps, range, steps)

    // Create the object to represent the function over the given range.
    val surface: Shape = Builder.buildDelaunay(grid.apply(mapper))

    surface.setColorMapper(
      new ColorMapper(
        new ColorMapRainbow,
        surface.getBounds.getZmin,
        surface.getBounds.getZmax,
        new Color(1, 1, 1, .5f))
    )

    surface.setFaceDisplayed(true)

    // Create a chart
    chart = AWTChartComponentFactory.chart(Quality.Advanced, getCanvasType)
    chart.getScene.getGraph.add(surface)
  }
}


class WaterfallDemo(function: (Double, Double) => Double) extends AbstractAnalysis {
  override def init(): Unit = {
    val x = new Array[Float](80)

    var i: Int = 0

    while (i < x.length) {
      x(i) = -3f + 6f * (i.toFloat / (x.length - 1))

      {
        i += 1; i - 1
      }
    }

    val y = new Array[Float](40)

    i = 0

    while (i < y.length) {
      y(i) = -3f + 2f * (i.toFloat / (y.length - 1))

      {
        i += 1; i - 1
      }
    }

    val z = getZ(x, y)

    val waterfall = new WaterfallTessellator

    val build = waterfall.build(x, y, z)

    build.setColorMapper(
      new ColorMapper(
        new ColorMapRainbow,
        build.getBounds.getZmin,
        build.getBounds.getZmax,
        new Color(1, 1, 1, .5f)))
    // Create a chart
    chart = AWTChartComponentFactory.chart(Quality.Intermediate, getCanvasType)
    chart.getScene.getGraph.add(build)
    chart.getView
  }

  private def getZ(x: Array[Float], y: Array[Float]) = {
    val z = new Array[Float](x.length * y.length)
    var i = 0
    while (i < y.length) {
      var j = 0
      while (j < x.length) {
        z(j + (x.length * i)) = f(x(j).toDouble, y(i).toDouble).toFloat

        {
          j += 1; j - 1
        }
      }

      {
        i += 1; i - 1
      }
    }
    z
  }

  private def f(x: Double, y: Double) = function(x, y)
}

