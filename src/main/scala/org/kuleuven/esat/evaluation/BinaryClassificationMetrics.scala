package org.kuleuven.esat.evaluation

import scalax.chart.module.ChartFactories.XYLineChart

class BinaryClassificationMetrics(val scoresAndLabels: List[(Double, Double)]){

  private val thresholds = (-100 to 100).map((x) => x.toDouble/100)

  def areaUnderCurve(points: List[(Double, Double)]): Double = {
    var sum = 0.0
    sum
  }

  def areaUnderPR(): Double = 0.0

  def areaUnderROC(): Double = 0.0


  def fMeasureByThreshold(): List[(Double, Double)] = fMeasureByThreshold(1.0)

  def fMeasureByThreshold(beta: Double): List[(Double, Double)] = tpfpByThreshold().map((couple) => {
    val tp = couple._2._1
    val fp = couple._2._2
    val betasq = math.pow(beta, 2.0)
    (couple._1, (1 + betasq)*tp/((1 + betasq)*tp + betasq*(1-tp) + fp))
  })

  def pr(): List[(Double, Double)] = recallByThreshold().zip(precisionByThreshold()).map((couple) =>
    (couple._1._2, couple._2._2)).sorted

  def recallByThreshold(): List[(Double, Double)]  = tpfpByThreshold().map((point) => (point._1, point._2._1))

  def precisionByThreshold(): List[(Double, Double)]  = tpfpByThreshold().map((point) =>
    (point._1, point._2._1/(point._2._1 + point._2._2)))

  def roc(): List[(Double, Double)] =
    tpfpByThreshold().map((point) => (point._2._2, point._2._1)).sorted


  def tpfpByThreshold(): List[(Double, (Double, Double))]  =
    this.thresholds.toList.map((th) => {
      var tp: Double = 0.0
      var fp: Double = 0.0
      var count: Double = 0.0
      this.scoresAndLabels.foreach((couple) => {
        count += 1.0
        if(math.signum(couple._1 - th) == couple._2) {
          tp += 1.0
        } else {
          fp += 1.0
        }
      })
      (th, (tp/count, fp/count))
    })

  def generatePlots(dir: String): Unit = {
    val roccurve = this.roc()
    val prcurve = this.pr()
    val fm = this.fMeasureByThreshold()
    val fm1 = this.fMeasureByThreshold(0.5)
    val fm2 = this.fMeasureByThreshold(1.5)


    val chart1 = XYLineChart(roccurve,
      title = "Receiver Operating Characteristic", legend = true)
    chart1.show()

    val chart2 = XYLineChart(prcurve,
      title = "Precision Recall Curve", legend = true)
    chart2.show()

    val chart3 = XYLineChart(fm,
      title = "F1 measure by threshold beta = 1", legend = true)
    chart3.show()

    val chart4 = XYLineChart(fm1,
      title = "F1 measure by threshold beta = 0.5", legend = true)
    chart4.show()

    val chart5 = XYLineChart(fm2,
      title = "F1 measure by threshold 0.5 beta = 1.5", legend = true)
    chart5.show()
  }
}
