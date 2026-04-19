#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# setup_scala_box_jenkins_demo_fixed.sh
#
# Data dictionary
# ------------------------------------------------------------
# PROJECT_DIR   : directory for the Scala demo project
# SRC_FILE      : Scala source file written by this script
# SCALA_VERSION : Scala version used by Scala CLI
# LIB_DEP       : Java ARIMA library used from Scala
# TEST_SIZE     : held-out forecast horizon
#
# What this script does
# ------------------------------------------------------------
# 1. Installs prerequisites
# 2. Installs Scala CLI if missing
# 3. Writes a corrected Scala Box-Jenkins-style ARIMA demo
# 4. Runs the demo
#
# Notes
# ------------------------------------------------------------
# - Uses synthetic data built into the Scala source
# - Searches a small ARIMA grid by AIC
# - Forecasts a test split
# - Reports MAE, RMSE, MAPE
# ============================================================

PROJECT_DIR="scala-box-jenkins-demo"
SRC_FILE="${PROJECT_DIR}/BoxJenkinsDemo.scala"

echo "[1/5] Installing system prerequisites..."
sudo apt-get update
sudo apt-get install -y curl ca-certificates bash

echo "[2/5] Installing Scala CLI if needed..."
if ! command -v scala-cli >/dev/null 2>&1; then
  curl -fsSL https://scala-cli.virtuslab.org/get | sh
  sudo mv scala-cli /usr/local/bin/
fi

echo "[3/5] Creating project directory..."
mkdir -p "${PROJECT_DIR}"

echo "[4/5] Writing corrected Scala source..."
cat > "${SRC_FILE}" <<'EOF'
//> using scala "3.3.1"
//> using dep "com.github.signaflo:timeseries:0.4"

import scala.util.Try
import com.github.signaflo.timeseries.TimeSeries
import com.github.signaflo.timeseries.model.arima.Arima
import com.github.signaflo.timeseries.model.arima.ArimaOrder

object BoxJenkinsDemo:

  final case class Candidate(
      p: Int,
      d: Int,
      q: Int,
      aic: Double,
      model: Arima
  )

  def main(args: Array[String]): Unit =
    val data = Array[Double](
      120.0, 123.0, 121.0, 126.0, 130.0, 129.0,
      133.0, 136.0, 138.0, 141.0, 145.0, 144.0,
      148.0, 151.0, 153.0, 156.0, 160.0, 162.0,
      165.0, 169.0, 171.0, 175.0, 178.0, 181.0,
      184.0, 188.0, 191.0, 194.0, 198.0, 201.0
    )

    require(data.length >= 16, "Need at least 16 observations for this demo.")

    val testSize = 6
    val trainSize = data.length - testSize
    require(trainSize >= 10, "Training set too small.")

    val train = data.take(trainSize)
    val test = data.drop(trainSize)

    println("=== Scala Box-Jenkins ARIMA Demo ===")
    println(s"Total observations : ${data.length}")
    println(s"Training size      : ${train.length}")
    println(s"Test size          : ${test.length}")
    println()

    val trainTs = TimeSeries.from(train*)

    val candidates: Seq[Option[Candidate]] =
      for
        p <- 0 to 2
        d <- 0 to 1
        q <- 0 to 2
        if !(p == 0 && d == 0 && q == 0)
      yield fitCandidate(trainTs, p, d, q)

    val successful: Seq[Candidate] =
      candidates.flatten.sortBy((c: Candidate) => c.aic)

    if successful.isEmpty then
      sys.error("No ARIMA model could be fit successfully on the training data.")

    println("Candidate models ranked by AIC:")
    successful.foreach { c =>
      println(f"  ARIMA(${c.p},${c.d},${c.q})  AIC = ${c.aic}%.4f")
    }
    println()

    val best = successful.head
    println(s"Selected model by minimum AIC: ARIMA(${best.p},${best.d},${best.q})")
    println()

    val forecast = best.model.forecast(test.length)
    val preds = forecast.pointEstimates().asArray()

    println("Forecast vs actual:")
    println("Step\tForecast\tActual\t\tAbsError")
    for i <- test.indices do
      val ae = math.abs(preds(i) - test(i))
      println(f"${i + 1}%d\t${preds(i)}%.4f\t\t${test(i)}%.4f\t\t$ae%.4f")
    println()

    val mae = meanAbsoluteError(test, preds)
    val rmse = rootMeanSquaredError(test, preds)
    val mape = meanAbsolutePercentageError(test, preds)

    println("Error metrics on held-out test set:")
    println(f"  MAE  = $mae%.6f")
    println(f"  RMSE = $rmse%.6f")
    println(f"  MAPE = $mape%.4f%%")
    println()

    println("Model summary:")
    println(best.model)

  def fitCandidate(trainTs: TimeSeries, p: Int, d: Int, q: Int): Option[Candidate] =
    Try {
      val order = ArimaOrder.order(p, d, q)
      val model = Arima.model(trainTs, order)
      val aic = model.aic()
      Candidate(p, d, q, aic, model)
    }.toOption.filter(c => !java.lang.Double.isNaN(c.aic) && !java.lang.Double.isInfinite(c.aic))

  def meanAbsoluteError(actual: Array[Double], predicted: Array[Double]): Double =
    require(actual.length == predicted.length, "Length mismatch in MAE.")
    actual.indices.map(i => math.abs(actual(i) - predicted(i))).sum / actual.length.toDouble

  def rootMeanSquaredError(actual: Array[Double], predicted: Array[Double]): Double =
    require(actual.length == predicted.length, "Length mismatch in RMSE.")
    val mse = actual.indices.map { i =>
      val e = actual(i) - predicted(i)
      e * e
    }.sum / actual.length.toDouble
    math.sqrt(mse)

  def meanAbsolutePercentageError(actual: Array[Double], predicted: Array[Double]): Double =
    require(actual.length == predicted.length, "Length mismatch in MAPE.")
    val nonZeroPairs = actual.indices.filter(i => actual(i) != 0.0)
    require(nonZeroPairs.nonEmpty, "MAPE undefined: all actual values are zero.")
    val frac = nonZeroPairs.map { i =>
      math.abs((actual(i) - predicted(i)) / actual(i))
    }.sum / nonZeroPairs.length.toDouble
    frac * 100.0
EOF

echo "[5/5] Running corrected Box-Jenkins demo..."
scala-cli run "${SRC_FILE}"

echo
echo "Done."
echo "Source file:"
echo "  ${SRC_FILE}"
echo
echo "To rerun later:"
echo "  scala-cli run ${SRC_FILE}"
