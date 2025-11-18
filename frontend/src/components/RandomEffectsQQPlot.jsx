import { TrendingUp, Info, CheckCircle2, AlertTriangle } from 'lucide-react'
import { useState } from 'react'

const RandomEffectsQQPlot = ({ blupsData }) => {
  const [selectedFactor, setSelectedFactor] = useState(null)

  if (!blupsData || Object.keys(blupsData).length === 0) return null

  // Initialize with first factor
  const factors = Object.keys(blupsData)
  const currentFactor = selectedFactor || factors[0]
  const factorData = blupsData[currentFactor]

  if (factorData.error) return null

  const blups = factorData.blups || []
  if (blups.length < 3) return null // Need at least 3 points for meaningful Q-Q plot

  // Calculate theoretical quantiles (normal distribution)
  const calculateTheoreticalQuantiles = (n) => {
    const quantiles = []
    for (let i = 1; i <= n; i++) {
      // Use approximation: Φ^(-1)((i - 0.5) / n)
      const p = (i - 0.5) / n
      // Inverse normal CDF approximation
      const z = inverseNormalCDF(p)
      quantiles.push(z)
    }
    return quantiles
  }

  // Inverse normal CDF (approximation using Beasley-Springer-Moro algorithm)
  const inverseNormalCDF = (p) => {
    const a = [2.515517, 0.802853, 0.010328]
    const b = [1.432788, 0.189269, 0.001308]

    if (p < 0.5) {
      const t = Math.sqrt(-2 * Math.log(p))
      return -(t - (a[0] + a[1] * t + a[2] * t * t) / (1 + b[0] * t + b[1] * t * t + b[2] * t * t * t))
    } else {
      const t = Math.sqrt(-2 * Math.log(1 - p))
      return t - (a[0] + a[1] * t + a[2] * t * t) / (1 + b[0] * t + b[1] * t * t + b[2] * t * t * t)
    }
  }

  // Prepare Q-Q data
  const prepareQQData = () => {
    // Extract BLUP values and sort them
    const blupValues = blups.map(b => b.blup).sort((a, b) => a - b)

    // Standardize BLUPs (subtract mean, divide by SD)
    const mean = blupValues.reduce((sum, val) => sum + val, 0) / blupValues.length
    const variance = blupValues.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / blupValues.length
    const sd = Math.sqrt(variance)

    const standardizedBlups = blupValues.map(val => (val - mean) / (sd || 1))

    // Get theoretical quantiles
    const theoreticalQuantiles = calculateTheoreticalQuantiles(standardizedBlups.length)

    // Pair them up
    const qqData = standardizedBlups.map((sample, i) => ({
      theoretical: theoreticalQuantiles[i],
      sample: sample
    }))

    return { qqData, standardizedBlups, theoreticalQuantiles }
  }

  const { qqData, standardizedBlups, theoreticalQuantiles } = prepareQQData()

  // Calculate correlation coefficient (measure of normality)
  const calculateCorrelation = () => {
    const n = qqData.length
    const meanTheoretical = theoreticalQuantiles.reduce((sum, val) => sum + val, 0) / n
    const meanSample = standardizedBlups.reduce((sum, val) => sum + val, 0) / n

    let numerator = 0
    let denomTheoretical = 0
    let denomSample = 0

    for (let i = 0; i < n; i++) {
      const devTheoretical = theoreticalQuantiles[i] - meanTheoretical
      const devSample = standardizedBlups[i] - meanSample
      numerator += devTheoretical * devSample
      denomTheoretical += devTheoretical * devTheoretical
      denomSample += devSample * devSample
    }

    const r = numerator / Math.sqrt(denomTheoretical * denomSample)
    return r
  }

  const correlation = calculateCorrelation()

  // Assess normality based on correlation
  const assessNormality = () => {
    if (correlation > 0.98) return { status: 'excellent', color: 'green', message: 'Excellent fit to normal distribution' }
    if (correlation > 0.95) return { status: 'good', color: 'blue', message: 'Good fit to normal distribution' }
    if (correlation > 0.90) return { status: 'acceptable', color: 'yellow', message: 'Acceptable fit to normal distribution' }
    return { status: 'poor', color: 'red', message: 'Poor fit - normality assumption may be violated' }
  }

  const normalityAssessment = assessNormality()

  // Q-Q Plot
  const QQPlotSVG = () => {
    const plotWidth = 450
    const plotHeight = 450
    const marginLeft = 60
    const marginRight = 30
    const marginTop = 30
    const marginBottom = 60

    // Find ranges for scaling
    const minTheoretical = Math.min(...theoreticalQuantiles)
    const maxTheoretical = Math.max(...theoreticalQuantiles)
    const minSample = Math.min(...standardizedBlups)
    const maxSample = Math.max(...standardizedBlups)

    const xScale = (val) => {
      return marginLeft + ((val - minTheoretical) / (maxTheoretical - minTheoretical)) * (plotWidth - marginLeft - marginRight)
    }

    const yScale = (val) => {
      return plotHeight - marginBottom - ((val - minSample) / (maxSample - minSample)) * (plotHeight - marginTop - marginBottom)
    }

    // Reference line (y = x)
    const refLineStart = Math.max(minTheoretical, minSample)
    const refLineEnd = Math.min(maxTheoretical, maxSample)

    return (
      <svg width={plotWidth} height={plotHeight} className="bg-slate-900/30 rounded-lg">
        {/* Grid lines */}
        {[...Array(5)].map((_, i) => {
          const valueX = minTheoretical + (i * (maxTheoretical - minTheoretical)) / 4
          const valueY = minSample + (i * (maxSample - minSample)) / 4
          const x = xScale(valueX)
          const y = yScale(valueY)

          return (
            <g key={i}>
              {/* Vertical grid */}
              <line
                x1={x}
                y1={marginTop}
                x2={x}
                y2={plotHeight - marginBottom}
                stroke="currentColor"
                className="text-slate-700"
                strokeWidth="1"
                opacity="0.3"
              />
              {/* Horizontal grid */}
              <line
                x1={marginLeft}
                y1={y}
                x2={plotWidth - marginRight}
                y2={y}
                stroke="currentColor"
                className="text-slate-700"
                strokeWidth="1"
                opacity="0.3"
              />
              {/* X-axis labels */}
              <text
                x={x}
                y={plotHeight - marginBottom + 20}
                fill="currentColor"
                className="text-gray-400"
                fontSize="10"
                textAnchor="middle"
              >
                {valueX.toFixed(1)}
              </text>
              {/* Y-axis labels */}
              <text
                x={marginLeft - 10}
                y={y}
                fill="currentColor"
                className="text-gray-400"
                fontSize="10"
                textAnchor="end"
                dominantBaseline="middle"
              >
                {valueY.toFixed(1)}
              </text>
            </g>
          )
        })}

        {/* Reference line (y = x) */}
        <line
          x1={xScale(refLineStart)}
          y1={yScale(refLineStart)}
          x2={xScale(refLineEnd)}
          y2={yScale(refLineEnd)}
          stroke="currentColor"
          className="text-red-400"
          strokeWidth="2"
          strokeDasharray="4,4"
        />

        {/* Data points */}
        {qqData.map((point, i) => (
          <circle
            key={i}
            cx={xScale(point.theoretical)}
            cy={yScale(point.sample)}
            r="4"
            fill="#8b5cf6"
            stroke="white"
            strokeWidth="1.5"
            opacity="0.8"
          />
        ))}

        {/* Axes */}
        <line
          x1={marginLeft}
          y1={plotHeight - marginBottom}
          x2={plotWidth - marginRight}
          y2={plotHeight - marginBottom}
          stroke="currentColor"
          className="text-gray-500"
          strokeWidth="2"
        />
        <line
          x1={marginLeft}
          y1={marginTop}
          x2={marginLeft}
          y2={plotHeight - marginBottom}
          stroke="currentColor"
          className="text-gray-500"
          strokeWidth="2"
        />

        {/* Axis labels */}
        <text
          x={plotWidth / 2}
          y={plotHeight - 10}
          fill="currentColor"
          className="text-gray-300"
          fontSize="12"
          textAnchor="middle"
        >
          Theoretical Quantiles
        </text>
        <text
          x={15}
          y={plotHeight / 2}
          fill="currentColor"
          className="text-gray-300"
          fontSize="12"
          textAnchor="middle"
          transform={`rotate(-90, 15, ${plotHeight / 2})`}
        >
          Sample Quantiles (Standardized BLUPs)
        </text>

        {/* Title */}
        <text
          x={plotWidth / 2}
          y={15}
          fill="currentColor"
          className="text-gray-200"
          fontSize="13"
          fontWeight="bold"
          textAnchor="middle"
        >
          Normal Q-Q Plot
        </text>
      </svg>
    )
  }

  return (
    <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
      <div className="flex items-center space-x-3 mb-4">
        <TrendingUp className="w-6 h-6 text-cyan-400" />
        <h3 className="text-xl font-bold text-gray-100">Random Effects Normality Check</h3>
      </div>

      <p className="text-gray-400 text-sm mb-6">
        Q-Q plot checks if random effects follow a normal distribution, a key assumption of mixed models.
      </p>

      {/* Factor selector */}
      {factors.length > 1 && (
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-300 mb-2">Select Random Factor:</label>
          <div className="flex space-x-2">
            {factors.map(factor => (
              <button
                key={factor}
                onClick={() => setSelectedFactor(factor)}
                className={`px-4 py-2 rounded-lg transition-colors ${
                  currentFactor === factor
                    ? 'bg-cyan-500 text-white'
                    : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
                }`}
              >
                {factor}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Normality assessment */}
      <div className={`mb-6 bg-${normalityAssessment.color}-500/10 border border-${normalityAssessment.color}-500/50 rounded-lg p-4 flex items-start space-x-3`}>
        {normalityAssessment.status === 'excellent' || normalityAssessment.status === 'good' ? (
          <CheckCircle2 className={`w-5 h-5 text-${normalityAssessment.color}-400 mt-0.5`} />
        ) : (
          <AlertTriangle className={`w-5 h-5 text-${normalityAssessment.color}-400 mt-0.5`} />
        )}
        <div>
          <h5 className={`font-semibold text-${normalityAssessment.color}-300 mb-1`}>
            {normalityAssessment.message}
          </h5>
          <p className={`text-sm text-${normalityAssessment.color}-200`}>
            Correlation coefficient: <span className="font-mono font-bold">{correlation.toFixed(4)}</span>
            {normalityAssessment.status === 'poor' && (
              <span className="block mt-1">
                Consider transforming your data or using robust mixed models if normality is severely violated.
              </span>
            )}
          </p>
        </div>
      </div>

      {/* Q-Q Plot */}
      <div className="flex justify-center mb-6">
        <QQPlotSVG />
      </div>

      {/* Interpretation guide */}
      <div className="bg-slate-700/30 rounded-lg p-4">
        <h5 className="font-semibold text-gray-200 mb-3 flex items-center">
          <Info className="w-4 h-4 mr-2" />
          Interpreting the Q-Q Plot
        </h5>
        <div className="space-y-2 text-xs text-gray-400">
          <p>
            <strong className="text-gray-300">Points on the line:</strong> Random effects follow a normal distribution.
            This is ideal and supports the model assumptions.
          </p>
          <p>
            <strong className="text-gray-300">Points above the line:</strong> Heavy-tailed distribution (more extreme values than expected).
            May indicate outliers or non-normality.
          </p>
          <p>
            <strong className="text-gray-300">Points below the line:</strong> Light-tailed distribution (fewer extreme values than expected).
            May indicate clustering or non-normality.
          </p>
          <p>
            <strong className="text-gray-300">S-shaped pattern:</strong> Skewed distribution. Consider data transformation.
          </p>
        </div>
      </div>
    </div>
  )
}

export default RandomEffectsQQPlot
