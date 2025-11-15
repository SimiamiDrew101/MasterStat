import { LineChart, TrendingUp, AlertCircle } from 'lucide-react'

const DiagnosticPlots = ({ diagnosticPlots }) => {
  if (!diagnosticPlots) return null

  const { scale_location, leverage_residuals } = diagnosticPlots

  // Helper function to normalize values to plot coordinates
  const normalizeValue = (value, min, max, height) => {
    const normalized = (value - min) / (max - min)
    return height - (normalized * height)
  }

  const renderScatterPlot = (xData, yData, xLabel, yLabel, threshold = null) => {
    const plotWidth = 500
    const plotHeight = 300
    const padding = 40

    const xMin = Math.min(...xData)
    const xMax = Math.max(...xData)

    // If there's a threshold, ensure it's included in the y-range
    let yMin = Math.min(...yData)
    let yMax = Math.max(...yData)
    if (threshold !== null) {
      yMin = Math.min(yMin, threshold)
      yMax = Math.max(yMax, threshold)
    }

    // Add 10% padding to the ranges
    const xRange = xMax - xMin
    const yRange = yMax - yMin
    const xPadding = xRange * 0.1
    const yPadding = yRange * 0.1

    // Adjust min/max with padding
    const xMinPadded = xMin - xPadding
    const xMaxPadded = xMax + xPadding
    const yMinPadded = yMin - yPadding
    const yMaxPadded = yMax + yPadding

    const points = xData.map((x, i) => {
      const xPos = padding + ((x - xMinPadded) / (xMaxPadded - xMinPadded)) * (plotWidth - 2 * padding)
      const yPos = plotHeight - padding - ((yData[i] - yMinPadded) / (yMaxPadded - yMinPadded)) * (plotHeight - 2 * padding)
      return { x: xPos, y: yPos, originalX: x, originalY: yData[i] }
    })

    // Calculate where y=0 is positioned in the plot (for threshold line)
    const y0Position = threshold !== null
      ? plotHeight - padding - ((threshold - yMinPadded) / (yMaxPadded - yMinPadded)) * (plotHeight - 2 * padding)
      : null

    return (
      <div className="relative bg-slate-900/50 rounded-lg p-4">
        <svg width={plotWidth} height={plotHeight} className="mx-auto">
          {/* Background grid */}
          <defs>
            <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
              <path d="M 50 0 L 0 0 0 50" fill="none" stroke="rgba(148, 163, 184, 0.1)" strokeWidth="1"/>
            </pattern>
          </defs>
          <rect width={plotWidth} height={plotHeight} fill="url(#grid)" />

          {/* Axes */}
          <line
            x1={padding}
            y1={plotHeight - padding}
            x2={plotWidth - padding}
            y2={plotHeight - padding}
            stroke="rgba(148, 163, 184, 0.5)"
            strokeWidth="2"
          />
          <line
            x1={padding}
            y1={padding}
            x2={padding}
            y2={plotHeight - padding}
            stroke="rgba(148, 163, 184, 0.5)"
            strokeWidth="2"
          />

          {/* Data points */}
          {points.map((point, i) => (
            <circle
              key={i}
              cx={point.x}
              cy={point.y}
              r="4"
              fill="rgba(59, 130, 246, 0.7)"
              stroke="rgba(96, 165, 250, 1)"
              strokeWidth="1"
              className="hover:fill-blue-400 transition-colors cursor-pointer"
            >
              <title>({point.originalX.toFixed(2)}, {point.originalY.toFixed(2)})</title>
            </circle>
          ))}

          {/* Reference line at y=0 for leverage-residuals plot */}
          {y0Position !== null && (
            <line
              x1={padding}
              y1={y0Position}
              x2={plotWidth - padding}
              y2={y0Position}
              stroke="rgba(248, 113, 113, 0.5)"
              strokeWidth="2"
              strokeDasharray="5,5"
            />
          )}

          {/* Axis labels */}
          <text
            x={plotWidth / 2}
            y={plotHeight - 5}
            textAnchor="middle"
            fill="rgba(203, 213, 225, 0.9)"
            fontSize="12"
          >
            {xLabel}
          </text>
          <text
            x={15}
            y={plotHeight / 2}
            textAnchor="middle"
            fill="rgba(203, 213, 225, 0.9)"
            fontSize="12"
            transform={`rotate(-90, 15, ${plotHeight / 2})`}
          >
            {yLabel}
          </text>
        </svg>
      </div>
    )
  }

  return (
    <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
      <div className="flex items-center space-x-3 mb-4">
        <LineChart className="w-6 h-6 text-cyan-400" />
        <h3 className="text-xl font-bold text-gray-100">Diagnostic Plots</h3>
      </div>

      <p className="text-gray-400 text-sm mb-6">
        Additional diagnostic plots to assess model assumptions and identify potential issues with the ANOVA model.
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Scale-Location Plot */}
        <div className="bg-slate-700/30 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-3">
            <TrendingUp className="w-5 h-5 text-cyan-400" />
            <h4 className="font-semibold text-gray-100">Scale-Location Plot</h4>
          </div>
          <p className="text-xs text-gray-400 mb-4">{scale_location.interpretation}</p>

          {renderScatterPlot(
            scale_location.fitted_values,
            scale_location.sqrt_abs_std_residuals,
            "Fitted Values",
            "√|Standardized Residuals|"
          )}

          <div className="mt-3 bg-slate-800/50 rounded-lg p-3">
            <h5 className="text-xs font-semibold text-gray-300 mb-2">How to interpret:</h5>
            <ul className="text-xs text-gray-400 space-y-1">
              <li>• <strong>Good:</strong> Random scatter with no pattern</li>
              <li>• <strong>Warning:</strong> Funnel shape (heteroscedasticity)</li>
              <li>• <strong>Warning:</strong> Curved pattern (non-linearity)</li>
            </ul>
          </div>
        </div>

        {/* Residuals vs Leverage Plot */}
        <div className="bg-slate-700/30 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-3">
            <AlertCircle className="w-5 h-5 text-orange-400" />
            <h4 className="font-semibold text-gray-100">Residuals vs Leverage</h4>
          </div>
          <p className="text-xs text-gray-400 mb-4">{leverage_residuals.interpretation}</p>

          {renderScatterPlot(
            leverage_residuals.leverage,
            leverage_residuals.std_residuals,
            "Leverage",
            "Standardized Residuals",
            0
          )}

          <div className="mt-3 bg-slate-800/50 rounded-lg p-3">
            <h5 className="text-xs font-semibold text-gray-300 mb-2">How to interpret:</h5>
            <ul className="text-xs text-gray-400 space-y-1">
              <li>• <strong>High Leverage:</strong> Points far from mean predictor values</li>
              <li>• <strong>High Residual:</strong> Points poorly fit by the model</li>
              <li>• <strong>Concerning:</strong> Points with both high leverage AND high residuals</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Overall Guidance */}
      <div className="mt-6 bg-blue-900/20 border border-blue-700/50 rounded-lg p-4">
        <h5 className="font-semibold text-blue-200 mb-2">Diagnostic Plots Summary</h5>
        <p className="text-sm text-blue-100/90">
          These plots complement the influence diagnostics and assumption tests. Use them to visually
          identify patterns that violate ANOVA assumptions:
        </p>
        <ul className="text-sm text-blue-100/80 mt-2 space-y-1 ml-4">
          <li>• <strong>Scale-Location</strong> checks for homoscedasticity (equal variance assumption)</li>
          <li>• <strong>Residuals vs Leverage</strong> identifies influential observations that may distort results</li>
        </ul>
      </div>
    </div>
  )
}

export default DiagnosticPlots
