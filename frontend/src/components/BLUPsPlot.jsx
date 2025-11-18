import { BarChart3, Info, AlertTriangle } from 'lucide-react'
import { useState } from 'react'

const BLUPsPlot = ({ blupsData }) => {
  const [selectedFactor, setSelectedFactor] = useState(null)
  const [showDetails, setShowDetails] = useState(false)

  if (!blupsData || Object.keys(blupsData).length === 0) return null

  // Initialize with first factor
  const factors = Object.keys(blupsData)
  const currentFactor = selectedFactor || factors[0]
  const factorData = blupsData[currentFactor]

  if (factorData.error) {
    return (
      <div className="bg-red-500/10 border border-red-500/50 rounded-lg p-4">
        <p className="text-red-400 text-sm">
          <strong>Error extracting BLUPs:</strong> {factorData.error}
        </p>
      </div>
    )
  }

  const blups = factorData.blups || []
  const summary = factorData.summary || {}

  // Caterpillar Plot (sorted BLUPs with CI)
  const CaterpillarPlot = () => {
    if (blups.length === 0) return null

    const maxAbsValue = Math.max(...blups.map(b => Math.max(Math.abs(b.ci_lower), Math.abs(b.ci_upper))))
    const plotWidth = 600
    const plotHeight = Math.min(600, blups.length * 30 + 60)
    const marginLeft = 100
    const marginRight = 40
    const marginTop = 40
    const marginBottom = 40
    const barHeight = 20
    const spacing = Math.max(5, (plotHeight - marginTop - marginBottom - blups.length * barHeight) / (blups.length + 1))

    const xScale = (value) => {
      return marginLeft + ((value + maxAbsValue) / (2 * maxAbsValue)) * (plotWidth - marginLeft - marginRight)
    }

    const zeroX = xScale(0)

    // Identify potential outliers (> 2 SE from zero)
    const identifyOutliers = () => {
      return blups.map((b, i) => {
        const isOutlier = Math.abs(b.blup / b.se) > 2
        return { ...b, index: i, isOutlier }
      })
    }

    const blupsWithOutliers = identifyOutliers()

    return (
      <div className="flex flex-col items-center">
        <svg width={plotWidth} height={plotHeight} className="bg-slate-900/30 rounded-lg">
          {/* Grid lines */}
          {[...Array(5)].map((_, i) => {
            const value = -maxAbsValue + (i * 2 * maxAbsValue) / 4
            const x = xScale(value)
            return (
              <g key={i}>
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
                <text
                  x={x}
                  y={plotHeight - marginBottom + 20}
                  fill="currentColor"
                  className="text-gray-400"
                  fontSize="10"
                  textAnchor="middle"
                >
                  {value.toFixed(2)}
                </text>
              </g>
            )
          })}

          {/* Zero line (reference) */}
          <line
            x1={zeroX}
            y1={marginTop}
            x2={zeroX}
            y2={plotHeight - marginBottom}
            stroke="currentColor"
            className="text-blue-400"
            strokeWidth="2"
            strokeDasharray="4,4"
          />

          {/* BLUPs with CIs */}
          {blupsWithOutliers.map((blup, i) => {
            const y = marginTop + spacing + i * (barHeight + spacing) + barHeight / 2
            const xBlup = xScale(blup.blup)
            const xCILower = xScale(blup.ci_lower)
            const xCIUpper = xScale(blup.ci_upper)

            const color = blup.isOutlier ? '#ef4444' : '#3b82f6'
            const textColor = blup.isOutlier ? '#fca5a5' : '#93c5fd'

            return (
              <g key={i}>
                {/* CI line */}
                <line
                  x1={xCILower}
                  y1={y}
                  x2={xCIUpper}
                  y2={y}
                  stroke={color}
                  strokeWidth="2"
                  opacity="0.6"
                />

                {/* CI caps */}
                <line
                  x1={xCILower}
                  y1={y - 5}
                  x2={xCILower}
                  y2={y + 5}
                  stroke={color}
                  strokeWidth="2"
                />
                <line
                  x1={xCIUpper}
                  y1={y - 5}
                  x2={xCIUpper}
                  y2={y + 5}
                  stroke={color}
                  strokeWidth="2"
                />

                {/* BLUP point */}
                <circle
                  cx={xBlup}
                  cy={y}
                  r="4"
                  fill={color}
                  stroke="white"
                  strokeWidth="1.5"
                />

                {/* Level label */}
                <text
                  x={marginLeft - 10}
                  y={y}
                  fill={textColor}
                  fontSize="10"
                  textAnchor="end"
                  dominantBaseline="middle"
                >
                  {blup.level}
                </text>

                {/* Outlier indicator */}
                {blup.isOutlier && (
                  <g>
                    <circle
                      cx={marginLeft - 85}
                      cy={y}
                      r="6"
                      fill="#ef4444"
                      opacity="0.2"
                    />
                    <text
                      x={marginLeft - 85}
                      y={y}
                      fill="#ef4444"
                      fontSize="10"
                      textAnchor="middle"
                      dominantBaseline="middle"
                      fontWeight="bold"
                    >
                      !
                    </text>
                  </g>
                )}
              </g>
            )
          })}

          {/* Title */}
          <text
            x={plotWidth / 2}
            y={20}
            fill="currentColor"
            className="text-gray-200"
            fontSize="14"
            fontWeight="bold"
            textAnchor="middle"
          >
            BLUPs with 95% Confidence Intervals
          </text>

          {/* X-axis label */}
          <text
            x={plotWidth / 2}
            y={plotHeight - 5}
            fill="currentColor"
            className="text-gray-400"
            fontSize="11"
            textAnchor="middle"
          >
            BLUP Value (Deviation from Grand Mean)
          </text>
        </svg>

        {/* Legend */}
        <div className="mt-4 flex items-center space-x-6 text-sm">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-blue-500"></div>
            <span className="text-gray-300">Normal Range</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-red-500"></div>
            <span className="text-gray-300">Potential Outlier</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-8 h-0.5 bg-blue-400 border-dashed"></div>
            <span className="text-gray-300">Zero Reference</span>
          </div>
        </div>
      </div>
    )
  }

  // Shrinkage Comparison
  const ShrinkageComparison = () => {
    if (blups.length === 0) return null

    const plotWidth = 500
    const plotHeight = 400
    const marginLeft = 60
    const marginRight = 20
    const marginTop = 30
    const marginBottom = 50

    const maxObserved = Math.max(...blups.map(b => Math.abs(b.observed_deviation)))
    const maxBlup = Math.max(...blups.map(b => Math.abs(b.blup)))
    const maxValue = Math.max(maxObserved, maxBlup)

    const scale = (value) => {
      return ((value + maxValue) / (2 * maxValue))
    }

    return (
      <div className="bg-slate-900/30 rounded-lg p-6">
        <h4 className="text-sm font-semibold text-gray-200 mb-4 flex items-center">
          <Info className="w-4 h-4 mr-2" />
          Shrinkage Visualization
        </h4>
        <div className="flex justify-center">
          <svg width={plotWidth} height={plotHeight}>
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

            {/* Diagonal reference line (no shrinkage) */}
            <line
              x1={marginLeft}
              y1={plotHeight - marginBottom}
              x2={plotWidth - marginRight}
              y2={marginTop}
              stroke="currentColor"
              className="text-yellow-500"
              strokeWidth="1"
              strokeDasharray="4,4"
              opacity="0.5"
            />

            {/* Points */}
            {blups.map((blup, i) => {
              const x = marginLeft + scale(blup.observed_deviation) * (plotWidth - marginLeft - marginRight)
              const y = plotHeight - marginBottom - scale(blup.blup) * (plotHeight - marginTop - marginBottom)

              return (
                <g key={i}>
                  <circle
                    cx={x}
                    cy={y}
                    r="4"
                    fill="#8b5cf6"
                    stroke="white"
                    strokeWidth="1"
                    opacity="0.7"
                  />
                </g>
              )
            })}

            {/* Axis labels */}
            <text
              x={plotWidth / 2}
              y={plotHeight - 10}
              fill="currentColor"
              className="text-gray-400"
              fontSize="11"
              textAnchor="middle"
            >
              Observed Deviation
            </text>
            <text
              x={20}
              y={plotHeight / 2}
              fill="currentColor"
              className="text-gray-400"
              fontSize="11"
              textAnchor="middle"
              transform={`rotate(-90, 20, ${plotHeight / 2})`}
            >
              BLUP
            </text>
          </svg>
        </div>
        <p className="text-xs text-gray-400 mt-3">
          Points below the diagonal line show shrinkage toward zero. More shrinkage indicates less reliable group estimates.
        </p>
      </div>
    )
  }

  return (
    <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <BarChart3 className="w-6 h-6 text-orange-400" />
          <h3 className="text-xl font-bold text-gray-100">Random Effects (BLUPs)</h3>
        </div>
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="text-gray-400 hover:text-gray-200 transition-colors"
        >
          <Info className="w-5 h-5" />
        </button>
      </div>

      <p className="text-gray-400 text-sm mb-6">
        Best Linear Unbiased Predictions (BLUPs) show the estimated deviation of each group from the overall mean,
        accounting for shrinkage based on data reliability.
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
                    ? 'bg-orange-500 text-white'
                    : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
                }`}
              >
                {factor}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Summary statistics */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
        <div className="bg-slate-700/30 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Levels</div>
          <div className="text-lg font-bold text-gray-200">{factorData.n_levels}</div>
        </div>
        <div className="bg-slate-700/30 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Mean BLUP</div>
          <div className="text-lg font-mono text-gray-200">{summary.mean_blup?.toFixed(3)}</div>
        </div>
        <div className="bg-slate-700/30 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">SD</div>
          <div className="text-lg font-mono text-gray-200">{summary.std_blup?.toFixed(3)}</div>
        </div>
        <div className="bg-slate-700/30 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Range</div>
          <div className="text-sm font-mono text-gray-200">
            [{summary.min_blup?.toFixed(2)}, {summary.max_blup?.toFixed(2)}]
          </div>
        </div>
        <div className="bg-slate-700/30 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Avg Shrinkage</div>
          <div className="text-lg font-mono text-gray-200">{(summary.mean_shrinkage * 100)?.toFixed(1)}%</div>
        </div>
      </div>

      {/* Caterpillar Plot */}
      <div className="mb-6">
        <CaterpillarPlot />
      </div>

      {/* Shrinkage Comparison */}
      {showDetails && <ShrinkageComparison />}

      {/* Outlier warning */}
      {blups.some(b => Math.abs(b.blup / b.se) > 2) && (
        <div className="mt-6 bg-red-500/10 border border-red-500/50 rounded-lg p-4 flex items-start space-x-3">
          <AlertTriangle className="w-5 h-5 text-red-400 mt-0.5" />
          <div>
            <h5 className="font-semibold text-red-300 mb-1">Potential Outliers Detected</h5>
            <p className="text-sm text-red-200">
              Some random effects deviate significantly from zero (|BLUP/SE| {'>'} 2). These groups may have unusual characteristics worth investigating.
            </p>
          </div>
        </div>
      )}

      {/* Interpretation guide */}
      {showDetails && (
        <div className="mt-6 bg-slate-700/30 rounded-lg p-4">
          <h5 className="font-semibold text-gray-200 mb-3 flex items-center">
            <Info className="w-4 h-4 mr-2" />
            Understanding BLUPs
          </h5>
          <div className="space-y-2 text-xs text-gray-400">
            <p>
              <strong className="text-gray-300">Shrinkage:</strong> BLUPs are "shrunk" toward zero compared to observed means.
              Groups with less data or higher within-group variability experience more shrinkage.
            </p>
            <p>
              <strong className="text-gray-300">Confidence Intervals:</strong> Wider intervals indicate less certainty about the true random effect.
              Groups with more observations have narrower intervals.
            </p>
            <p>
              <strong className="text-gray-300">Outliers:</strong> Groups where the confidence interval doesn't include zero may be systematically different
              from the population average.
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

export default BLUPsPlot
