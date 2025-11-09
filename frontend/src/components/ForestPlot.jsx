import { useRef } from 'react'
import { Target, Download } from 'lucide-react'
import { exportSvgToPng } from '../utils/exportChart'

const ForestPlot = ({ pilotResult }) => {
  const svgRef = useRef(null)

  if (!pilotResult || pilotResult.error) return null

  // Extract data
  const cohensD = pilotResult.cohens_d
  const cohensF = pilotResult.cohens_f
  const etaSquared = pilotResult.eta_squared
  const ci = pilotResult.confidence_interval_95

  // Determine what to display
  const hasCI = ci !== undefined
  const effectSizeType = cohensD !== undefined ? 'd' : (cohensF !== undefined ? 'f' : 'η²')
  const effectSizeValue = cohensD !== undefined ? cohensD : (cohensF !== undefined ? cohensF : etaSquared)

  // SVG dimensions
  const width = 800
  const height = 300
  const margin = { top: 60, right: 100, bottom: 60, left: 150 }
  const plotWidth = width - margin.left - margin.right
  const plotHeight = height - margin.top - margin.bottom

  // Determine scale range based on effect size type and CI
  let xMin, xMax
  if (hasCI && cohensD !== undefined) {
    xMin = Math.min(0, ci.lower - 0.2)
    xMax = Math.max(ci.upper + 0.2, 1.5)
  } else if (cohensF !== undefined) {
    xMin = 0
    xMax = Math.max(cohensF * 1.5, 0.6)
  } else {
    xMin = 0
    xMax = Math.min(1, Math.max(effectSizeValue * 1.5, 0.3))
  }

  const xRange = xMax - xMin
  const xScale = (x) => ((x - xMin) / xRange) * plotWidth

  // Cohen's conventions for reference lines
  let referenceLines = []
  if (effectSizeType === 'd') {
    referenceLines = [
      { value: 0.2, label: 'Small', color: '#fbbf24' },
      { value: 0.5, label: 'Medium', color: '#fb923c' },
      { value: 0.8, label: 'Large', color: '#f87171' }
    ]
  } else if (effectSizeType === 'f') {
    referenceLines = [
      { value: 0.1, label: 'Small', color: '#fbbf24' },
      { value: 0.25, label: 'Medium', color: '#fb923c' },
      { value: 0.4, label: 'Large', color: '#f87171' }
    ]
  } else if (effectSizeType === 'η²') {
    referenceLines = [
      { value: 0.01, label: 'Small', color: '#fbbf24' },
      { value: 0.06, label: 'Medium', color: '#fb923c' },
      { value: 0.14, label: 'Large', color: '#f87171' }
    ]
  }

  // Generate x-axis ticks
  const numTicks = 8
  const tickStep = xRange / numTicks
  const xTicks = []
  for (let i = 0; i <= numTicks; i++) {
    xTicks.push(xMin + i * tickStep)
  }

  const yCenter = plotHeight / 2

  return (
    <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50 mt-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <Target className="w-6 h-6 text-cyan-400" />
          <h4 className="text-xl font-bold text-gray-100">Effect Size Visualization</h4>
        </div>

        <button
          onClick={() => {
            if (svgRef.current) {
              const filename = `forest-plot-${effectSizeType}-${new Date().toISOString().split('T')[0]}`
              exportSvgToPng(svgRef.current, filename)
            }
          }}
          className="px-3 py-2 rounded-lg text-sm font-medium bg-slate-700/50 text-gray-300 hover:bg-slate-700 transition-all flex items-center space-x-2"
          title="Export as PNG"
        >
          <Download className="w-4 h-4" />
          <span>Export PNG</span>
        </button>
      </div>

      <p className="text-gray-400 text-sm mb-4">
        {hasCI
          ? 'Forest plot showing the estimated effect size with 95% confidence interval. The vertical reference lines indicate conventional thresholds.'
          : 'Effect size estimate with reference thresholds for interpretation.'}
      </p>

      <div className="bg-slate-900/50 rounded-lg p-4">
        <svg ref={svgRef} width={width} height={height} className="mx-auto">
          <g transform={`translate(${margin.left}, ${margin.top})`}>
            {/* Reference lines */}
            {referenceLines.map((ref, i) => (
              xScale(ref.value) >= 0 && xScale(ref.value) <= plotWidth && (
                <g key={`ref-${i}`}>
                  <line
                    x1={xScale(ref.value)}
                    y1={0}
                    x2={xScale(ref.value)}
                    y2={plotHeight}
                    stroke={ref.color}
                    strokeWidth="2"
                    strokeDasharray="6,4"
                    opacity="0.4"
                  />
                  <text
                    x={xScale(ref.value)}
                    y={-10}
                    fill={ref.color}
                    fontSize="11"
                    textAnchor="middle"
                    fontWeight="600"
                  >
                    {ref.label}
                  </text>
                </g>
              )
            ))}

            {/* Zero line (for Cohen's d only) */}
            {effectSizeType === 'd' && xScale(0) >= 0 && xScale(0) <= plotWidth && (
              <line
                x1={xScale(0)}
                y1={0}
                x2={xScale(0)}
                y2={plotHeight}
                stroke="#6b7280"
                strokeWidth="2"
                opacity="0.5"
              />
            )}

            {/* X-axis */}
            <line
              x1={0}
              y1={plotHeight}
              x2={plotWidth}
              y2={plotHeight}
              stroke="#9ca3af"
              strokeWidth="2"
            />

            {/* X-axis ticks */}
            {xTicks.map((tick, i) => (
              <g key={`x-tick-${i}`}>
                <line
                  x1={xScale(tick)}
                  y1={plotHeight}
                  x2={xScale(tick)}
                  y2={plotHeight + 5}
                  stroke="#9ca3af"
                  strokeWidth="2"
                />
                <text
                  x={xScale(tick)}
                  y={plotHeight + 20}
                  fill="#d1d5db"
                  fontSize="12"
                  textAnchor="middle"
                >
                  {tick.toFixed(2)}
                </text>
              </g>
            ))}

            {/* Confidence interval (if available) */}
            {hasCI && (
              <g>
                {/* CI line */}
                <line
                  x1={xScale(ci.lower)}
                  y1={yCenter}
                  x2={xScale(ci.upper)}
                  y2={yCenter}
                  stroke="#3b82f6"
                  strokeWidth="3"
                  strokeLinecap="round"
                />

                {/* CI endpoints */}
                <line
                  x1={xScale(ci.lower)}
                  y1={yCenter - 10}
                  x2={xScale(ci.lower)}
                  y2={yCenter + 10}
                  stroke="#3b82f6"
                  strokeWidth="3"
                  strokeLinecap="round"
                />
                <line
                  x1={xScale(ci.upper)}
                  y1={yCenter - 10}
                  x2={xScale(ci.upper)}
                  y2={yCenter + 10}
                  stroke="#3b82f6"
                  strokeWidth="3"
                  strokeLinecap="round"
                />
              </g>
            )}

            {/* Point estimate (diamond) */}
            <g>
              <path
                d={`M ${xScale(effectSizeValue)} ${yCenter - 12}
                    L ${xScale(effectSizeValue) + 8} ${yCenter}
                    L ${xScale(effectSizeValue)} ${yCenter + 12}
                    L ${xScale(effectSizeValue) - 8} ${yCenter}
                    Z`}
                fill="#06b6d4"
                stroke="#0891b2"
                strokeWidth="2"
              />
            </g>

            {/* Value label */}
            <text
              x={xScale(effectSizeValue)}
              y={yCenter - 25}
              fill="#06b6d4"
              fontSize="14"
              fontWeight="700"
              textAnchor="middle"
            >
              {effectSizeType}: {effectSizeValue.toFixed(3)}
            </text>

            {/* X-axis label */}
            <text
              x={plotWidth / 2}
              y={plotHeight + 45}
              fill="#d1d5db"
              fontSize="14"
              fontWeight="600"
              textAnchor="middle"
            >
              Effect Size (Cohen's {effectSizeType})
            </text>

            {/* Y-axis label (Study name) */}
            <text
              x={-20}
              y={yCenter}
              fill="#e5e7eb"
              fontSize="13"
              fontWeight="600"
              textAnchor="end"
              dominantBaseline="middle"
            >
              Pilot Study
            </text>

            {/* Title */}
            <text
              x={plotWidth / 2}
              y={-35}
              fill="#e5e7eb"
              fontSize="16"
              fontWeight="700"
              textAnchor="middle"
            >
              Effect Size Estimate from Pilot Data
            </text>

            {/* Subtitle with CI */}
            {hasCI && (
              <text
                x={plotWidth / 2}
                y={-15}
                fill="#9ca3af"
                fontSize="12"
                textAnchor="middle"
              >
                95% CI: [{ci.lower.toFixed(3)}, {ci.upper.toFixed(3)}]
              </text>
            )}
          </g>
        </svg>
      </div>

      {/* Interpretation */}
      <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-slate-700/30 rounded-lg p-4">
          <h5 className="text-gray-200 font-semibold mb-2 text-sm">Interpretation</h5>
          <p className="text-gray-400 text-xs leading-relaxed">
            {pilotResult.interpretation || pilotResult.interpretation_f}
            {hasCI && (
              <span className="block mt-2">
                The confidence interval shows the range of plausible values.
                Wider intervals indicate more uncertainty in the estimate.
              </span>
            )}
          </p>
        </div>

        <div className="bg-slate-700/30 rounded-lg p-4">
          <h5 className="text-gray-200 font-semibold mb-2 text-sm">Using This Estimate</h5>
          <p className="text-gray-400 text-xs leading-relaxed">
            Effect sizes from pilot studies can be used for power analysis, but use caution:
            small pilots tend to overestimate effects. Consider using the lower bound of the CI
            for more conservative sample size planning.
          </p>
        </div>
      </div>
    </div>
  )
}

export default ForestPlot
