import { useRef } from 'react'
import { Download } from 'lucide-react'
import { exportSvgToPng } from '../utils/exportChart'

const FDistributionPlot = ({ fStatistic, fCritical, alpha, df1, df2 }) => {
  const svgRef = useRef(null)

  if (!fStatistic || !fCritical) return null

  // SVG dimensions
  const width = 700
  const height = 350
  const margin = { top: 30, right: 40, bottom: 60, left: 60 }
  const plotWidth = width - margin.left - margin.right
  const plotHeight = height - margin.top - margin.bottom

  // F-distribution probability density function
  const fPDF = (x, df1, df2) => {
    if (x <= 0) return 0

    const numerator = Math.pow((df1 * x) / df2, df1 / 2) * Math.pow(df2, df2 / 2)
    const denominator = x * beta(df1 / 2, df2 / 2) * Math.pow(df1 * x + df2, (df1 + df2) / 2)

    return numerator / denominator
  }

  // Beta function using gamma function approximation
  const beta = (a, b) => {
    return (gamma(a) * gamma(b)) / gamma(a + b)
  }

  // Stirling's approximation for gamma function
  const gamma = (z) => {
    if (z < 0.5) {
      return Math.PI / (Math.sin(Math.PI * z) * gamma(1 - z))
    }
    z -= 1
    const g = 7
    const C = [
      0.99999999999980993,
      676.5203681218851,
      -1259.1392167224028,
      771.32342877765313,
      -176.61502916214059,
      12.507343278686905,
      -0.13857109526572012,
      9.9843695780195716e-6,
      1.5056327351493116e-7
    ]

    let x = C[0]
    for (let i = 1; i < g + 2; i++) {
      x += C[i] / (z + i)
    }

    const t = z + g + 0.5
    return Math.sqrt(2 * Math.PI) * Math.pow(t, z + 0.5) * Math.exp(-t) * x
  }

  // Calculate range for x-axis
  const maxX = Math.max(fStatistic * 1.2, fCritical * 1.5, 5)
  const minX = 0
  const numPoints = 300

  // Generate F-distribution curve points
  const curvePoints = []
  for (let i = 0; i <= numPoints; i++) {
    const x = minX + (i / numPoints) * maxX
    const y = fPDF(x, df1, df2)
    curvePoints.push({ x, y })
  }

  // Find max y for scaling
  const maxY = Math.max(...curvePoints.map(p => p.y)) * 1.1

  // Scale functions
  const xScale = (x) => (x / maxX) * plotWidth
  const yScale = (y) => plotHeight - (y / maxY) * plotHeight

  // Generate path for the curve
  const curvePath = curvePoints.map((point, i) => {
    const x = xScale(point.x)
    const y = yScale(point.y)
    return `${i === 0 ? 'M' : 'L'} ${x} ${y}`
  }).join(' ')

  // Generate shaded area for rejection region (right tail)
  const rejectionPoints = curvePoints.filter(p => p.x >= fCritical)
  const rejectionPath = [
    `M ${xScale(fCritical)} ${yScale(0)}`,
    ...rejectionPoints.map(p => `L ${xScale(p.x)} ${yScale(p.y)}`),
    `L ${xScale(maxX)} ${yScale(0)}`,
    'Z'
  ].join(' ')

  return (
    <div className="bg-slate-700/50 rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-gray-100 font-semibold">
          F-Distribution (df₁ = {df1}, df₂ = {df2})
        </h4>
        <button
          type="button"
          onClick={() => {
            if (svgRef.current) {
              exportSvgToPng(svgRef.current, `f-distribution-${new Date().toISOString().split('T')[0]}`)
            }
          }}
          className="px-3 py-2 rounded-lg text-sm font-medium bg-slate-600/50 text-gray-300 hover:bg-slate-600 transition-all flex items-center space-x-2"
          title="Export as PNG"
        >
          <Download className="w-4 h-4" />
          <span>Export PNG</span>
        </button>
      </div>

      <div className="flex justify-center">
        <svg ref={svgRef} width={width} height={height} className="overflow-visible">
          <g transform={`translate(${margin.left}, ${margin.top})`}>
            {/* Grid lines */}
            {[0, 0.25, 0.5, 0.75, 1].map((fraction, i) => (
              <line
                key={`grid-${i}`}
                x1={0}
                y1={plotHeight * fraction}
                x2={plotWidth}
                y2={plotHeight * fraction}
                stroke="#475569"
                strokeWidth={1}
                strokeDasharray="4"
                opacity={0.3}
              />
            ))}

            {/* X-axis */}
            <line
              x1={0}
              y1={plotHeight}
              x2={plotWidth}
              y2={plotHeight}
              stroke="#64748b"
              strokeWidth={2}
            />

            {/* Y-axis */}
            <line
              x1={0}
              y1={0}
              x2={0}
              y2={plotHeight}
              stroke="#64748b"
              strokeWidth={2}
            />

            {/* Shaded rejection region */}
            <path
              d={rejectionPath}
              fill="#ef4444"
              opacity={0.3}
            />

            {/* F-distribution curve */}
            <path
              d={curvePath}
              fill="none"
              stroke="#3b82f6"
              strokeWidth={3}
            />

            {/* Critical value line */}
            <line
              x1={xScale(fCritical)}
              y1={0}
              x2={xScale(fCritical)}
              y2={plotHeight}
              stroke="#ef4444"
              strokeWidth={2}
              strokeDasharray="6,3"
            />

            {/* Critical value label */}
            <g transform={`translate(${xScale(fCritical)}, -10)`}>
              <rect
                x={-35}
                y={-18}
                width={70}
                height={20}
                fill="#1e293b"
                stroke="#ef4444"
                strokeWidth={1}
                rx={3}
              />
              <text
                x={0}
                y={-3}
                textAnchor="middle"
                fill="#ef4444"
                fontSize="11"
                fontWeight="600"
              >
                F-crit: {fCritical.toFixed(3)}
              </text>
            </g>

            {/* Observed F-statistic line */}
            {fStatistic > 0 && (
              <>
                <line
                  x1={xScale(fStatistic)}
                  y1={0}
                  x2={xScale(fStatistic)}
                  y2={plotHeight}
                  stroke="#10b981"
                  strokeWidth={3}
                />

                {/* Observed F label */}
                <g transform={`translate(${xScale(fStatistic)}, ${plotHeight + 25})`}>
                  <rect
                    x={-35}
                    y={-10}
                    width={70}
                    height={20}
                    fill="#1e293b"
                    stroke="#10b981"
                    strokeWidth={1}
                    rx={3}
                  />
                  <text
                    x={0}
                    y={5}
                    textAnchor="middle"
                    fill="#10b981"
                    fontSize="11"
                    fontWeight="600"
                  >
                    F: {fStatistic.toFixed(3)}
                  </text>
                </g>
              </>
            )}

            {/* Alpha annotation */}
            <text
              x={xScale(fCritical + (maxX - fCritical) / 2)}
              y={plotHeight / 4}
              textAnchor="middle"
              fill="#ef4444"
              fontSize="14"
              fontWeight="600"
            >
              α = {alpha}
            </text>

            {/* X-axis label */}
            <text
              x={plotWidth / 2}
              y={plotHeight + 45}
              textAnchor="middle"
              fill="#94a3b8"
              fontSize="14"
            >
              F-statistic
            </text>

            {/* Y-axis label */}
            <text
              x={-plotHeight / 2}
              y={-40}
              textAnchor="middle"
              fill="#94a3b8"
              fontSize="14"
              transform={`rotate(-90, -${plotHeight / 2}, -40)`}
            >
              Probability Density
            </text>

            {/* X-axis tick marks */}
            {[0, 1, 2, 3, 4, Math.floor(maxX)].filter(x => x <= maxX).map((tick, i) => (
              <g key={`tick-${i}`}>
                <line
                  x1={xScale(tick)}
                  y1={plotHeight}
                  x2={xScale(tick)}
                  y2={plotHeight + 5}
                  stroke="#64748b"
                  strokeWidth={2}
                />
                <text
                  x={xScale(tick)}
                  y={plotHeight + 20}
                  textAnchor="middle"
                  fill="#94a3b8"
                  fontSize="12"
                >
                  {tick}
                </text>
              </g>
            ))}
          </g>
        </svg>
      </div>

      {/* Legend */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
        <div className="flex items-center space-x-2">
          <div className="w-8 h-1 bg-blue-500"></div>
          <span className="text-gray-300">F-distribution curve</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-8 h-1 bg-red-500 opacity-50"></div>
          <span className="text-gray-300">Rejection region (α = {alpha})</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-8 h-1 bg-green-500"></div>
          <span className="text-gray-300">Observed F-statistic</span>
        </div>
      </div>

      {/* Interpretation */}
      <div className="mt-4 bg-slate-800/50 rounded-lg p-4">
        <p className="text-gray-300 text-sm">
          <strong className="text-gray-100">Interpretation:</strong>{' '}
          {fStatistic > fCritical ? (
            <>
              The observed F-statistic (<span className="text-green-400 font-semibold">{fStatistic.toFixed(3)}</span>)
              falls in the rejection region (beyond the red dashed line at{' '}
              <span className="text-red-400 font-semibold">{fCritical.toFixed(3)}</span>).{' '}
              <span className="text-green-400 font-semibold">Reject the null hypothesis</span> -
              there is significant evidence that at least one group mean differs.
            </>
          ) : (
            <>
              The observed F-statistic (<span className="text-green-400 font-semibold">{fStatistic.toFixed(3)}</span>)
              does not fall in the rejection region (it's before the red dashed line at{' '}
              <span className="text-red-400 font-semibold">{fCritical.toFixed(3)}</span>).{' '}
              <span className="text-blue-400 font-semibold">Fail to reject the null hypothesis</span> -
              insufficient evidence to conclude the group means differ.
            </>
          )}
        </p>
      </div>
    </div>
  )
}

export default FDistributionPlot
