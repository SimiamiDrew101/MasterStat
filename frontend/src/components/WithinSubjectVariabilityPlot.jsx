import { useRef } from 'react'
import { Download } from 'lucide-react'
import { exportSvgToPng } from '../utils/exportChart'

const WithinSubjectVariabilityPlot = ({ trajectories, profileData, withinFactor, responseName }) => {
  const svgRef = useRef(null)

  if (!trajectories || trajectories.length === 0 || !profileData || profileData.length === 0) return null

  // SVG dimensions
  const width = 800
  const height = 500
  const margin = { top: 40, right: 100, bottom: 80, left: 80 }
  const plotWidth = width - margin.left - margin.right
  const plotHeight = height - margin.top - margin.bottom

  // Find min/max for y-axis from all individual values
  const allValues = trajectories.flatMap(t => t.values)
  const yMin = Math.min(...allValues)
  const yMax = Math.max(...allValues)
  const yRange = yMax - yMin
  const yPadding = yRange * 0.15
  const chartMin = yMin - yPadding
  const chartMax = yMax + yPadding
  const chartRange = chartMax - chartMin

  // Scale functions
  const xScale = (index) => {
    return (plotWidth / (profileData.length - 1)) * index
  }

  const yScale = (value) => {
    return plotHeight - ((value - chartMin) / chartRange) * plotHeight
  }

  // Colors for different subjects
  const subjectColors = [
    '#3b82f6', // Blue
    '#10b981', // Green
    '#f59e0b', // Amber
    '#ef4444', // Red
    '#8b5cf6', // Purple
    '#06b6d4', // Cyan
    '#ec4899', // Pink
    '#14b8a6', // Teal
  ]

  // Generate overall mean path
  const meanPath = profileData
    .map((d, i) => {
      const x = xScale(i)
      const y = yScale(d.mean)
      return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`
    })
    .join(' ')

  return (
    <div className="bg-slate-700/50 rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-gray-100 font-semibold">Individual Subject Trajectories</h4>
        <button
          type="button"
          onClick={() => {
            if (svgRef.current) {
              exportSvgToPng(svgRef.current, `trajectories-plot-${new Date().toISOString().split('T')[0]}`)
            }
          }}
          className="px-3 py-2 rounded-lg text-sm font-medium bg-slate-600/50 text-gray-300 hover:bg-slate-600 transition-all flex items-center space-x-2"
          title="Export as PNG"
        >
          <Download className="w-4 h-4" />
          <span>Export PNG</span>
        </button>
      </div>
      <p className="text-gray-300 text-sm mb-4">
        Shows individual subject responses across {withinFactor} conditions. Parallel lines suggest
        consistent within-subject effects. The thick black line shows the overall mean.
      </p>

      <div className="flex justify-center">
        <svg ref={svgRef} width={width} height={height}>
          <g transform={`translate(${margin.left}, ${margin.top})`}>
            {/* Grid lines */}
            {[0, 0.25, 0.5, 0.75, 1].map((fraction, i) => {
              const y = plotHeight * fraction
              const value = chartMax - fraction * chartRange
              return (
                <g key={i}>
                  <line
                    x1={0}
                    y1={y}
                    x2={plotWidth}
                    y2={y}
                    stroke="#475569"
                    strokeWidth={1}
                    strokeDasharray="4"
                  />
                  <text
                    x={-10}
                    y={y + 4}
                    textAnchor="end"
                    fill="#94a3b8"
                    fontSize="12"
                  >
                    {value.toFixed(1)}
                  </text>
                </g>
              )
            })}

            {/* Axes */}
            <line
              x1={0}
              y1={plotHeight}
              x2={plotWidth}
              y2={plotHeight}
              stroke="#64748b"
              strokeWidth={2}
            />
            <line
              x1={0}
              y1={0}
              x2={0}
              y2={plotHeight}
              stroke="#64748b"
              strokeWidth={2}
            />

            {/* Individual subject trajectories */}
            {trajectories.map((trajectory, tIdx) => {
              const color = subjectColors[tIdx % subjectColors.length]

              // Generate path for this subject
              const path = trajectory.values
                .map((value, i) => {
                  const x = xScale(i)
                  const y = yScale(value)
                  return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`
                })
                .join(' ')

              return (
                <g key={tIdx}>
                  {/* Subject line */}
                  <path
                    d={path}
                    fill="none"
                    stroke={color}
                    strokeWidth={2}
                    opacity={0.6}
                  />

                  {/* Subject points */}
                  {trajectory.values.map((value, i) => {
                    const x = xScale(i)
                    const y = yScale(value)
                    return (
                      <circle
                        key={i}
                        cx={x}
                        cy={y}
                        r={4}
                        fill={color}
                        opacity={0.7}
                      />
                    )
                  })}
                </g>
              )
            })}

            {/* Overall mean line (thick black) */}
            <path
              d={meanPath}
              fill="none"
              stroke="#1e293b"
              strokeWidth={4}
            />
            <path
              d={meanPath}
              fill="none"
              stroke="#f1f5f9"
              strokeWidth={2}
            />

            {/* Mean points */}
            {profileData.map((d, i) => {
              const x = xScale(i)
              const y = yScale(d.mean)
              return (
                <circle
                  key={i}
                  cx={x}
                  cy={y}
                  r={6}
                  fill="#1e293b"
                  stroke="#f1f5f9"
                  strokeWidth={2}
                />
              )
            })}

            {/* Condition labels */}
            {profileData.map((d, i) => {
              const x = xScale(i)
              return (
                <text
                  key={i}
                  x={x}
                  y={plotHeight + 25}
                  textAnchor="middle"
                  fill="#e2e8f0"
                  fontSize="14"
                  fontWeight="600"
                >
                  {d.condition}
                </text>
              )
            })}

            {/* Axis labels */}
            <text
              x={plotWidth / 2}
              y={plotHeight + 55}
              textAnchor="middle"
              fill="#94a3b8"
              fontSize="14"
              fontWeight="600"
            >
              {withinFactor}
            </text>
            <text
              x={-plotHeight / 2}
              y={-50}
              textAnchor="middle"
              fill="#94a3b8"
              fontSize="14"
              fontWeight="600"
              transform={`rotate(-90, -${plotHeight / 2}, -50)`}
            >
              {responseName}
            </text>

            {/* Legend */}
            <g transform={`translate(${plotWidth + 10}, 20)`}>
              <rect x={0} y={0} width={80} height={Math.min(40 + trajectories.length * 20, 200)} fill="#1e293b" rx={4} />

              {/* Overall mean */}
              <line
                x1={10}
                y1={20}
                x2={30}
                y2={20}
                stroke="#1e293b"
                strokeWidth={4}
              />
              <line
                x1={10}
                y1={20}
                x2={30}
                y2={20}
                stroke="#f1f5f9"
                strokeWidth={2}
              />
              <text x={35} y={24} fill="#e2e8f0" fontSize="10" fontWeight="600">
                Mean
              </text>

              {/* Individual subjects */}
              {trajectories.slice(0, 6).map((trajectory, i) => {
                const color = subjectColors[i % subjectColors.length]
                const y = 40 + i * 20
                return (
                  <g key={i}>
                    <line
                      x1={10}
                      y1={y}
                      x2={30}
                      y2={y}
                      stroke={color}
                      strokeWidth={2}
                      opacity={0.6}
                    />
                    <text x={35} y={y + 4} fill="#e2e8f0" fontSize="9">
                      {trajectory.subject}
                    </text>
                  </g>
                )
              })}

              {trajectories.length > 6 && (
                <text x={35} y={40 + 6 * 20 + 4} fill="#94a3b8" fontSize="9" fontStyle="italic">
                  +{trajectories.length - 6} more
                </text>
              )}
            </g>
          </g>
        </svg>
      </div>

      <div className="mt-4 bg-slate-800/50 rounded-lg p-4">
        <p className="text-gray-300 text-sm">
          <strong className="text-gray-100">Interpretation:</strong> Each colored line represents one subject's
          trajectory across {withinFactor} conditions. Parallel lines suggest subjects respond similarly to the
          within-subjects factor. The thick line shows the overall mean response. Greater variability between
          subjects indicates individual differences in baseline levels or response patterns.
        </p>
      </div>
    </div>
  )
}

export default WithinSubjectVariabilityPlot
