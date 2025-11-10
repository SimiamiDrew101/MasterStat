import { useRef } from 'react'
import { Download } from 'lucide-react'
import { exportSvgToPng } from '../utils/exportChart'

const ProfilePlot = ({ profileData, withinFactor, responseName }) => {
  const svgRef = useRef(null)

  if (!profileData || profileData.length === 0) return null

  // SVG dimensions
  const width = 800
  const height = 500
  const margin = { top: 40, right: 100, bottom: 80, left: 80 }
  const plotWidth = width - margin.left - margin.right
  const plotHeight = height - margin.top - margin.bottom

  // Find min/max for y-axis
  const allValues = profileData.flatMap(d => [
    d.mean - d.sem,
    d.mean + d.sem
  ])
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

  // Generate path for line
  const linePath = profileData
    .map((d, i) => {
      const x = xScale(i)
      const y = yScale(d.mean)
      return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`
    })
    .join(' ')

  return (
    <div className="bg-slate-700/50 rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-gray-100 font-semibold">Profile Plot</h4>
        <button
          type="button"
          onClick={() => {
            if (svgRef.current) {
              exportSvgToPng(svgRef.current, `profile-plot-${new Date().toISOString().split('T')[0]}`)
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
        Shows mean {responseName} across {withinFactor} conditions with standard error bars.
        A clear trend indicates a strong within-subjects effect.
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

            {/* Line connecting means */}
            <path
              d={linePath}
              fill="none"
              stroke="#8b5cf6"
              strokeWidth={3}
            />

            {/* Error bars and points */}
            {profileData.map((d, i) => {
              const x = xScale(i)
              const yMean = yScale(d.mean)
              const yLower = yScale(d.mean - d.sem)
              const yUpper = yScale(d.mean + d.sem)

              return (
                <g key={i}>
                  {/* Error bar */}
                  <line
                    x1={x}
                    y1={yLower}
                    x2={x}
                    y2={yUpper}
                    stroke="#a78bfa"
                    strokeWidth={2}
                  />
                  {/* Error bar caps */}
                  <line
                    x1={x - 6}
                    y1={yLower}
                    x2={x + 6}
                    y2={yLower}
                    stroke="#a78bfa"
                    strokeWidth={2}
                  />
                  <line
                    x1={x - 6}
                    y1={yUpper}
                    x2={x + 6}
                    y2={yUpper}
                    stroke="#a78bfa"
                    strokeWidth={2}
                  />

                  {/* Mean point */}
                  <circle
                    cx={x}
                    cy={yMean}
                    r={6}
                    fill="#8b5cf6"
                    stroke="#f1f5f9"
                    strokeWidth={2}
                  />

                  {/* Condition label */}
                  <text
                    x={x}
                    y={plotHeight + 25}
                    textAnchor="middle"
                    fill="#e2e8f0"
                    fontSize="14"
                    fontWeight="600"
                  >
                    {d.condition}
                  </text>

                  {/* Mean value label */}
                  <text
                    x={x}
                    y={yMean - 15}
                    textAnchor="middle"
                    fill="#c4b5fd"
                    fontSize="11"
                    fontWeight="600"
                  >
                    {d.mean.toFixed(1)}
                  </text>
                </g>
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
              Mean {responseName}
            </text>

            {/* Legend */}
            <g transform={`translate(${plotWidth + 20}, 20)`}>
              <rect x={0} y={0} width={70} height={60} fill="#1e293b" rx={4} />
              <line
                x1={10}
                y1={20}
                x2={30}
                y2={20}
                stroke="#8b5cf6"
                strokeWidth={3}
              />
              <circle cx={20} cy={20} r={4} fill="#8b5cf6" stroke="#f1f5f9" strokeWidth={1} />
              <text x={35} y={24} fill="#e2e8f0" fontSize="11">
                Mean
              </text>
              <line
                x1={10}
                y1={45}
                x2={30}
                y2={45}
                stroke="#a78bfa"
                strokeWidth={2}
              />
              <line x1={15} y1={40} x2={15} y2={50} stroke="#a78bfa" strokeWidth={2} />
              <line x1={25} y1={40} x2={25} y2={50} stroke="#a78bfa" strokeWidth={2} />
              <text x={35} y={49} fill="#e2e8f0" fontSize="11">
                ±SEM
              </text>
            </g>
          </g>
        </svg>
      </div>

      <div className="mt-4 bg-slate-800/50 rounded-lg p-4">
        <p className="text-gray-300 text-sm">
          <strong className="text-gray-100">Interpretation:</strong> This profile plot shows how the mean response
          changes across {withinFactor} conditions. The purple line connects the mean values, and error bars represent
          standard error of the mean (SEM). A clear upward or downward trend suggests a systematic within-subjects effect.
        </p>
      </div>
    </div>
  )
}

export default ProfilePlot
