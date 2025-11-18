import { PieChart, Info } from 'lucide-react'
import { useState } from 'react'

const VarianceDecomposition = ({ varianceComponents, iccData }) => {
  const [hoveredSegment, setHoveredSegment] = useState(null)

  if (!varianceComponents || Object.keys(varianceComponents).length === 0) return null

  // Prepare data for visualization
  const prepareData = () => {
    const components = []
    let totalVariance = 0

    // Extract variance from each component
    Object.entries(varianceComponents).forEach(([name, data]) => {
      let variance = 0

      // Handle different variance component structures
      if (typeof data === 'number') {
        variance = data
      } else if (data.variance !== undefined) {
        variance = data.variance
      } else if (data.ms_between !== undefined && data.ms_within !== undefined) {
        // For ICC data structure
        variance = data.ms_between
      }

      if (variance > 0) {
        components.push({ name, variance })
        totalVariance += variance
      }
    })

    // Calculate percentages
    return components.map(comp => ({
      ...comp,
      percentage: (comp.variance / totalVariance) * 100
    })).sort((a, b) => b.percentage - a.percentage)
  }

  const data = prepareData()
  if (data.length === 0) return null

  // Color palette for different components
  const colors = [
    { fill: '#8b5cf6', stroke: '#a78bfa', light: '#c4b5fd' }, // Purple
    { fill: '#3b82f6', stroke: '#60a5fa', light: '#93c5fd' }, // Blue
    { fill: '#06b6d4', stroke: '#22d3ee', light: '#67e8f9' }, // Cyan
    { fill: '#10b981', stroke: '#34d399', light: '#6ee7b7' }, // Green
    { fill: '#f59e0b', stroke: '#fbbf24', light: '#fcd34d' }, // Amber
    { fill: '#ef4444', stroke: '#f87171', light: '#fca5a5' }, // Red
  ]

  // SVG Donut Chart
  const DonutChart = () => {
    const radius = 80
    const innerRadius = 50
    const centerX = 100
    const centerY = 100

    let cumulativePercentage = 0

    const createArc = (startAngle, endAngle, innerR, outerR) => {
      const start = polarToCartesian(centerX, centerY, outerR, endAngle)
      const end = polarToCartesian(centerX, centerY, outerR, startAngle)
      const innerStart = polarToCartesian(centerX, centerY, innerR, endAngle)
      const innerEnd = polarToCartesian(centerX, centerY, innerR, startAngle)

      const largeArcFlag = endAngle - startAngle <= 180 ? '0' : '1'

      return [
        'M', start.x, start.y,
        'A', outerR, outerR, 0, largeArcFlag, 0, end.x, end.y,
        'L', innerEnd.x, innerEnd.y,
        'A', innerR, innerR, 0, largeArcFlag, 1, innerStart.x, innerStart.y,
        'Z'
      ].join(' ')
    }

    const polarToCartesian = (centerX, centerY, radius, angleInDegrees) => {
      const angleInRadians = ((angleInDegrees - 90) * Math.PI) / 180.0
      return {
        x: centerX + radius * Math.cos(angleInRadians),
        y: centerY + radius * Math.sin(angleInRadians)
      }
    }

    return (
      <div className="flex justify-center my-6">
        <svg width="300" height="300" viewBox="0 0 200 200" className="drop-shadow-lg">
          {/* Segments */}
          {data.map((item, index) => {
            const startAngle = (cumulativePercentage / 100) * 360
            cumulativePercentage += item.percentage
            const endAngle = (cumulativePercentage / 100) * 360

            const isHovered = hoveredSegment === index
            const color = colors[index % colors.length]
            const outerR = isHovered ? radius + 5 : radius

            const path = createArc(startAngle, endAngle, innerRadius, outerR)

            // Calculate label position (midpoint of arc)
            const midAngle = (startAngle + endAngle) / 2
            const labelRadius = (innerRadius + outerR) / 2
            const labelPos = polarToCartesian(centerX, centerY, labelRadius, midAngle)

            return (
              <g key={index}>
                <path
                  d={path}
                  fill={color.fill}
                  stroke={color.stroke}
                  strokeWidth="2"
                  className="transition-all duration-300 cursor-pointer"
                  onMouseEnter={() => setHoveredSegment(index)}
                  onMouseLeave={() => setHoveredSegment(null)}
                  opacity={hoveredSegment !== null && hoveredSegment !== index ? 0.4 : 1}
                />
                {/* Percentage label */}
                {item.percentage > 5 && (
                  <text
                    x={labelPos.x}
                    y={labelPos.y}
                    fill="white"
                    fontSize="10"
                    fontWeight="bold"
                    textAnchor="middle"
                    dominantBaseline="middle"
                    className="pointer-events-none"
                  >
                    {item.percentage.toFixed(1)}%
                  </text>
                )}
              </g>
            )
          })}

          {/* Center circle */}
          <circle
            cx={centerX}
            cy={centerY}
            r={innerRadius}
            fill="#1e293b"
            stroke="#334155"
            strokeWidth="2"
          />

          {/* Center text */}
          <text
            x={centerX}
            y={centerY - 5}
            fill="#cbd5e1"
            fontSize="12"
            fontWeight="bold"
            textAnchor="middle"
          >
            Variance
          </text>
          <text
            x={centerX}
            y={centerY + 10}
            fill="#94a3b8"
            fontSize="10"
            textAnchor="middle"
          >
            Components
          </text>
        </svg>
      </div>
    )
  }

  return (
    <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
      <div className="flex items-center space-x-3 mb-4">
        <PieChart className="w-6 h-6 text-cyan-400" />
        <h3 className="text-xl font-bold text-gray-100">Variance Decomposition</h3>
      </div>

      <p className="text-gray-400 text-sm mb-6">
        Breakdown of total variance by source. This shows how much variability is attributed to each factor.
      </p>

      {/* Donut Chart */}
      <DonutChart />

      {/* Legend */}
      <div className="space-y-3">
        {data.map((item, index) => {
          const color = colors[index % colors.length]
          const isHovered = hoveredSegment === index

          return (
            <div
              key={index}
              className={`flex items-center justify-between p-3 rounded-lg transition-all duration-200 cursor-pointer ${
                isHovered ? 'bg-slate-700/50 scale-105' : 'bg-slate-700/20'
              }`}
              onMouseEnter={() => setHoveredSegment(index)}
              onMouseLeave={() => setHoveredSegment(null)}
            >
              <div className="flex items-center space-x-3">
                <div
                  className="w-4 h-4 rounded"
                  style={{ backgroundColor: color.fill }}
                />
                <span className="text-sm font-medium text-gray-200">{item.name}</span>
              </div>
              <div className="flex items-center space-x-4">
                <span className="text-sm font-mono text-gray-400">
                  {item.variance.toFixed(4)}
                </span>
                <span className={`text-sm font-bold ${isHovered ? 'text-cyan-400' : 'text-gray-300'}`}>
                  {item.percentage.toFixed(1)}%
                </span>
              </div>
            </div>
          )
        })}
      </div>

      {/* Summary statistics */}
      <div className="mt-6 grid grid-cols-2 gap-4 p-4 bg-slate-700/30 rounded-lg">
        <div className="text-center">
          <div className="text-xs text-gray-400 mb-1">Total Variance</div>
          <div className="text-sm font-mono text-gray-200">
            {data.reduce((sum, item) => sum + item.variance, 0).toFixed(4)}
          </div>
        </div>
        <div className="text-center">
          <div className="text-xs text-gray-400 mb-1">Components</div>
          <div className="text-sm font-mono text-gray-200">{data.length}</div>
        </div>
      </div>

      {/* ICC Integration */}
      {iccData && Object.keys(iccData).length > 0 && (
        <div className="mt-6 bg-purple-500/10 border border-purple-500/50 rounded-lg p-4">
          <h5 className="font-semibold text-purple-300 mb-2 flex items-center">
            <Info className="w-4 h-4 mr-2" />
            Related ICC Values
          </h5>
          <div className="space-y-2">
            {Object.entries(iccData).map(([factor, iccResult]) => {
              // Handle both number format (nested designs) and object format (other designs)
              if (iccResult?.error) return null

              const iccValue = typeof iccResult === 'number' ? iccResult : iccResult?.icc
              if (iccValue === undefined || iccValue === null) return null

              return (
                <div key={factor} className="flex justify-between text-sm">
                  <span className="text-gray-300">{factor}:</span>
                  <span className="font-mono text-purple-300">
                    ICC = {iccValue.toFixed(3)}
                  </span>
                </div>
              )
            })}
          </div>
          <p className="text-xs text-gray-400 mt-3">
            ICC values show the proportion of total variance attributable to each clustering factor.
          </p>
        </div>
      )}

      {/* Interpretation note */}
      <div className="mt-4 bg-slate-700/30 rounded-lg p-3">
        <p className="text-xs text-gray-400">
          <strong>Interpretation:</strong> Larger segments indicate factors that explain more of the total variance.
          This decomposition helps identify which sources of variation are most important in your model.
        </p>
      </div>
    </div>
  )
}

export default VarianceDecomposition
