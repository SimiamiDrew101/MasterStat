import { useRef } from 'react'
import { Download } from 'lucide-react'
import { exportSvgToPng } from '../utils/exportChart'

const VarianceComponentsChart = ({ variancePercentages, title = "Variance Components" }) => {
  const svgRef = useRef(null)

  if (!variancePercentages || Object.keys(variancePercentages).length === 0) return null

  // Prepare data
  const components = Object.entries(variancePercentages)
    .filter(([_, value]) => value > 0)
    .sort((a, b) => b[1] - a[1]) // Sort by percentage descending

  if (components.length === 0) return null

  // SVG dimensions
  const width = 500
  const height = 300
  const centerX = 200
  const centerY = 150
  const radius = 100

  // Colors for different components
  const colors = [
    '#8b5cf6', // Purple - highest level
    '#3b82f6', // Blue - middle level
    '#10b981', // Green - error
    '#f59e0b', // Amber - additional
    '#ef4444'  // Red - additional
  ]

  // Calculate angles
  let currentAngle = -90 // Start from top
  const slices = components.map(([label, percentage], index) => {
    const angle = (percentage / 100) * 360
    const startAngle = currentAngle
    const endAngle = currentAngle + angle
    currentAngle = endAngle

    // Calculate path for pie slice
    const startRad = (startAngle * Math.PI) / 180
    const endRad = (endAngle * Math.PI) / 180

    const x1 = centerX + radius * Math.cos(startRad)
    const y1 = centerY + radius * Math.sin(startRad)
    const x2 = centerX + radius * Math.cos(endRad)
    const y2 = centerY + radius * Math.sin(endRad)

    const largeArcFlag = angle > 180 ? 1 : 0

    const path = [
      `M ${centerX} ${centerY}`,
      `L ${x1} ${y1}`,
      `A ${radius} ${radius} 0 ${largeArcFlag} 1 ${x2} ${y2}`,
      'Z'
    ].join(' ')

    // Calculate label position
    const midAngle = (startAngle + endAngle) / 2
    const midRad = (midAngle * Math.PI) / 180
    const labelRadius = radius * 0.65
    const labelX = centerX + labelRadius * Math.cos(midRad)
    const labelY = centerY + labelRadius * Math.sin(midRad)

    return {
      label,
      percentage,
      path,
      color: colors[index % colors.length],
      labelX,
      labelY
    }
  })

  return (
    <div className="bg-slate-700/50 rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-gray-100 font-semibold">{title}</h4>
        <button
          type="button"
          onClick={() => {
            if (svgRef.current) {
              exportSvgToPng(svgRef.current, `variance-components-${new Date().toISOString().split('T')[0]}`)
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
        Shows the relative contribution of each source to total variance.
      </p>

      <div className="flex justify-center items-center">
        <svg ref={svgRef} width={width} height={height}>
          {/* Pie slices */}
          {slices.map((slice, index) => (
            <g key={index}>
              <path
                d={slice.path}
                fill={slice.color}
                stroke="#1e293b"
                strokeWidth={2}
                opacity={0.9}
              />
              {/* Percentage label on slice */}
              {slice.percentage >= 5 && (
                <text
                  x={slice.labelX}
                  y={slice.labelY}
                  textAnchor="middle"
                  fill="white"
                  fontSize="14"
                  fontWeight="bold"
                >
                  {slice.percentage.toFixed(1)}%
                </text>
              )}
            </g>
          ))}

          {/* Center circle for donut effect */}
          <circle
            cx={centerX}
            cy={centerY}
            r={radius * 0.4}
            fill="#1e293b"
            stroke="#334155"
            strokeWidth={2}
          />
          <text
            x={centerX}
            y={centerY}
            textAnchor="middle"
            fill="#e2e8f0"
            fontSize="12"
            fontWeight="600"
          >
            Total
          </text>
          <text
            x={centerX}
            y={centerY + 16}
            textAnchor="middle"
            fill="#e2e8f0"
            fontSize="12"
          >
            Variance
          </text>
        </svg>

        {/* Legend */}
        <div className="ml-8 space-y-2">
          {slices.map((slice, index) => (
            <div key={index} className="flex items-center space-x-2">
              <div
                className="w-4 h-4 rounded"
                style={{ backgroundColor: slice.color }}
              />
              <div className="text-sm">
                <div className="text-gray-200 font-medium">
                  {slice.label.replace('σ²_', '')}
                </div>
                <div className="text-gray-400 text-xs">
                  {slice.percentage.toFixed(1)}%
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="mt-4 bg-slate-800/50 rounded-lg p-4">
        <p className="text-gray-300 text-sm">
          <strong className="text-gray-100">Interpretation:</strong> Larger slices indicate sources that contribute more to the total variability in the response.
          In nested designs, understanding where most variation comes from helps identify which level (e.g., between schools vs. between teachers)
          is the primary source of differences.
        </p>
      </div>
    </div>
  )
}

export default VarianceComponentsChart
