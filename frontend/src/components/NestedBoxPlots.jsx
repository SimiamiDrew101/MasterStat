import { useRef } from 'react'
import { Download } from 'lucide-react'
import { exportSvgToPng } from '../utils/exportChart'

const NestedBoxPlots = ({ boxPlotDataNested, factorA }) => {
  const svgRef = useRef(null)

  if (!boxPlotDataNested || Object.keys(boxPlotDataNested).length === 0) return null

  // Helper function to calculate box plot statistics
  const calculateBoxStats = (values) => {
    const cleanValues = values.filter(v => v !== null).sort((a, b) => a - b)
    if (cleanValues.length === 0) return null

    const q1Index = Math.floor(cleanValues.length * 0.25)
    const medianIndex = Math.floor(cleanValues.length * 0.5)
    const q3Index = Math.floor(cleanValues.length * 0.75)

    const q1 = cleanValues[q1Index]
    const median = cleanValues[medianIndex]
    const q3 = cleanValues[q3Index]
    const iqr = q3 - q1

    const lowerFence = q1 - 1.5 * iqr
    const upperFence = q3 + 1.5 * iqr

    const outliers = cleanValues.filter(v => v < lowerFence || v > upperFence)
    const nonOutliers = cleanValues.filter(v => v >= lowerFence && v <= upperFence)

    return {
      min: nonOutliers.length > 0 ? Math.min(...nonOutliers) : cleanValues[0],
      q1,
      median,
      q3,
      max: nonOutliers.length > 0 ? Math.max(...nonOutliers) : cleanValues[cleanValues.length - 1],
      outliers
    }
  }

  // Process data
  const groups = Object.entries(boxPlotDataNested).map(([groupName, boxData]) => ({
    name: groupName,
    boxes: boxData.map(d => ({
      label: d.level,
      stats: calculateBoxStats(d.values),
      values: d.values
    })).filter(d => d.stats !== null)
  }))

  if (groups.length === 0) return null

  // Find global min/max for scaling
  const allValues = groups.flatMap(g =>
    g.boxes.flatMap(b => [b.stats.min, b.stats.max, ...b.stats.outliers])
  )
  const globalMin = Math.min(...allValues)
  const globalMax = Math.max(...allValues)
  const range = globalMax - globalMin
  const padding = range * 0.1
  const chartMin = globalMin - padding
  const chartMax = globalMax + padding
  const chartRange = chartMax - chartMin

  // SVG dimensions
  const width = 800
  const height = 400
  const margin = { top: 40, right: 40, bottom: 80, left: 60 }
  const plotWidth = width - margin.left - margin.right
  const plotHeight = height - margin.top - margin.bottom

  // Scale function
  const yScale = (value) => {
    return plotHeight - ((value - chartMin) / chartRange) * plotHeight
  }

  // Colors for groups
  const groupColors = [
    '#8b5cf6', // Purple
    '#3b82f6', // Blue
    '#10b981', // Green
    '#f59e0b', // Amber
    '#ef4444', // Red
    '#06b6d4', // Cyan
  ]

  // Calculate layout
  const totalBoxes = groups.reduce((sum, g) => sum + g.boxes.length, 0)
  const boxWidth = Math.min(40, plotWidth / (totalBoxes * 1.5))
  const groupSpacing = plotWidth / groups.length

  return (
    <div className="bg-slate-700/50 rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-gray-100 font-semibold">Nested Box Plots</h4>
        <button
          type="button"
          onClick={() => {
            if (svgRef.current) {
              exportSvgToPng(svgRef.current, `nested-boxplots-${new Date().toISOString().split('T')[0]}`)
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
        Box plots grouped by {factorA}. Each group shows the distribution of nested factors.
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
                    fontSize="11"
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

            {/* Box plots */}
            {groups.map((group, groupIndex) => {
              const groupX = groupSpacing * groupIndex + groupSpacing / 2
              const numBoxes = group.boxes.length
              const boxesWidth = boxWidth * numBoxes + 10 * (numBoxes - 1)
              const startX = groupX - boxesWidth / 2
              const color = groupColors[groupIndex % groupColors.length]

              return (
                <g key={groupIndex}>
                  {/* Group background */}
                  <rect
                    x={groupX - groupSpacing * 0.45}
                    y={0}
                    width={groupSpacing * 0.9}
                    height={plotHeight}
                    fill={color}
                    opacity={0.05}
                  />

                  {/* Group label */}
                  <text
                    x={groupX}
                    y={plotHeight + 25}
                    textAnchor="middle"
                    fill="#e2e8f0"
                    fontSize="14"
                    fontWeight="600"
                  >
                    {group.name}
                  </text>

                  {/* Individual box plots */}
                  {group.boxes.map((box, boxIndex) => {
                    const x = startX + boxIndex * (boxWidth + 10) + boxWidth / 2
                    const stats = box.stats

                    return (
                      <g key={boxIndex}>
                        {/* Whiskers */}
                        <line
                          x1={x}
                          y1={yScale(stats.min)}
                          x2={x}
                          y2={yScale(stats.max)}
                          stroke={color}
                          strokeWidth={2}
                        />

                        {/* Min cap */}
                        <line
                          x1={x - boxWidth / 4}
                          y1={yScale(stats.min)}
                          x2={x + boxWidth / 4}
                          y2={yScale(stats.min)}
                          stroke={color}
                          strokeWidth={2}
                        />

                        {/* Max cap */}
                        <line
                          x1={x - boxWidth / 4}
                          y1={yScale(stats.max)}
                          x2={x + boxWidth / 4}
                          y2={yScale(stats.max)}
                          stroke={color}
                          strokeWidth={2}
                        />

                        {/* Box */}
                        <rect
                          x={x - boxWidth / 2}
                          y={yScale(stats.q3)}
                          width={boxWidth}
                          height={yScale(stats.q1) - yScale(stats.q3)}
                          fill={color}
                          fillOpacity={0.6}
                          stroke={color}
                          strokeWidth={2}
                        />

                        {/* Median line */}
                        <line
                          x1={x - boxWidth / 2}
                          y1={yScale(stats.median)}
                          x2={x + boxWidth / 2}
                          y2={yScale(stats.median)}
                          stroke="#f1f5f9"
                          strokeWidth={3}
                        />

                        {/* Outliers */}
                        {stats.outliers.map((outlier, outIndex) => (
                          <circle
                            key={outIndex}
                            cx={x}
                            cy={yScale(outlier)}
                            r={3}
                            fill="#ef4444"
                            opacity={0.7}
                          />
                        ))}

                        {/* Label */}
                        <text
                          x={x}
                          y={plotHeight + 45}
                          textAnchor="middle"
                          fill="#cbd5e1"
                          fontSize="10"
                        >
                          {box.label}
                        </text>
                      </g>
                    )
                  })}
                </g>
              )
            })}

            {/* Y-axis label */}
            <text
              x={-35}
              y={plotHeight / 2}
              textAnchor="middle"
              fill="#94a3b8"
              fontSize="12"
              transform={`rotate(-90, -35, ${plotHeight / 2})`}
            >
              Response Value
            </text>
          </g>
        </svg>
      </div>

      <div className="mt-4 bg-slate-800/50 rounded-lg p-4">
        <p className="text-gray-300 text-sm">
          <strong className="text-gray-100">Interpretation:</strong> Each shaded region represents a {factorA} group.
          Box plots within each region show the distribution for nested factors.
          Comparing box positions within a group shows within-group variation, while comparing across groups shows between-group variation.
        </p>
      </div>
    </div>
  )
}

export default NestedBoxPlots
