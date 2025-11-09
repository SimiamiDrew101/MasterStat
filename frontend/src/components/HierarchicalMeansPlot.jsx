import { useRef } from 'react'
import { Download } from 'lucide-react'
import { exportSvgToPng } from '../utils/exportChart'

const HierarchicalMeansPlot = ({ marginalMeansA, nestedMeans, factorA, factorB }) => {
  const svgRef = useRef(null)

  if (!marginalMeansA || !nestedMeans || marginalMeansA.length === 0 || nestedMeans.length === 0) {
    return null
  }

  // SVG dimensions
  const width = 700
  const height = 400
  const margin = { top: 40, right: 120, bottom: 60, left: 60 }
  const plotWidth = width - margin.left - margin.right
  const plotHeight = height - margin.top - margin.bottom

  // Colors for each group
  const groupColors = [
    '#8b5cf6', // Purple
    '#3b82f6', // Blue
    '#10b981', // Green
    '#f59e0b', // Amber
    '#ef4444', // Red
    '#06b6d4', // Cyan
  ]

  // Calculate scales
  const allMeans = [...marginalMeansA.map(d => d.mean), ...nestedMeans.map(d => d.mean)]
  const minMean = Math.min(...allMeans)
  const maxMean = Math.max(...allMeans)
  const range = maxMean - minMean
  const padding = range * 0.15
  const yMin = minMean - padding
  const yMax = maxMean + padding
  const yRange = yMax - yMin

  const yScale = (value) => plotHeight - ((value - yMin) / yRange) * plotHeight

  // X positions for groups
  const numGroups = marginalMeansA.length
  const groupSpacing = plotWidth / numGroups

  return (
    <div className="bg-slate-700/50 rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-gray-100 font-semibold">Hierarchical Means Plot</h4>
        <button
          type="button"
          onClick={() => {
            if (svgRef.current) {
              exportSvgToPng(svgRef.current, `hierarchical-means-${new Date().toISOString().split('T')[0]}`)
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
        Shows means at both hierarchical levels: {factorA} (large markers) and {factorB} nested within each {factorA} (small markers).
      </p>

      <div className="flex justify-center">
        <svg ref={svgRef} width={width} height={height}>
          <g transform={`translate(${margin.left}, ${margin.top})`}>
            {/* Grid lines */}
            {[0, 0.25, 0.5, 0.75, 1].map((fraction, i) => (
              <line
                key={i}
                x1={0}
                y1={plotHeight * fraction}
                x2={plotWidth}
                y2={plotHeight * fraction}
                stroke="#475569"
                strokeWidth={1}
                strokeDasharray="2"
                opacity={0.3}
              />
            ))}

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

            {/* Y-axis labels */}
            {[0, 0.25, 0.5, 0.75, 1].map((fraction, i) => {
              const value = yMax - fraction * yRange
              return (
                <text
                  key={i}
                  x={-10}
                  y={plotHeight * fraction + 4}
                  textAnchor="end"
                  fill="#94a3b8"
                  fontSize="11"
                >
                  {value.toFixed(1)}
                </text>
              )
            })}

            {/* Plot nested means and marginal means */}
            {marginalMeansA.map((groupData, groupIndex) => {
              const groupX = groupSpacing * groupIndex + groupSpacing / 2
              const color = groupColors[groupIndex % groupColors.length]

              // Get nested items for this group
              const nestedItems = nestedMeans.filter(item => item[factorA] === groupData.level)
              const numNested = nestedItems.length
              const nestedSpacing = Math.min(groupSpacing * 0.7, 80) / Math.max(numNested - 1, 1)
              const nestedStartX = groupX - (nestedSpacing * (numNested - 1)) / 2

              return (
                <g key={groupIndex}>
                  {/* Connect nested means with lines */}
                  {nestedItems.length > 1 && (
                    <line
                      x1={nestedStartX}
                      y1={yScale(nestedItems[0].mean)}
                      x2={nestedStartX + nestedSpacing * (numNested - 1)}
                      y2={yScale(nestedItems[nestedItems.length - 1].mean)}
                      stroke={color}
                      strokeWidth={1.5}
                      opacity={0.3}
                      strokeDasharray="3"
                    />
                  )}

                  {/* Nested means (small circles) */}
                  {nestedItems.map((item, nestedIndex) => {
                    const nestedX = nestedStartX + nestedIndex * nestedSpacing
                    return (
                      <g key={nestedIndex}>
                        <circle
                          cx={nestedX}
                          cy={yScale(item.mean)}
                          r={4}
                          fill={color}
                          opacity={0.7}
                          stroke={color}
                          strokeWidth={1.5}
                        />
                        {/* Label */}
                        <text
                          x={nestedX}
                          y={yScale(item.mean) - 10}
                          textAnchor="middle"
                          fill="#e2e8f0"
                          fontSize="9"
                        >
                          {item[factorB]}
                        </text>
                      </g>
                    )
                  })}

                  {/* Marginal mean (large circle) */}
                  <circle
                    cx={groupX}
                    cy={yScale(groupData.mean)}
                    r={8}
                    fill={color}
                    stroke="#1e293b"
                    strokeWidth={3}
                  />

                  {/* Value label */}
                  <text
                    x={groupX}
                    y={yScale(groupData.mean) - 16}
                    textAnchor="middle"
                    fill="#e2e8f0"
                    fontSize="12"
                    fontWeight="600"
                  >
                    {groupData.mean.toFixed(1)}
                  </text>

                  {/* Group label */}
                  <text
                    x={groupX}
                    y={plotHeight + 20}
                    textAnchor="middle"
                    fill="#e2e8f0"
                    fontSize="13"
                    fontWeight="600"
                  >
                    {groupData.level}
                  </text>
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
              Mean Response
            </text>

            {/* X-axis label */}
            <text
              x={plotWidth / 2}
              y={plotHeight + 45}
              textAnchor="middle"
              fill="#94a3b8"
              fontSize="12"
            >
              {factorA}
            </text>

            {/* Title */}
            <text
              x={plotWidth / 2}
              y={-15}
              textAnchor="middle"
              fill="#e2e8f0"
              fontSize="14"
              fontWeight="600"
            >
              {factorA} Marginal Means and {factorB}({factorA}) Nested Means
            </text>
          </g>

          {/* Legend */}
          <g transform={`translate(${width - margin.right + 10}, ${margin.top})`}>
            <rect x={0} y={0} width={100} height={60} fill="#1e293b" stroke="#475569" strokeWidth={1} rx={4} />

            <circle cx={15} cy={20} r={8} fill="#8b5cf6" stroke="#1e293b" strokeWidth={2} />
            <text x={30} y={24} fill="#e2e8f0" fontSize="11">{factorA} Mean</text>

            <circle cx={15} cy={40} r={4} fill="#8b5cf6" opacity={0.7} />
            <text x={30} y={44} fill="#e2e8f0" fontSize="11">{factorB} Mean</text>
          </g>
        </svg>
      </div>

      <div className="mt-4 bg-slate-800/50 rounded-lg p-4">
        <p className="text-gray-300 text-sm">
          <strong className="text-gray-100">Interpretation:</strong> Large circles show the average for each {factorA}.
          Small circles show individual {factorB} means within each {factorA}.
          Greater spread of small circles within a {factorA} indicates more variability at the {factorB} level.
          Differences between large circles indicate variability at the {factorA} level.
        </p>
      </div>
    </div>
  )
}

export default HierarchicalMeansPlot
