import { useRef } from 'react'
import { Download } from 'lucide-react'
import { exportSvgToPng } from '../utils/exportChart'

const InteractionPlot = ({ data, factorAName, factorBName }) => {
  const svgRef = useRef(null)

  if (!data || Object.keys(data).length === 0) return null

  // Parse the interaction means data
  // Format: "factorA_value, factorB_value": mean
  const parsedData = {}
  Object.entries(data).forEach(([key, mean]) => {
    const [aVal, bVal] = key.split(', ')
    if (!parsedData[aVal]) {
      parsedData[aVal] = {}
    }
    parsedData[aVal][bVal] = mean
  })

  const factorALevels = Object.keys(parsedData).sort()
  const factorBLevels = [...new Set(Object.values(parsedData).flatMap(obj => Object.keys(obj)))].sort()

  // Prepare data for each factor B level (each line)
  const lines = factorBLevels.map(bLevel => ({
    label: bLevel,
    points: factorALevels.map(aLevel => ({
      x: aLevel,
      y: parsedData[aLevel]?.[bLevel] || 0
    }))
  }))

  // Find min and max for scaling
  const allValues = lines.flatMap(line => line.points.map(p => p.y))
  const minY = Math.min(...allValues)
  const maxY = Math.max(...allValues)
  const range = maxY - minY
  const padding = range * 0.15

  const chartMin = minY - padding
  const chartMax = maxY + padding
  const chartRange = chartMax - chartMin

  // SVG dimensions
  const width = 600
  const height = 350
  const margin = { top: 30, right: 120, bottom: 60, left: 60 }
  const plotWidth = width - margin.left - margin.right
  const plotHeight = height - margin.top - margin.bottom

  // Scale functions
  const xScale = (index) => (index / (factorALevels.length - 1)) * plotWidth
  const yScale = (value) => plotHeight - ((value - chartMin) / chartRange) * plotHeight

  // Colors for different lines
  const colors = ['#3b82f6', '#10b981', '#ef4444', '#f59e0b', '#8b5cf6', '#ec4899']

  return (
    <div className="bg-slate-700/50 rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-gray-100 font-semibold">Interaction Plot</h4>
        <button
          type="button"
          onClick={() => {
            if (svgRef.current) {
              exportSvgToPng(svgRef.current, `interaction-plot-${new Date().toISOString().split('T')[0]}`)
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
        Parallel lines indicate no interaction. Crossing or non-parallel lines suggest interaction between factors.
      </p>

      <div className="flex justify-center">
        <svg ref={svgRef} width={width} height={height} className="overflow-visible">
          <g transform={`translate(${margin.left}, ${margin.top})`}>
            {/* Grid lines */}
            {[0, 0.25, 0.5, 0.75, 1].map((fraction, i) => {
              const y = plotHeight * fraction
              const value = chartMax - fraction * chartRange
              return (
                <g key={`grid-${i}`}>
                  <line
                    x1={0}
                    y1={y}
                    x2={plotWidth}
                    y2={y}
                    stroke="#475569"
                    strokeWidth={1}
                    strokeDasharray="4"
                    opacity={0.3}
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

            {/* Plot lines for each factor B level */}
            {lines.map((line, lineIndex) => {
              const color = colors[lineIndex % colors.length]
              const pathData = line.points.map((point, i) => {
                const x = xScale(i)
                const y = yScale(point.y)
                return `${i === 0 ? 'M' : 'L'} ${x} ${y}`
              }).join(' ')

              return (
                <g key={lineIndex}>
                  {/* Line */}
                  <path
                    d={pathData}
                    fill="none"
                    stroke={color}
                    strokeWidth={3}
                    strokeLinecap="round"
                  />

                  {/* Points */}
                  {line.points.map((point, i) => (
                    <g key={i}>
                      <circle
                        cx={xScale(i)}
                        cy={yScale(point.y)}
                        r={6}
                        fill={color}
                        stroke="#1e293b"
                        strokeWidth={2}
                      />

                      {/* Hover tooltip */}
                      <g className="group">
                        <circle
                          cx={xScale(i)}
                          cy={yScale(point.y)}
                          r={12}
                          fill="transparent"
                          className="cursor-pointer"
                        />
                        <g className="opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                          <rect
                            x={xScale(i) + 15}
                            y={yScale(point.y) - 25}
                            width="90"
                            height="50"
                            fill="#1e293b"
                            stroke={color}
                            strokeWidth="1"
                            rx="4"
                          />
                          <text x={xScale(i) + 25} y={yScale(point.y) - 5} fill="#e2e8f0" fontSize="11">
                            <tspan x={xScale(i) + 25} dy="0">{factorBName}: {line.label}</tspan>
                            <tspan x={xScale(i) + 25} dy="15">{factorAName}: {point.x}</tspan>
                            <tspan x={xScale(i) + 25} dy="15">Mean: {point.y.toFixed(2)}</tspan>
                          </text>
                        </g>
                      </g>
                    </g>
                  ))}
                </g>
              )
            })}

            {/* X-axis labels */}
            {factorALevels.map((level, i) => (
              <text
                key={i}
                x={xScale(i)}
                y={plotHeight + 20}
                textAnchor="middle"
                fill="#e2e8f0"
                fontSize="13"
                fontWeight="500"
              >
                {level}
              </text>
            ))}

            {/* Axis labels */}
            <text
              x={plotWidth / 2}
              y={plotHeight + 45}
              textAnchor="middle"
              fill="#94a3b8"
              fontSize="14"
            >
              {factorAName}
            </text>

            <text
              x={-plotHeight / 2}
              y={-40}
              textAnchor="middle"
              fill="#94a3b8"
              fontSize="14"
              transform={`rotate(-90, -${plotHeight / 2}, -40)`}
            >
              Mean Response
            </text>
          </g>

          {/* Legend */}
          <g transform={`translate(${width - margin.right + 20}, ${margin.top})`}>
            <text x={0} y={0} fill="#94a3b8" fontSize="13" fontWeight="600">
              {factorBName}
            </text>
            {lines.map((line, i) => {
              const color = colors[i % colors.length]
              return (
                <g key={i} transform={`translate(0, ${20 + i * 25})`}>
                  <line
                    x1={0}
                    y1={0}
                    x2={30}
                    y2={0}
                    stroke={color}
                    strokeWidth={3}
                  />
                  <circle
                    cx={15}
                    cy={0}
                    r={5}
                    fill={color}
                    stroke="#1e293b"
                    strokeWidth={2}
                  />
                  <text
                    x={40}
                    y={4}
                    fill="#e2e8f0"
                    fontSize="12"
                  >
                    {line.label}
                  </text>
                </g>
              )
            })}
          </g>
        </svg>
      </div>

      {/* Interpretation note */}
      <div className="mt-4 bg-slate-800/50 rounded-lg p-4">
        <p className="text-gray-300 text-sm">
          <strong className="text-gray-100">Interpretation:</strong>{' '}
          {lines.length > 1 ? (
            <>
              The lines represent how the response changes across levels of {factorAName} for each level of {factorBName}.
              If the lines are parallel, there is no interaction. If they cross or are non-parallel,
              the effect of {factorAName} depends on the level of {factorBName} (interaction present).
            </>
          ) : (
            `Plot shows the response across levels of ${factorAName}.`
          )}
        </p>
      </div>
    </div>
  )
}

export default InteractionPlot
