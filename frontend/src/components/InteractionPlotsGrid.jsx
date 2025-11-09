import { useRef } from 'react'
import { Download } from 'lucide-react'
import { exportMultipleSvgsToPng } from '../utils/exportChart'

const InteractionPlotsGrid = ({ data, responseName }) => {
  const svgRefs = useRef([])

  if (!data || Object.keys(data).length === 0) return null

  const interactionKeys = Object.keys(data)

  return (
    <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-gray-100 font-bold text-xl">Interaction Plots</h4>
        <button
          type="button"
          onClick={() => {
            const validRefs = svgRefs.current.filter(ref => ref !== null)
            if (validRefs.length > 0) {
              exportMultipleSvgsToPng(validRefs, `interaction-plots-${new Date().toISOString().split('T')[0]}`)
            }
          }}
          className="px-3 py-2 rounded-lg text-sm font-medium bg-slate-700/50 text-gray-300 hover:bg-slate-700 transition-all flex items-center space-x-2"
          title="Export as PNG"
        >
          <Download className="w-4 h-4" />
          <span>Export PNG</span>
        </button>
      </div>
      <p className="text-gray-400 text-sm mb-6">
        Parallel lines indicate no interaction. Crossing or non-parallel lines suggest an interaction effect.
      </p>

      <div className={`grid gap-6 ${interactionKeys.length === 1 ? 'grid-cols-1' : interactionKeys.length === 2 ? 'grid-cols-1 md:grid-cols-2' : 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3'}`}>
        {interactionKeys.map((key, index) => {
          const plotData = data[key]
          return (
            <SingleInteractionPlot
              key={key}
              title={key}
              plotData={plotData}
              responseName={responseName}
              svgRef={(el) => svgRefs.current[index] = el}
            />
          )
        })}
      </div>
    </div>
  )
}

const SingleInteractionPlot = ({ title, plotData, responseName, svgRef }) => {
  const { x_factor, line_factor, x_levels, lines } = plotData

  // Find min and max for scaling
  const allValues = lines.flatMap(line => line.values.filter(v => v !== null))
  if (allValues.length === 0) return null

  const minY = Math.min(...allValues)
  const maxY = Math.max(...allValues)
  const range = maxY - minY
  const padding = range * 0.2

  const chartMin = minY - padding
  const chartMax = maxY + padding
  const chartRange = chartMax - chartMin || 1

  // SVG dimensions
  const width = 400
  const height = 300
  const margin = { top: 20, right: 120, bottom: 50, left: 60 }
  const plotWidth = width - margin.left - margin.right
  const plotHeight = height - margin.top - margin.bottom

  // Scale functions
  const xScale = (index) => (index / Math.max(1, x_levels.length - 1)) * plotWidth
  const yScale = (value) => plotHeight - ((value - chartMin) / chartRange) * plotHeight

  // Colors for different lines
  const colors = ['#3b82f6', '#10b981', '#ef4444', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16']

  // Calculate y-axis ticks
  const numTicks = 5
  const tickStep = chartRange / (numTicks - 1)
  const yTicks = Array.from({ length: numTicks }, (_, i) => chartMin + i * tickStep)

  return (
    <div className="bg-slate-700/30 rounded-lg p-4 border border-slate-600">
      <h5 className="text-gray-200 font-semibold text-center mb-3">{title}</h5>

      <svg ref={svgRef} width={width} height={height} className="mx-auto">
        <g transform={`translate(${margin.left}, ${margin.top})`}>
          {/* Grid lines */}
          {yTicks.map((tick, i) => (
            <line
              key={i}
              x1={0}
              y1={yScale(tick)}
              x2={plotWidth}
              y2={yScale(tick)}
              stroke="#374151"
              strokeWidth="1"
              strokeDasharray="2,2"
            />
          ))}

          {/* Y-axis */}
          <line x1={0} y1={0} x2={0} y2={plotHeight} stroke="#9ca3af" strokeWidth="2" />

          {/* X-axis */}
          <line x1={0} y1={plotHeight} x2={plotWidth} y2={plotHeight} stroke="#9ca3af" strokeWidth="2" />

          {/* Y-axis ticks and labels */}
          {yTicks.map((tick, i) => (
            <g key={i}>
              <line
                x1={-5}
                y1={yScale(tick)}
                x2={0}
                y2={yScale(tick)}
                stroke="#9ca3af"
                strokeWidth="2"
              />
              <text
                x={-10}
                y={yScale(tick)}
                fill="#d1d5db"
                fontSize="11"
                textAnchor="end"
                dominantBaseline="middle"
              >
                {tick.toFixed(1)}
              </text>
            </g>
          ))}

          {/* X-axis ticks and labels */}
          {x_levels.map((level, i) => (
            <g key={i}>
              <line
                x1={xScale(i)}
                y1={plotHeight}
                x2={xScale(i)}
                y2={plotHeight + 5}
                stroke="#9ca3af"
                strokeWidth="2"
              />
              <text
                x={xScale(i)}
                y={plotHeight + 20}
                fill="#d1d5db"
                fontSize="12"
                textAnchor="middle"
              >
                {level}
              </text>
            </g>
          ))}

          {/* Plot lines and points */}
          {lines.map((line, lineIndex) => {
            const color = colors[lineIndex % colors.length]
            const validPoints = line.values
              .map((value, i) => ({ x: i, y: value }))
              .filter(p => p.y !== null)

            if (validPoints.length === 0) return null

            // Create path
            const pathData = validPoints
              .map((p, i) => {
                const command = i === 0 ? 'M' : 'L'
                return `${command} ${xScale(p.x)} ${yScale(p.y)}`
              })
              .join(' ')

            return (
              <g key={lineIndex}>
                {/* Line */}
                <path
                  d={pathData}
                  fill="none"
                  stroke={color}
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />

                {/* Points */}
                {validPoints.map((p, i) => (
                  <circle
                    key={i}
                    cx={xScale(p.x)}
                    cy={yScale(p.y)}
                    r="4"
                    fill={color}
                    stroke="white"
                    strokeWidth="1.5"
                  />
                ))}
              </g>
            )
          })}

          {/* Legend */}
          {lines.map((line, i) => (
            <g key={i} transform={`translate(${plotWidth + 15}, ${i * 25})`}>
              <line
                x1={0}
                y1={0}
                x2={20}
                y2={0}
                stroke={colors[i % colors.length]}
                strokeWidth="2.5"
              />
              <circle
                cx={10}
                cy={0}
                r="3"
                fill={colors[i % colors.length]}
              />
              <text
                x={25}
                y={0}
                fill="#d1d5db"
                fontSize="11"
                dominantBaseline="middle"
              >
                {line_factor}={line.label}
              </text>
            </g>
          ))}

          {/* Axis labels */}
          <text
            x={plotWidth / 2}
            y={plotHeight + 40}
            fill="#d1d5db"
            fontSize="13"
            fontWeight="600"
            textAnchor="middle"
          >
            {x_factor}
          </text>

          <text
            x={-plotHeight / 2}
            y={-45}
            fill="#d1d5db"
            fontSize="13"
            fontWeight="600"
            textAnchor="middle"
            transform={`rotate(-90, -${plotHeight / 2}, -45)`}
          >
            Mean {responseName || 'Response'}
          </text>
        </g>
      </svg>
    </div>
  )
}

export default InteractionPlotsGrid
