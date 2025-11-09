import { useRef } from 'react'
import { Download } from 'lucide-react'
import { exportSvgToPng } from '../utils/exportChart'

const ParetoChart = ({ data, title = "Pareto Chart of Effects" }) => {
  const svgRef = useRef(null)

  if (!data || data.length === 0) return null

  // SVG dimensions
  const width = 700
  const height = 400
  const margin = { top: 30, right: 80, bottom: 100, left: 60 }
  const plotWidth = width - margin.left - margin.right
  const plotHeight = height - margin.top - margin.bottom

  const maxEffect = Math.max(...data.map(d => d.abs_effect))
  const barWidth = plotWidth / data.length * 0.8

  // Calculate cumulative percentage
  const totalAbsEffect = data.reduce((sum, d) => sum + d.abs_effect, 0)
  let cumulative = 0
  const dataWithCumulative = data.map(d => {
    cumulative += d.abs_effect
    return {
      ...d,
      cumulative_pct: (cumulative / totalAbsEffect) * 100
    }
  })

  // Scale functions
  const xScale = (index) => (index + 0.5) * (plotWidth / data.length)
  const yScale = (value) => plotHeight - (value / maxEffect) * plotHeight
  const pctScale = (pct) => plotHeight - (pct / 100) * plotHeight

  return (
    <div className="bg-slate-700/50 rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-gray-100 font-semibold">{title}</h4>
        <button
          type="button"
          onClick={() => {
            if (svgRef.current) {
              exportSvgToPng(svgRef.current, `pareto-chart-${new Date().toISOString().split('T')[0]}`)
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
        Shows effects ranked by magnitude. The cumulative line helps identify the vital few effects.
      </p>

      <div className="flex justify-center">
        <svg ref={svgRef} width={width} height={height}>
          <g transform={`translate(${margin.left}, ${margin.top})`}>
            {/* Grid lines for bars */}
            {[0, 0.25, 0.5, 0.75, 1].map((fraction, i) => (
              <line
                key={`grid-bar-${i}`}
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

            {/* Y-axis (bars) */}
            <line
              x1={0}
              y1={0}
              x2={0}
              y2={plotHeight}
              stroke="#64748b"
              strokeWidth={2}
            />

            {/* X-axis */}
            <line
              x1={0}
              y1={plotHeight}
              x2={plotWidth}
              y2={plotHeight}
              stroke="#64748b"
              strokeWidth={2}
            />

            {/* Bars */}
            {dataWithCumulative.map((item, index) => {
              const x = xScale(index) - barWidth / 2
              const barHeight = plotHeight - yScale(item.abs_effect)

              return (
                <g key={index}>
                  <rect
                    x={x}
                    y={yScale(item.abs_effect)}
                    width={barWidth}
                    height={barHeight}
                    fill="#3b82f6"
                    opacity={0.7}
                    stroke="#2563eb"
                    strokeWidth={1}
                  />

                  {/* Effect label on bar */}
                  <text
                    x={xScale(index)}
                    y={yScale(item.abs_effect) - 5}
                    textAnchor="middle"
                    fill="#e2e8f0"
                    fontSize="11"
                    fontWeight="600"
                  >
                    {item.abs_effect.toFixed(2)}
                  </text>

                  {/* Factor name */}
                  <text
                    x={xScale(index)}
                    y={plotHeight + 15}
                    textAnchor="end"
                    fill="#e2e8f0"
                    fontSize="12"
                    fontWeight="500"
                    transform={`rotate(-45, ${xScale(index)}, ${plotHeight + 15})`}
                  >
                    {item.name}
                  </text>
                </g>
              )
            })}

            {/* Cumulative percentage line */}
            <path
              d={dataWithCumulative.map((item, i) => {
                const x = xScale(i)
                const y = pctScale(item.cumulative_pct)
                return `${i === 0 ? 'M' : 'L'} ${x} ${y}`
              }).join(' ')}
              fill="none"
              stroke="#ef4444"
              strokeWidth={3}
            />

            {/* Cumulative percentage points */}
            {dataWithCumulative.map((item, index) => (
              <circle
                key={`pct-${index}`}
                cx={xScale(index)}
                cy={pctScale(item.cumulative_pct)}
                r={4}
                fill="#ef4444"
                stroke="#7f1d1d"
                strokeWidth={2}
              />
            ))}

            {/* Y-axis labels (left - absolute effect) */}
            {[0, 0.25, 0.5, 0.75, 1].map((fraction, i) => {
              const value = maxEffect * (1 - fraction)
              return (
                <text
                  key={`y-label-${i}`}
                  x={-10}
                  y={plotHeight * fraction + 4}
                  textAnchor="end"
                  fill="#94a3b8"
                  fontSize="12"
                >
                  {value.toFixed(1)}
                </text>
              )
            })}

            {/* Y-axis labels (right - cumulative %) */}
            {[0, 25, 50, 75, 100].map((pct, i) => {
              return (
                <text
                  key={`pct-label-${i}`}
                  x={plotWidth + 10}
                  y={pctScale(pct) + 4}
                  textAnchor="start"
                  fill="#ef4444"
                  fontSize="12"
                >
                  {pct}%
                </text>
              )
            })}

            {/* Axis titles */}
            <text
              x={plotWidth / 2}
              y={plotHeight + 80}
              textAnchor="middle"
              fill="#94a3b8"
              fontSize="14"
              fontWeight="600"
            >
              Effects
            </text>

            <text
              x={-plotHeight / 2}
              y={-40}
              textAnchor="middle"
              fill="#94a3b8"
              fontSize="14"
              fontWeight="600"
              transform={`rotate(-90, -${plotHeight / 2}, -40)`}
            >
              Absolute Effect
            </text>

            <text
              x={plotWidth + 60}
              y={plotHeight / 2}
              textAnchor="middle"
              fill="#ef4444"
              fontSize="14"
              fontWeight="600"
              transform={`rotate(90, ${plotWidth + 60}, ${plotHeight / 2})`}
            >
              Cumulative %
            </text>

            {/* 80% reference line */}
            <line
              x1={0}
              y1={pctScale(80)}
              x2={plotWidth}
              y2={pctScale(80)}
              stroke="#f59e0b"
              strokeWidth={2}
              strokeDasharray="6,3"
            />
            <text
              x={plotWidth - 5}
              y={pctScale(80) - 5}
              textAnchor="end"
              fill="#f59e0b"
              fontSize="11"
              fontWeight="600"
            >
              80% (vital few)
            </text>
          </g>
        </svg>
      </div>

      {/* Legend */}
      <div className="mt-4 flex flex-wrap gap-4 justify-center text-sm text-gray-300">
        <div className="flex items-center space-x-2">
          <div className="w-6 h-4 bg-blue-500 opacity-70 border border-blue-600"></div>
          <span>Absolute Effect</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-8 h-1 bg-red-500"></div>
          <span>Cumulative %</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-8 h-1 bg-orange-500" style={{borderTop: '2px dashed'}}></div>
          <span>80% Line (Pareto Principle)</span>
        </div>
      </div>
    </div>
  )
}

export default ParetoChart
