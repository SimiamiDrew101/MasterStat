import { useState, useRef } from 'react'
import { Users, Download, DollarSign, AlertCircle, Info, CheckCircle } from 'lucide-react'
import { exportSvgToPng } from '../utils/exportChart'

const OptimalAllocationChart = ({ allocationData }) => {
  const [hoveredPoint, setHoveredPoint] = useState(null)
  const svgRef = useRef(null)

  if (!allocationData) return null

  const { equal_allocation, optimal_allocation, comparison_curve, feasibility, interpretation, recommendations } = allocationData

  // SVG dimensions
  const width = 800
  const height = 500
  const margin = { top: 60, right: 60, bottom: 70, left: 80 }
  const plotWidth = width - margin.left - margin.right
  const plotHeight = height - margin.top - margin.bottom

  // Scales for comparison curve
  const ratios = comparison_curve.ratios || []
  const totalCosts = comparison_curve.total_costs.filter(c => c !== null) || []

  if (ratios.length === 0) return null

  const xMin = Math.min(...ratios)
  const xMax = Math.max(...ratios)
  const xRange = xMax - xMin || 1
  const xScale = (x) => ((x - xMin) / xRange) * plotWidth

  const yMin = Math.min(...totalCosts) * 0.95
  const yMax = Math.max(...totalCosts) * 1.05
  const yRange = yMax - yMin
  const yScale = (y) => plotHeight - ((y - yMin) / yRange) * plotHeight

  // Generate path for cost curve
  const linePath = ratios
    .map((ratio, i) => {
      if (comparison_curve.total_costs[i] === null) return null
      const command = i === 0 ? 'M' : 'L'
      return `${command} ${xScale(ratio)} ${yScale(comparison_curve.total_costs[i])}`
    })
    .filter(p => p !== null)
    .join(' ')

  // Y-axis ticks
  const numYTicks = 6
  const yTickStep = Math.ceil(yRange / numYTicks / 100) * 100
  const yTicks = []
  for (let i = 0; i <= numYTicks; i++) {
    const val = yMin + i * yTickStep
    if (val <= yMax) yTicks.push(val)
  }

  // X-axis ticks
  const xTicks = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0].filter(r => r >= xMin && r <= xMax)

  // Find optimal and equal allocation points
  const optimalRatio = optimal_allocation.ratio
  const equalRatio = 1.0

  const costSavings = optimal_allocation.cost_savings
  const costSavingsPercent = optimal_allocation.cost_savings_percent

  return (
    <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50 mt-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <Users className="w-6 h-6 text-cyan-400" />
          <h4 className="text-xl font-bold text-gray-100">Optimal Allocation Analysis</h4>
        </div>

        <button
          type="button"
          onClick={() => {
            if (svgRef.current) {
              const filename = `optimal-allocation-${new Date().toISOString().split('T')[0]}`
              exportSvgToPng(svgRef.current, filename)
            }
          }}
          className="px-3 py-2 rounded-lg text-sm font-medium bg-slate-700/50 text-gray-300 hover:bg-slate-700 transition-all flex items-center space-x-2"
          title="Export as PNG"
        >
          <Download className="w-4 h-4" />
          <span>Export PNG</span>
        </button>
      </div>

      {/* Feasibility Warnings */}
      {!feasibility.budget_feasible && (
        <div className="mb-4 bg-red-900/20 rounded-lg p-4 border border-red-700/30">
          <div className="flex items-start space-x-3">
            <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
            <div>
              <h5 className="text-red-200 font-semibold mb-1 text-sm">Budget Exceeded</h5>
              <p className="text-red-100 text-xs">{feasibility.budget_message}</p>
            </div>
          </div>
        </div>
      )}

      {!feasibility.constraints_met && feasibility.constraint_messages.length > 0 && (
        <div className="mb-4 bg-orange-900/20 rounded-lg p-4 border border-orange-700/30">
          <div className="flex items-start space-x-3">
            <AlertCircle className="w-5 h-5 text-orange-400 flex-shrink-0 mt-0.5" />
            <div>
              <h5 className="text-orange-200 font-semibold mb-1 text-sm">Constraints Violated</h5>
              <ul className="space-y-1">
                {feasibility.constraint_messages.map((msg, idx) => (
                  <li key={idx} className="text-orange-100 text-xs">{msg}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Comparison Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        {/* Equal Allocation */}
        <div className="bg-slate-700/30 rounded-lg p-4 border border-slate-600">
          <h5 className="text-gray-200 font-semibold mb-3 text-sm">Equal Allocation (1:1)</h5>
          <div className="space-y-2 text-xs">
            <div className="flex justify-between">
              <span className="text-gray-300">Group 1:</span>
              <span className="text-gray-100 font-semibold">{equal_allocation.n_group1} participants</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-300">Group 2:</span>
              <span className="text-gray-100 font-semibold">{equal_allocation.n_group2} participants</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-300">Total:</span>
              <span className="text-gray-100 font-semibold">{equal_allocation.total_n} participants</span>
            </div>
            <div className="flex justify-between pt-2 border-t border-slate-600">
              <span className="text-gray-300">Total Cost:</span>
              <span className="text-gray-100 font-semibold">${equal_allocation.total_cost.toLocaleString()}</span>
            </div>
          </div>
        </div>

        {/* Optimal Allocation */}
        <div className="bg-green-900/20 rounded-lg p-4 border border-green-700/30">
          <h5 className="text-green-200 font-semibold mb-3 text-sm flex items-center gap-2">
            <CheckCircle className="w-4 h-4" />
            Optimal Allocation ({optimalRatio.toFixed(2)}:1)
          </h5>
          <div className="space-y-2 text-xs">
            <div className="flex justify-between">
              <span className="text-gray-300">Group 1:</span>
              <span className="text-green-100 font-semibold">{optimal_allocation.n_group1} participants</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-300">Group 2:</span>
              <span className="text-green-100 font-semibold">{optimal_allocation.n_group2} participants</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-300">Total:</span>
              <span className="text-green-100 font-semibold">{optimal_allocation.total_n} participants</span>
            </div>
            <div className="flex justify-between pt-2 border-t border-green-700/30">
              <span className="text-gray-300">Total Cost:</span>
              <span className="text-green-100 font-semibold">${optimal_allocation.total_cost.toLocaleString()}</span>
            </div>
            <div className="flex justify-between bg-green-950/50 rounded p-2 mt-2">
              <span className="text-green-200 font-semibold">Savings:</span>
              <span className="text-green-100 font-bold">
                ${costSavings.toLocaleString()} ({costSavingsPercent.toFixed(1)}%)
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className="bg-slate-900/50 rounded-lg p-4">
        <svg ref={svgRef} width={width} height={height} className="mx-auto">
          <g transform={`translate(${margin.left}, ${margin.top})`}>
            {/* Grid lines */}
            {yTicks.map((tick, i) => (
              <line
                key={`y-grid-${i}`}
                x1={0}
                y1={yScale(tick)}
                x2={plotWidth}
                y2={yScale(tick)}
                stroke="#374151"
                strokeWidth="1"
                strokeDasharray="4,4"
                opacity="0.3"
              />
            ))}

            {/* Axes */}
            <line x1={0} y1={0} x2={0} y2={plotHeight} stroke="#9ca3af" strokeWidth="2" />
            <line x1={0} y1={plotHeight} x2={plotWidth} y2={plotHeight} stroke="#9ca3af" strokeWidth="2" />

            {/* Y-axis ticks and labels */}
            {yTicks.map((tick, i) => (
              <g key={`y-tick-${i}`}>
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
                  fontSize="12"
                  textAnchor="end"
                  dominantBaseline="middle"
                >
                  ${(tick / 1000).toFixed(1)}k
                </text>
              </g>
            ))}

            {/* X-axis ticks and labels */}
            {xTicks.map((tick, i) => (
              <g key={`x-tick-${i}`}>
                <line
                  x1={xScale(tick)}
                  y1={plotHeight}
                  x2={xScale(tick)}
                  y2={plotHeight + 5}
                  stroke="#9ca3af"
                  strokeWidth="2"
                />
                <text
                  x={xScale(tick)}
                  y={plotHeight + 20}
                  fill="#d1d5db"
                  fontSize="12"
                  textAnchor="middle"
                >
                  {tick.toFixed(2)}
                </text>
              </g>
            ))}

            {/* Cost curve */}
            <path
              d={linePath}
              fill="none"
              stroke="#8b5cf6"
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
            />

            {/* Equal allocation point (1:1) */}
            <g>
              <circle
                cx={xScale(equalRatio)}
                cy={yScale(equal_allocation.total_cost)}
                r="8"
                fill="#64748b"
                stroke="white"
                strokeWidth="2"
              />
              <text
                x={xScale(equalRatio)}
                y={yScale(equal_allocation.total_cost) - 15}
                fill="#64748b"
                fontSize="11"
                fontWeight="600"
                textAnchor="middle"
              >
                Equal (1:1)
              </text>
            </g>

            {/* Optimal allocation point */}
            <g>
              <circle
                cx={xScale(optimalRatio)}
                cy={yScale(optimal_allocation.total_cost)}
                r="8"
                fill="#22c55e"
                stroke="white"
                strokeWidth="2"
              />
              <text
                x={xScale(optimalRatio)}
                y={yScale(optimal_allocation.total_cost) - 15}
                fill="#22c55e"
                fontSize="11"
                fontWeight="600"
                textAnchor="middle"
              >
                Optimal
              </text>
              {/* Vertical line showing optimal */}
              <line
                x1={xScale(optimalRatio)}
                y1={yScale(optimal_allocation.total_cost)}
                x2={xScale(optimalRatio)}
                y2={plotHeight}
                stroke="#22c55e"
                strokeWidth="1.5"
                strokeDasharray="4,4"
                opacity="0.5"
              />
            </g>

            {/* Hover points */}
            {ratios.map((ratio, i) => {
              if (comparison_curve.total_costs[i] === null) return null
              return (
                <circle
                  key={`point-${i}`}
                  cx={xScale(ratio)}
                  cy={yScale(comparison_curve.total_costs[i])}
                  r="4"
                  fill="transparent"
                  stroke="transparent"
                  strokeWidth="10"
                  style={{ cursor: 'pointer' }}
                  onMouseEnter={() => setHoveredPoint({ ratio, cost: comparison_curve.total_costs[i], index: i })}
                  onMouseLeave={() => setHoveredPoint(null)}
                />
              )
            })}

            {/* Hover tooltip */}
            {hoveredPoint && (
              <g transform={`translate(${xScale(hoveredPoint.ratio)}, ${yScale(hoveredPoint.cost) - 50})`}>
                <rect
                  x="-70"
                  y="-35"
                  width="140"
                  height="33"
                  fill="#1e293b"
                  stroke="#8b5cf6"
                  strokeWidth="1.5"
                  rx="4"
                />
                <text x="0" y="-22" fill="#e5e7eb" fontSize="11" textAnchor="middle" fontWeight="600">
                  Ratio: {hoveredPoint.ratio.toFixed(2)}:1
                </text>
                <text x="0" y="-10" fill="#8b5cf6" fontSize="11" textAnchor="middle" fontWeight="600">
                  Cost: ${hoveredPoint.cost.toLocaleString()}
                </text>
              </g>
            )}

            {/* Axis labels */}
            <text
              x={plotWidth / 2}
              y={plotHeight + 55}
              fill="#d1d5db"
              fontSize="14"
              fontWeight="600"
              textAnchor="middle"
            >
              Allocation Ratio (n�/n�)
            </text>

            <text
              x={-plotHeight / 2}
              y={-55}
              fill="#d1d5db"
              fontSize="14"
              fontWeight="600"
              textAnchor="middle"
              transform={`rotate(-90, -${plotHeight / 2}, -55)`}
            >
              Total Study Cost ($)
            </text>

            {/* Title */}
            <text
              x={plotWidth / 2}
              y={-30}
              fill="#e5e7eb"
              fontSize="16"
              fontWeight="700"
              textAnchor="middle"
            >
              Total Cost vs. Allocation Ratio
            </text>

            {/* Subtitle */}
            <text
              x={plotWidth / 2}
              y={-10}
              fill="#9ca3af"
              fontSize="12"
              textAnchor="middle"
            >
              Group 1: ${optimal_allocation.cost_per_group1}, Group 2: ${optimal_allocation.cost_per_group2}
            </text>
          </g>
        </svg>
      </div>

      {/* Interpretation */}
      {interpretation && (
        <div className="mt-4 bg-blue-900/20 rounded-lg p-4 border border-blue-700/30">
          <div className="flex items-start space-x-3">
            <Info className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
            <div className="flex-1 space-y-2">
              <h5 className="text-blue-200 font-semibold text-sm">Neyman Optimal Allocation</h5>
              <p className="text-gray-300 text-xs font-mono bg-slate-800/50 rounded px-2 py-1">
                {interpretation.formula}
              </p>
              <p className="text-gray-300 text-xs leading-relaxed">
                {interpretation.explanation}
              </p>
              <p className="text-green-200 text-xs font-semibold bg-green-950/30 rounded px-2 py-1">
                {interpretation.savings_summary}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Recommendations */}
      {recommendations && recommendations.length > 0 && (
        <div className="mt-4 bg-purple-900/20 rounded-lg p-4 border border-purple-700/30">
          <div className="flex items-start space-x-3">
            <DollarSign className="w-5 h-5 text-purple-400 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <h5 className="text-purple-200 font-semibold mb-2 text-sm">Recommendations</h5>
              <ul className="space-y-1">
                {recommendations.map((rec, idx) => (
                  <li key={idx} className="text-gray-300 text-xs">{rec}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* How to Use */}
      <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-slate-700/30 rounded-lg p-4">
          <h5 className="text-gray-200 font-semibold mb-2 text-sm">When to Use This</h5>
          <p className="text-gray-400 text-xs leading-relaxed">
            Use optimal allocation when one group costs significantly more than the other (e.g., treatment group
            requires expensive procedures vs. control group). The formula ensures you minimize total cost while
            maintaining statistical power.
          </p>
        </div>

        <div className="bg-slate-700/30 rounded-lg p-4">
          <h5 className="text-gray-200 font-semibold mb-2 text-sm">Trade-offs</h5>
          <p className="text-gray-400 text-xs leading-relaxed">
            While optimal allocation saves money, highly unequal group sizes may raise practical concerns about
            bias, generalizability, or recruitment logistics. Consider whether the cost savings justify the
            imbalance for your specific study context.
          </p>
        </div>
      </div>
    </div>
  )
}

export default OptimalAllocationChart
