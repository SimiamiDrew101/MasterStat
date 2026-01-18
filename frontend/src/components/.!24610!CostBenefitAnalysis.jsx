import { useState, useRef } from 'react'
import { DollarSign, Download, TrendingUp, AlertCircle, Info } from 'lucide-react'
import { exportSvgToPng } from '../utils/exportChart'

const CostBenefitAnalysis = ({ costBenefitData }) => {
  const [hoveredPoint, setHoveredPoint] = useState(null)
  const svgRef = useRef(null)

  if (!costBenefitData) return null

  const { cost_curve, optimal_design, most_efficient_design, budget_info, recommendations } = costBenefitData

  // SVG dimensions
  const width = 800
  const height = 500
  const margin = { top: 60, right: 60, bottom: 70, left: 80 }
  const plotWidth = width - margin.left - margin.right
  const plotHeight = height - margin.top - margin.bottom

  // Scales
  const powerLevels = cost_curve.power_levels || []
  const totalCosts = cost_curve.total_costs || []
  const sampleSizes = cost_curve.sample_sizes || []

  if (powerLevels.length === 0) return null

  const xMin = Math.min(...powerLevels)
  const xMax = Math.max(...powerLevels)
  const xRange = xMax - xMin || 1
  const xScale = (x) => ((x - xMin) / xRange) * plotWidth

  const yMin = 0
  const yMax = Math.max(...totalCosts) * 1.1
  const yRange = yMax - yMin
  const yScale = (y) => plotHeight - ((y - yMin) / yRange) * plotHeight

  // Generate path for curve
  const linePath = powerLevels.map((power, i) => {
    const command = i === 0 ? 'M' : 'L'
    return `${command} ${xScale(power)} ${yScale(totalCosts[i])}`
  }).join(' ')

  // Y-axis ticks (costs)
  const numYTicks = 6
  const yTickStep = Math.ceil(yRange / numYTicks / 100) * 100
  const yTicks = []
  for (let i = 0; i <= numYTicks; i++) {
    const val = yMin + i * yTickStep
    if (val <= yMax) yTicks.push(val)
  }

  // X-axis ticks (power)
  const xTicks = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95].filter(p => p >= xMin && p <= xMax)

  // Find optimal point index
  const optimalIndex = powerLevels.findIndex(p => Math.abs(p - optimal_design.power) < 0.01)
  const efficientIndex = powerLevels.findIndex(p => Math.abs(p - most_efficient_design.power) < 0.01)

  return (
    <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50 mt-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <DollarSign className="w-6 h-6 text-green-400" />
          <h4 className="text-xl font-bold text-gray-100">Cost-Benefit Analysis</h4>
        </div>

        <button
          type="button"
          onClick={() => {
            if (svgRef.current) {
              const filename = `cost-benefit-${new Date().toISOString().split('T')[0]}`
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

      {/* Budget Warning */}
      {optimal_design.budget_exceeded && (
        <div className="mb-4 bg-orange-900/20 rounded-lg p-4 border border-orange-700/30">
          <div className="flex items-start space-x-3">
            <AlertCircle className="w-5 h-5 text-orange-400 flex-shrink-0 mt-0.5" />
            <div>
              <h5 className="text-orange-200 font-semibold mb-1 text-sm">Budget Constraint Active</h5>
              <p className="text-orange-100 text-xs">
                Your budget limits you to {(optimal_design.power * 100).toFixed(0)}% power.
                Consider increasing budget or adjusting expectations.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {/* Optimal Design */}
        <div className="bg-blue-900/20 rounded-lg p-4 border border-blue-700/30">
          <h5 className="text-blue-200 font-semibold mb-2 text-sm flex items-center gap-2">
            <TrendingUp className="w-4 h-4" />
            Optimal Design
          </h5>
          <div className="space-y-1 text-xs">
            <div className="flex justify-between">
              <span className="text-gray-300">Sample Size:</span>
              <span className="text-blue-100 font-semibold">{optimal_design.sample_size}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-300">Power:</span>
              <span className="text-blue-100 font-semibold">{(optimal_design.power * 100).toFixed(0)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-300">Total Cost:</span>
              <span className="text-blue-100 font-semibold">${optimal_design.total_cost.toLocaleString()}</span>
            </div>
          </div>
        </div>

        {/* Most Efficient */}
        <div className="bg-green-900/20 rounded-lg p-4 border border-green-700/30">
          <h5 className="text-green-200 font-semibold mb-2 text-sm flex items-center gap-2">
            <DollarSign className="w-4 h-4" />
            Most Efficient
          </h5>
          <div className="space-y-1 text-xs">
            <div className="flex justify-between">
              <span className="text-gray-300">Sample Size:</span>
              <span className="text-green-100 font-semibold">{most_efficient_design.sample_size}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-300">Power:</span>
              <span className="text-green-100 font-semibold">{(most_efficient_design.power * 100).toFixed(0)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-300">Cost/Power:</span>
              <span className="text-green-100 font-semibold">${most_efficient_design.cost_per_power_unit.toLocaleString()}</span>
            </div>
          </div>
        </div>

        {/* Budget Info */}
        {budget_info.has_constraint && (
          <div className="bg-purple-900/20 rounded-lg p-4 border border-purple-700/30">
            <h5 className="text-purple-200 font-semibold mb-2 text-sm">Budget Info</h5>
            <div className="space-y-1 text-xs">
              <div className="flex justify-between">
                <span className="text-gray-300">Max Budget:</span>
                <span className="text-purple-100 font-semibold">${budget_info.max_budget.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Max Participants:</span>
                <span className="text-purple-100 font-semibold">{budget_info.max_affordable_participants}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Remaining:</span>
                <span className="text-purple-100 font-semibold">
                  ${(budget_info.max_budget - optimal_design.total_cost).toLocaleString()}
                </span>
              </div>
            </div>
          </div>
        )}
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

            {/* Budget constraint line */}
            {budget_info.has_constraint && (
              <g>
                <line
                  x1={0}
                  y1={yScale(budget_info.max_budget)}
                  x2={plotWidth}
                  y2={yScale(budget_info.max_budget)}
                  stroke="#ef4444"
                  strokeWidth="2"
                  strokeDasharray="6,3"
                  opacity="0.7"
                />
                <text
                  x={plotWidth - 5}
                  y={yScale(budget_info.max_budget) - 5}
                  fill="#ef4444"
                  fontSize="11"
                  fontWeight="600"
                  textAnchor="end"
                >
                  Budget Limit
                </text>
              </g>
            )}

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
                  ${(tick / 1000).toFixed(0)}k
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
                  {(tick * 100).toFixed(0)}%
                </text>
              </g>
            ))}

            {/* Cost curve */}
            <path
              d={linePath}
              fill="none"
              stroke="#10b981"
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
            />

            {/* Fill area under curve */}
            <path
              d={`${linePath} L ${xScale(powerLevels[powerLevels.length - 1])} ${plotHeight} L ${xScale(powerLevels[0])} ${plotHeight} Z`}
              fill="#10b981"
              opacity="0.1"
            />

            {/* Optimal design point */}
            {optimalIndex >= 0 && (
              <g>
                <circle
                  cx={xScale(powerLevels[optimalIndex])}
                  cy={yScale(totalCosts[optimalIndex])}
                  r="8"
                  fill="#3b82f6"
                  stroke="white"
                  strokeWidth="2"
                />
                <text
                  x={xScale(powerLevels[optimalIndex])}
                  y={yScale(totalCosts[optimalIndex]) - 15}
                  fill="#3b82f6"
                  fontSize="11"
                  fontWeight="600"
                  textAnchor="middle"
                >
                  Optimal
                </text>
              </g>
            )}

            {/* Most efficient point */}
            {efficientIndex >= 0 && efficientIndex !== optimalIndex && (
              <g>
                <circle
                  cx={xScale(powerLevels[efficientIndex])}
                  cy={yScale(totalCosts[efficientIndex])}
                  r="7"
                  fill="#22c55e"
                  stroke="white"
                  strokeWidth="2"
                />
                <text
                  x={xScale(powerLevels[efficientIndex])}
                  y={yScale(totalCosts[efficientIndex]) - 15}
                  fill="#22c55e"
                  fontSize="11"
                  fontWeight="600"
                  textAnchor="middle"
                >
                  Efficient
                </text>
              </g>
            )}

            {/* Hover points */}
            {powerLevels.map((power, i) => (
              <circle
                key={`point-${i}`}
                cx={xScale(power)}
                cy={yScale(totalCosts[i])}
                r="4"
                fill="transparent"
                stroke="transparent"
                strokeWidth="10"
                style={{ cursor: 'pointer' }}
                onMouseEnter={() => setHoveredPoint({ power, cost: totalCosts[i], n: sampleSizes[i], index: i })}
                onMouseLeave={() => setHoveredPoint(null)}
              />
            ))}

            {/* Hover tooltip */}
            {hoveredPoint && (
              <g transform={`translate(${xScale(hoveredPoint.power)}, ${yScale(hoveredPoint.cost) - 50})`}>
                <rect
                  x="-70"
                  y="-40"
                  width="140"
                  height="38"
                  fill="#1e293b"
                  stroke="#10b981"
                  strokeWidth="1.5"
                  rx="4"
                />
                <text x="0" y="-26" fill="#e5e7eb" fontSize="11" textAnchor="middle" fontWeight="600">
                  n = {hoveredPoint.n}
                </text>
                <text x="0" y="-14" fill="#10b981" fontSize="11" textAnchor="middle" fontWeight="600">
                  Power: {(hoveredPoint.power * 100).toFixed(0)}%
                </text>
                <text x="0" y="-2" fill="#10b981" fontSize="11" textAnchor="middle" fontWeight="600">
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
