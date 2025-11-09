import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import { LineChart, TrendingUp, Download } from 'lucide-react'
import { exportSvgToPng } from '../utils/exportChart'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const PowerCurveChart = ({
  calculatorType,
  testType,
  anovaType,
  alpha,
  alternative,
  effectSize,
  calculatedSampleSize,
  calculatedPower,
  numGroups,
  numLevelsA,
  numLevelsB,
  effectOfInterest,
  ratio,
  correlation
}) => {
  const [curveType, setCurveType] = useState('power_vs_n')
  const [curveData, setCurveData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [hoveredPoint, setHoveredPoint] = useState(null)
  const containerRef = useRef(null)
  const svgRef = useRef(null)

  useEffect(() => {
    fetchCurveData()
  }, [curveType, effectSize, calculatedSampleSize, alpha])

  const handleCurveTypeChange = (newType) => {
    if (newType === curveType) return // Don't do anything if already selected

    // Get current position of the container relative to viewport
    const rect = containerRef.current?.getBoundingClientRect()
    const offsetTop = rect?.top + window.scrollY

    // Change the curve type
    setCurveType(newType)

    // After React updates, restore position
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        if (containerRef.current) {
          const newRect = containerRef.current.getBoundingClientRect()
          const newOffsetTop = newRect.top + window.scrollY
          const diff = newOffsetTop - offsetTop
          window.scrollBy(0, diff)
        }
      })
    })
  }

  const fetchCurveData = async () => {
    setLoading(true)
    try {
      const payload = {
        test_family: calculatorType,
        test_type: calculatorType === 't-test' ? testType : anovaType,
        alpha: alpha,
        curve_type: curveType
      }

      if (curveType === 'power_vs_n') {
        payload.effect_size = effectSize
      } else {
        payload.sample_size = calculatedSampleSize
      }

      // Add test-specific parameters
      if (calculatorType === 't-test') {
        payload.alternative = alternative
        payload.ratio = ratio || 1.0
        payload.correlation = correlation || 0.5
      } else {
        payload.num_groups = numGroups || 3
        if (anovaType === 'two-way') {
          payload.num_levels_a = numLevelsA || 2
          payload.num_levels_b = numLevelsB || 2
          payload.effect_of_interest = effectOfInterest || 'main_a'
        }
      }

      const response = await axios.post(`${API_URL}/api/power/power-curve`, payload)
      setCurveData(response.data)
    } catch (err) {
      console.error('Error fetching power curve:', err)
    } finally {
      setLoading(false)
    }
  }

  if (!loading && (!curveData || !curveData.curves || curveData.curves.length === 0)) {
    return null
  }

  const curve = curveData?.curves?.[0]
  const x_values = curve?.x_values || []
  const y_values = curve?.y_values || []
  const x_label = curve?.x_label || ''
  const y_label = curve?.y_label || ''

  // SVG dimensions
  const width = 800
  const height = 500
  const margin = { top: 40, right: 40, bottom: 70, left: 70 }
  const plotWidth = width - margin.left - margin.right
  const plotHeight = height - margin.top - margin.bottom

  // Scale functions (only calculate if we have data)
  const xMin = x_values.length > 0 ? Math.min(...x_values) : 0
  const xMax = x_values.length > 0 ? Math.max(...x_values) : 100
  const xRange = xMax - xMin || 1
  const xScale = (x) => ((x - xMin) / xRange) * plotWidth

  const yMin = 0
  const yMax = 1
  const yRange = yMax - yMin
  const yScale = (y) => plotHeight - ((y - yMin) / yRange) * plotHeight

  // Generate path for curve
  const linePath = x_values.length > 0 ? x_values
    .map((x, i) => {
      const command = i === 0 ? 'M' : 'L'
      return `${command} ${xScale(x)} ${yScale(y_values[i])}`
    })
    .join(' ') : ''

  // Y-axis ticks
  const yTicks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

  // X-axis ticks (adaptive based on range)
  const numXTicks = 8
  const xTickStep = Math.ceil(xRange / numXTicks) || 10
  const xTicks = []
  for (let i = 0; i <= numXTicks; i++) {
    const val = xMin + i * xTickStep
    if (val <= xMax) xTicks.push(val)
  }

  // Find closest point on curve for highlighting
  let highlightIndex = -1
  if (x_values.length > 0) {
    if (curveType === 'power_vs_n' && calculatedSampleSize) {
      highlightIndex = x_values.findIndex(x => x >= calculatedSampleSize)
      if (highlightIndex === -1) highlightIndex = x_values.length - 1
    } else if (curveType === 'power_vs_effect' && effectSize) {
      highlightIndex = x_values.findIndex(x => x >= effectSize)
      if (highlightIndex === -1) highlightIndex = x_values.length - 1
    }
  }

  return (
    <div ref={containerRef} className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <TrendingUp className="w-6 h-6 text-cyan-400" />
          <h4 className="text-xl font-bold text-gray-100">Power Analysis Curve</h4>
        </div>

        <div className="flex items-center space-x-4">
          {/* Export Button */}
          <button
            type="button"
            onClick={() => {
              if (svgRef.current) {
                const filename = `power-curve-${curveType}-${new Date().toISOString().split('T')[0]}`
                exportSvgToPng(svgRef.current, filename)
              }
            }}
            className="px-3 py-2 rounded-lg text-sm font-medium bg-slate-700/50 text-gray-300 hover:bg-slate-700 transition-all flex items-center space-x-2"
            title="Export as PNG"
          >
            <Download className="w-4 h-4" />
            <span>Export PNG</span>
          </button>

          {/* Curve Type Toggle */}
          <div className="flex space-x-2">
          <button
            type="button"
            onClick={(e) => {
              e.preventDefault()
              e.stopPropagation()
              handleCurveTypeChange('power_vs_n')
            }}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              curveType === 'power_vs_n'
                ? 'bg-cyan-500 text-white'
                : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
            }`}
          >
            Power vs. Sample Size
          </button>
          <button
            type="button"
            onClick={(e) => {
              e.preventDefault()
              e.stopPropagation()
              handleCurveTypeChange('power_vs_effect')
            }}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              curveType === 'power_vs_effect'
                ? 'bg-cyan-500 text-white'
                : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
            }`}
          >
            Power vs. Effect Size
          </button>
          </div>
        </div>
      </div>

      <p className="text-gray-400 text-sm mb-4">
        {curveType === 'power_vs_n'
          ? 'Shows how statistical power increases with sample size. The red dot marks your calculated design.'
          : 'Shows the minimum detectable effect size at different power levels for your sample size.'}
      </p>

      {/* Chart */}
      <div className="bg-slate-900/50 rounded-lg p-4 overflow-x-auto" style={{ minHeight: '500px' }}>
        {loading ? (
          <div className="flex items-center justify-center h-full" style={{ minHeight: '500px' }}>
            <div className="text-gray-400">Generating power curve...</div>
          </div>
        ) : (
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

            {/* Reference line for 0.8 power */}
            <line
              x1={0}
              y1={yScale(0.8)}
              x2={plotWidth}
              y2={yScale(0.8)}
              stroke="#10b981"
              strokeWidth="2"
              strokeDasharray="6,3"
              opacity="0.5"
            />
            <text
              x={plotWidth - 5}
              y={yScale(0.8) - 5}
              fill="#10b981"
              fontSize="11"
              textAnchor="end"
            >
              80% Power
            </text>

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
                  {tick.toFixed(1)}
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
                  {Math.round(tick)}
                </text>
              </g>
            ))}

            {/* Power curve */}
            <path
              d={linePath}
              fill="none"
              stroke="#3b82f6"
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
            />

            {/* Highlight point (calculated design) */}
            {highlightIndex >= 0 && (
              <g>
                <circle
                  cx={xScale(x_values[highlightIndex])}
                  cy={yScale(y_values[highlightIndex])}
                  r="6"
                  fill="#ef4444"
                  stroke="white"
                  strokeWidth="2"
                />
                {/* Vertical line to x-axis */}
                <line
                  x1={xScale(x_values[highlightIndex])}
                  y1={yScale(y_values[highlightIndex])}
                  x2={xScale(x_values[highlightIndex])}
                  y2={plotHeight}
                  stroke="#ef4444"
                  strokeWidth="1.5"
                  strokeDasharray="4,4"
                  opacity="0.6"
                />
                {/* Horizontal line to y-axis */}
                <line
                  x1={0}
                  y1={yScale(y_values[highlightIndex])}
                  x2={xScale(x_values[highlightIndex])}
                  y2={yScale(y_values[highlightIndex])}
                  stroke="#ef4444"
                  strokeWidth="1.5"
                  strokeDasharray="4,4"
                  opacity="0.6"
                />
              </g>
            )}

            {/* Hover points */}
            {x_values.map((x, i) => (
              <circle
                key={`point-${i}`}
                cx={xScale(x)}
                cy={yScale(y_values[i])}
                r="4"
                fill="transparent"
                stroke="transparent"
                strokeWidth="8"
                style={{ cursor: 'pointer' }}
                onMouseEnter={() => setHoveredPoint({ x: x, y: y_values[i], index: i })}
                onMouseLeave={() => setHoveredPoint(null)}
              />
            ))}

            {/* Hover tooltip */}
            {hoveredPoint && (
              <g transform={`translate(${xScale(hoveredPoint.x)}, ${yScale(hoveredPoint.y) - 30})`}>
                <rect
                  x="-60"
                  y="-30"
                  width="120"
                  height="28"
                  fill="#1e293b"
                  stroke="#3b82f6"
                  strokeWidth="1.5"
                  rx="4"
                />
                <text
                  x="0"
                  y="-18"
                  fill="#e5e7eb"
                  fontSize="11"
                  textAnchor="middle"
                  fontWeight="600"
                >
                  {curveType === 'power_vs_n' ? 'n' : 'Effect'}: {hoveredPoint.x.toFixed(hoveredPoint.x < 10 ? 2 : 0)}
                </text>
                <text
                  x="0"
                  y="-6"
                  fill="#3b82f6"
                  fontSize="11"
                  textAnchor="middle"
                  fontWeight="600"
                >
                  Power: {(hoveredPoint.y * 100).toFixed(1)}%
                </text>
              </g>
            )}

            {/* Axis labels */}
            <text
              x={plotWidth / 2}
              y={plotHeight + 50}
              fill="#d1d5db"
              fontSize="14"
              fontWeight="600"
              textAnchor="middle"
            >
              {x_label}
            </text>

            <text
              x={-plotHeight / 2}
              y={-50}
              fill="#d1d5db"
              fontSize="14"
              fontWeight="600"
              textAnchor="middle"
              transform={`rotate(-90, -${plotHeight / 2}, -50)`}
            >
              {y_label}
            </text>

            {/* Title */}
            {curve?.name && (
              <text
                x={plotWidth / 2}
                y={-15}
                fill="#e5e7eb"
                fontSize="16"
                fontWeight="700"
                textAnchor="middle"
              >
                {curve.name}
              </text>
            )}
          </g>
        </svg>
        )}
      </div>

      {/* Insights */}
      <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-slate-700/30 rounded-lg p-4">
          <h5 className="text-gray-200 font-semibold mb-2 text-sm">What This Shows</h5>
          <p className="text-gray-400 text-xs leading-relaxed">
            {curveType === 'power_vs_n' ? (
              <>
                As sample size increases, statistical power improves. The curve shows diminishing returns -
                doubling your sample from 20 to 40 has more impact than from 100 to 120.
              </>
            ) : (
              <>
                This shows the minimum effect size detectable at different power levels. Smaller effects
                require larger samples to detect reliably.
              </>
            )}
          </p>
        </div>

        <div className="bg-slate-700/30 rounded-lg p-4">
          <h5 className="text-gray-200 font-semibold mb-2 text-sm">How to Use This</h5>
          <p className="text-gray-400 text-xs leading-relaxed">
            {curveType === 'power_vs_n' ? (
              <>
                Hover over the curve to explore trade-offs. If you can't achieve your target sample size,
                check what power you'd have with fewer participants.
              </>
            ) : (
              <>
                Use this to understand sensitivity. If the true effect is smaller than assumed,
                you can see how much power you'd actually have to detect it.
              </>
            )}
          </p>
        </div>
      </div>
    </div>
  )
}

export default PowerCurveChart
