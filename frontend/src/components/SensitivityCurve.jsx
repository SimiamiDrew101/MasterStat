import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import { TrendingDown, Download } from 'lucide-react'
import { exportSvgToPng } from '../utils/exportChart'

const API_URL = import.meta.env.VITE_API_URL || ''

const SensitivityCurve = ({
  testFamily,
  testType,
  sampleSize,
  power,
  alpha,
  alternative,
  ratio,
  correlation,
  numGroups,
  numLevelsA,
  numLevelsB
}) => {
  const [curveData, setCurveData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [hoveredPoint, setHoveredPoint] = useState(null)
  const svgRef = useRef(null)

  useEffect(() => {
    fetchSensitivityCurve()
  }, [testFamily, testType, sampleSize, power, alpha])

  const fetchSensitivityCurve = async () => {
    setLoading(true)
    try {
      // Generate a range of sample sizes
      const sampleSizes = []
      const minN = Math.max(5, Math.floor(sampleSize * 0.3))
      const maxN = Math.ceil(sampleSize * 2.5)
      const step = Math.ceil((maxN - minN) / 30)

      for (let n = minN; n <= maxN; n += step) {
        sampleSizes.push(n)
      }

      // Calculate minimum detectable effect size for each sample size
      const effectSizes = []

      for (const n of sampleSizes) {
        try {
          let payload = {
            test_family: testFamily,
            test_type: testType,
            sample_size: n,
            power: power,
            alpha: alpha
          }

          if (testFamily === 't-test') {
            payload.alternative = alternative
            if (testType === 'two-sample') {
              payload.ratio = ratio || 1.0
            }
            if (testType === 'paired') {
              payload.correlation = correlation || 0.5
            }
          } else {
            if (testType === 'one-way') {
              payload.num_groups = numGroups || 3
            } else {
              payload.num_levels_a = numLevelsA || 2
              payload.num_levels_b = numLevelsB || 2
            }
          }

          const response = await axios.post(`${API_URL}/api/power/minimum-effect-size`, payload)
          const effectSize = response.data.minimum_effect_size ||
                           response.data.minimum_effect_size_f
          effectSizes.push(effectSize)
        } catch (err) {
          console.error('Error calculating effect size for n=' + n, err)
          effectSizes.push(null)
        }
      }

      setCurveData({
        sampleSizes: sampleSizes.filter((_, i) => effectSizes[i] !== null),
        effectSizes: effectSizes.filter(e => e !== null)
      })
    } catch (err) {
      console.error('Error generating sensitivity curve:', err)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50 mt-4">
        <div className="flex items-center justify-center h-64">
          <div className="text-gray-400">Generating sensitivity curve...</div>
        </div>
      </div>
    )
  }

  if (!curveData || curveData.sampleSizes.length === 0) {
    return null
  }

  const { sampleSizes, effectSizes } = curveData

  // SVG dimensions
  const width = 800
  const height = 500
  const margin = { top: 60, right: 60, bottom: 70, left: 80 }
  const plotWidth = width - margin.left - margin.right
  const plotHeight = height - margin.top - margin.bottom

  // Scales
  const xMin = Math.min(...sampleSizes)
  const xMax = Math.max(...sampleSizes)
  const xRange = xMax - xMin || 1
  const xScale = (x) => ((x - xMin) / xRange) * plotWidth

  const yMin = 0
  const yMax = Math.max(...effectSizes) * 1.1
  const yRange = yMax - yMin
  const yScale = (y) => plotHeight - ((y - yMin) / yRange) * plotHeight

  // Generate path
  const linePath = sampleSizes
    .map((x, i) => {
      const command = i === 0 ? 'M' : 'L'
      return `${command} ${xScale(x)} ${yScale(effectSizes[i])}`
    })
    .join(' ')

  // Axes ticks
  const numYTicks = 8
  const yTickStep = yRange / numYTicks
  const yTicks = []
  for (let i = 0; i <= numYTicks; i++) {
    yTicks.push(yMin + i * yTickStep)
  }

  const numXTicks = 8
  const xTickStep = Math.ceil(xRange / numXTicks) || 10
  const xTicks = []
  for (let i = 0; i <= numXTicks; i++) {
    const val = xMin + i * xTickStep
    if (val <= xMax) xTicks.push(val)
  }

  // Find current sample size point
  const currentIndex = sampleSizes.findIndex(n => n >= sampleSize)
  const highlightIndex = currentIndex >= 0 ? currentIndex : sampleSizes.length - 1

  // Effect size metric label
  const effectLabel = testFamily === 't-test' ? "Cohen's d" : "Cohen's f"

  // Reference lines for effect size thresholds
  let referenceLines = []
  if (testFamily === 't-test') {
    referenceLines = [
      { value: 0.2, label: 'Small (0.2)', color: '#fbbf24' },
      { value: 0.5, label: 'Medium (0.5)', color: '#fb923c' },
      { value: 0.8, label: 'Large (0.8)', color: '#f87171' }
    ]
  } else {
    referenceLines = [
      { value: 0.1, label: 'Small (0.1)', color: '#fbbf24' },
      { value: 0.25, label: 'Medium (0.25)', color: '#fb923c' },
      { value: 0.4, label: 'Large (0.4)', color: '#f87171' }
    ]
  }

  return (
    <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50 mt-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <TrendingDown className="w-6 h-6 text-cyan-400" />
          <h4 className="text-xl font-bold text-gray-100">Sensitivity Analysis</h4>
        </div>

        <button
          onClick={() => {
            if (svgRef.current) {
              const filename = `sensitivity-curve-${testFamily}-${new Date().toISOString().split('T')[0]}`
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

      <p className="text-gray-400 text-sm mb-4">
        Shows how the minimum detectable effect size changes with sample size. Larger samples can detect smaller effects.
        Your current design is marked with a red dot.
      </p>

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

            {/* Reference lines for effect size thresholds */}
            {referenceLines.map((ref, i) => (
              ref.value <= yMax && (
                <g key={`ref-${i}`}>
                  <line
                    x1={0}
                    y1={yScale(ref.value)}
                    x2={plotWidth}
                    y2={yScale(ref.value)}
                    stroke={ref.color}
                    strokeWidth="2"
                    strokeDasharray="6,3"
                    opacity="0.5"
                  />
                  <text
                    x={plotWidth - 5}
                    y={yScale(ref.value) - 5}
                    fill={ref.color}
                    fontSize="11"
                    textAnchor="end"
                    fontWeight="600"
                  >
                    {ref.label}
                  </text>
                </g>
              )
            ))}

            {/* Axes */}
            <line x1={0} y1={0} x2={0} y2={plotHeight} stroke="#9ca3af" strokeWidth="2" />
            <line x1={0} y1={plotHeight} x2={plotWidth} y2={plotHeight} stroke="#9ca3af" strokeWidth="2" />

            {/* Y-axis ticks */}
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
                  {tick.toFixed(2)}
                </text>
              </g>
            ))}

            {/* X-axis ticks */}
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

            {/* Sensitivity curve */}
            <path
              d={linePath}
              fill="none"
              stroke="#8b5cf6"
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
            />

            {/* Fill area under curve */}
            <path
              d={`${linePath} L ${xScale(sampleSizes[sampleSizes.length - 1])} ${plotHeight} L ${xScale(sampleSizes[0])} ${plotHeight} Z`}
              fill="#8b5cf6"
              opacity="0.1"
            />

            {/* Highlight current sample size */}
            {highlightIndex >= 0 && (
              <g>
                <circle
                  cx={xScale(sampleSizes[highlightIndex])}
                  cy={yScale(effectSizes[highlightIndex])}
                  r="6"
                  fill="#ef4444"
                  stroke="white"
                  strokeWidth="2"
                />
                {/* Vertical line to x-axis */}
                <line
                  x1={xScale(sampleSizes[highlightIndex])}
                  y1={yScale(effectSizes[highlightIndex])}
                  x2={xScale(sampleSizes[highlightIndex])}
                  y2={plotHeight}
                  stroke="#ef4444"
                  strokeWidth="1.5"
                  strokeDasharray="4,4"
                  opacity="0.6"
                />
                {/* Horizontal line to y-axis */}
                <line
                  x1={0}
                  y1={yScale(effectSizes[highlightIndex])}
                  x2={xScale(sampleSizes[highlightIndex])}
                  y2={yScale(effectSizes[highlightIndex])}
                  stroke="#ef4444"
                  strokeWidth="1.5"
                  strokeDasharray="4,4"
                  opacity="0.6"
                />
              </g>
            )}

            {/* Hover points */}
            {sampleSizes.map((n, i) => (
              <circle
                key={`point-${i}`}
                cx={xScale(n)}
                cy={yScale(effectSizes[i])}
                r="4"
                fill="transparent"
                stroke="transparent"
                strokeWidth="8"
                style={{ cursor: 'pointer' }}
                onMouseEnter={() => setHoveredPoint({ n, effect: effectSizes[i], index: i })}
                onMouseLeave={() => setHoveredPoint(null)}
              />
            ))}

            {/* Hover tooltip */}
            {hoveredPoint && (
              <g transform={`translate(${xScale(hoveredPoint.n)}, ${yScale(hoveredPoint.effect) - 40})`}>
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
                <text
                  x="0"
                  y="-22"
                  fill="#e5e7eb"
                  fontSize="11"
                  textAnchor="middle"
                  fontWeight="600"
                >
                  n = {hoveredPoint.n}
                </text>
                <text
                  x="0"
                  y="-10"
                  fill="#8b5cf6"
                  fontSize="11"
                  textAnchor="middle"
                  fontWeight="600"
                >
                  Min Effect: {hoveredPoint.effect.toFixed(3)}
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
              Sample Size {testFamily === 't-test' && testType !== 'two-sample' ? '(n)' : '(per group)'}
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
              Minimum Detectable Effect Size ({effectLabel})
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
              Sensitivity Analysis: Sample Size vs. Detectable Effect
            </text>

            {/* Subtitle */}
            <text
              x={plotWidth / 2}
              y={-10}
              fill="#9ca3af"
              fontSize="12"
              textAnchor="middle"
            >
              Power = {(power * 100).toFixed(0)}%, α = {alpha}
            </text>
          </g>
        </svg>
      </div>

      {/* Insights */}
      <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-slate-700/30 rounded-lg p-4">
          <h5 className="text-gray-200 font-semibold mb-2 text-sm">What This Shows</h5>
          <p className="text-gray-400 text-xs leading-relaxed">
            The curve demonstrates the fundamental trade-off in study design: larger samples can detect
            smaller effects. Notice how the curve flattens - doubling your sample size doesn't halve
            the minimum detectable effect. This shows diminishing returns of increasing sample size.
          </p>
        </div>

        <div className="bg-slate-700/30 rounded-lg p-4">
          <h5 className="text-gray-200 font-semibold mb-2 text-sm">How to Use This</h5>
          <p className="text-gray-400 text-xs leading-relaxed">
            If your expected effect might be smaller than shown, you'll need more participants.
            If budget constraints force a smaller sample, this shows what effect sizes you can
            realistically detect. Use this to set realistic expectations and make informed decisions.
          </p>
        </div>
      </div>
    </div>
  )
}

export default SensitivityCurve
