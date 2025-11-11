import { useRef } from 'react'
import { Download } from 'lucide-react'
import { exportSvgToPng } from '../utils/exportChart'

const ContourPlot = ({ surfaceData, factor1, factor2, responseName }) => {
  const svgRef = useRef(null)

  if (!surfaceData || surfaceData.length === 0) return null

  // SVG dimensions
  const width = 700
  const height = 700
  const margin = { top: 60, right: 100, bottom: 70, left: 70 }
  const plotWidth = width - margin.left - margin.right
  const plotHeight = height - margin.top - margin.bottom

  // Extract unique x and y values
  const xValues = [...new Set(surfaceData.map(d => d.x))].sort((a, b) => a - b)
  const yValues = [...new Set(surfaceData.map(d => d.y))].sort((a, b) => a - b)

  // Create 2D grid of z values
  const zGrid = []
  for (let i = 0; i < yValues.length; i++) {
    zGrid[i] = []
    for (let j = 0; j < xValues.length; j++) {
      const point = surfaceData.find(d => d.x === xValues[j] && d.y === yValues[i])
      zGrid[i][j] = point ? point.z : 0
    }
  }

  // Find min/max for z values
  const allZ = surfaceData.map(d => d.z)
  const zMin = Math.min(...allZ)
  const zMax = Math.max(...allZ)

  // Generate contour levels
  const numLevels = 10
  const contourLevels = []
  for (let i = 0; i <= numLevels; i++) {
    contourLevels.push(zMin + (i / numLevels) * (zMax - zMin))
  }

  // Color scale
  const getColor = (value) => {
    const normalized = (value - zMin) / (zMax - zMin)
    // Blue to red gradient
    const r = Math.round(255 * normalized)
    const b = Math.round(255 * (1 - normalized))
    const g = Math.round(100 * (1 - Math.abs(2 * normalized - 1)))
    return `rgb(${r}, ${g}, ${b})`
  }

  // Scale functions
  const xScale = (val) => ((val - xValues[0]) / (xValues[xValues.length - 1] - xValues[0])) * plotWidth
  const yScale = (val) => plotHeight - ((val - yValues[0]) / (yValues[yValues.length - 1] - yValues[0])) * plotHeight

  return (
    <div className="bg-slate-700/50 rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-gray-100 font-semibold">Contour Plot</h4>
        <button
          type="button"
          onClick={() => {
            if (svgRef.current) {
              exportSvgToPng(svgRef.current, `contour-plot-${new Date().toISOString().split('T')[0]}`)
            }
          }}
          className="px-3 py-2 rounded-lg text-sm font-medium bg-slate-600/50 text-gray-300 hover:bg-slate-600 transition-all flex items-center space-x-2"
          title="Export as PNG"
        >
          <Download className="w-4 h-4" />
          <span>Export PNG</span>
        </button>
      </div>

      <div className="flex justify-center">
        <svg ref={svgRef} width={width} height={height}>
          <g transform={`translate(${margin.left}, ${margin.top})`}>
            {/* Draw filled contours */}
            {yValues.map((yVal, i) => {
              if (i >= yValues.length - 1) return null
              return xValues.map((xVal, j) => {
                if (j >= xValues.length - 1) return null

                const z1 = zGrid[i][j]
                const z2 = zGrid[i][j + 1]
                const z3 = zGrid[i + 1][j + 1]
                const z4 = zGrid[i + 1][j]
                const avgZ = (z1 + z2 + z3 + z4) / 4

                const x1 = xScale(xVal)
                const y1 = yScale(yVal)
                const x2 = xScale(xValues[j + 1])
                const y2 = yScale(yValues[i + 1])

                return (
                  <rect
                    key={`${i}-${j}`}
                    x={x1}
                    y={y1}
                    width={x2 - x1}
                    height={y2 - y1}
                    fill={getColor(avgZ)}
                    opacity={0.7}
                  />
                )
              })
            })}

            {/* Draw contour lines */}
            {contourLevels.slice(1, -1).map((level, levelIdx) => {
              // Simple contour line drawing (approximate)
              const paths = []
              for (let i = 0; i < yValues.length - 1; i++) {
                for (let j = 0; j < xValues.length - 1; j++) {
                  const z1 = zGrid[i][j]
                  const z2 = zGrid[i][j + 1]
                  const z3 = zGrid[i + 1][j + 1]
                  const z4 = zGrid[i + 1][j]

                  // Check if contour passes through this cell
                  if (
                    (z1 <= level && (z2 >= level || z3 >= level || z4 >= level)) ||
                    (z1 >= level && (z2 <= level || z3 <= level || z4 <= level))
                  ) {
                    const x = xScale((xValues[j] + xValues[j + 1]) / 2)
                    const y = yScale((yValues[i] + yValues[i + 1]) / 2)
                    paths.push({ x, y })
                  }
                }
              }

              return paths.map((point, idx) => (
                <circle
                  key={`${levelIdx}-${idx}`}
                  cx={point.x}
                  cy={point.y}
                  r={1}
                  fill="#fff"
                  opacity={0.5}
                />
              ))
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

            {/* X-axis labels */}
            {[0, 0.25, 0.5, 0.75, 1].map((frac, i) => {
              const xVal = xValues[0] + frac * (xValues[xValues.length - 1] - xValues[0])
              const x = frac * plotWidth
              return (
                <g key={i}>
                  <line
                    x1={x}
                    y1={plotHeight}
                    x2={x}
                    y2={plotHeight + 5}
                    stroke="#94a3b8"
                    strokeWidth={1}
                  />
                  <text
                    x={x}
                    y={plotHeight + 20}
                    textAnchor="middle"
                    fill="#94a3b8"
                    fontSize="12"
                  >
                    {xVal.toFixed(1)}
                  </text>
                </g>
              )
            })}

            {/* Y-axis labels */}
            {[0, 0.25, 0.5, 0.75, 1].map((frac, i) => {
              const yVal = yValues[0] + frac * (yValues[yValues.length - 1] - yValues[0])
              const y = plotHeight - frac * plotHeight
              return (
                <g key={i}>
                  <line
                    x1={0}
                    y1={y}
                    x2={-5}
                    y2={y}
                    stroke="#94a3b8"
                    strokeWidth={1}
                  />
                  <text
                    x={-10}
                    y={y + 4}
                    textAnchor="end"
                    fill="#94a3b8"
                    fontSize="12"
                  >
                    {yVal.toFixed(1)}
                  </text>
                </g>
              )
            })}

            {/* Axis labels */}
            <text
              x={plotWidth / 2}
              y={plotHeight + 50}
              textAnchor="middle"
              fill="#e2e8f0"
              fontSize="14"
              fontWeight="600"
            >
              {factor1}
            </text>
            <text
              x={-plotHeight / 2}
              y={-45}
              textAnchor="middle"
              fill="#e2e8f0"
              fontSize="14"
              fontWeight="600"
              transform={`rotate(-90, -${plotHeight / 2}, -45)`}
            >
              {factor2}
            </text>

            {/* Title */}
            <text
              x={plotWidth / 2}
              y={-30}
              textAnchor="middle"
              fill="#f1f5f9"
              fontSize="16"
              fontWeight="600"
            >
              {responseName} Contour Plot
            </text>

            {/* Color legend */}
            <g transform={`translate(${plotWidth + 20}, 0)`}>
              <text
                x={0}
                y={-10}
                fill="#e2e8f0"
                fontSize="12"
                fontWeight="600"
              >
                {responseName}
              </text>
              {contourLevels.map((level, i) => {
                const y = (i / (contourLevels.length - 1)) * plotHeight
                return (
                  <g key={i}>
                    <rect
                      x={0}
                      y={y}
                      width={30}
                      height={plotHeight / (contourLevels.length - 1)}
                      fill={getColor(level)}
                    />
                    {i % 2 === 0 && (
                      <text
                        x={35}
                        y={y + 5}
                        fill="#94a3b8"
                        fontSize="10"
                      >
                        {level.toFixed(1)}
                      </text>
                    )}
                  </g>
                )
              })}
            </g>
          </g>
        </svg>
      </div>

      <div className="mt-4 bg-slate-800/50 rounded-lg p-4">
        <p className="text-gray-300 text-sm">
          <strong className="text-gray-100">Interpretation:</strong> Contour lines connect points with equal response values. The color gradient shows the response surface, with red indicating higher values and blue indicating lower values. Closely spaced contours indicate steep slopes in the response surface.
        </p>
      </div>
    </div>
  )
}

export default ContourPlot
