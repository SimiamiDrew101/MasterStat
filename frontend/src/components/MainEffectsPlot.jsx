const MainEffectsPlot = ({ data, title = "Main Effects Plot" }) => {
  if (!data || Object.keys(data).length === 0) return null

  const factors = Object.keys(data)
  const numFactors = factors.length

  // Calculate layout
  const plotsPerRow = Math.min(3, numFactors)
  const numRows = Math.ceil(numFactors / plotsPerRow)

  // Individual plot dimensions
  const plotWidth = 220
  const plotHeight = 200
  const margin = { top: 30, right: 20, bottom: 50, left: 50 }
  const innerWidth = plotWidth - margin.left - margin.right
  const innerHeight = plotHeight - margin.top - margin.bottom

  // Calculate global min/max for consistent y-scale
  const allMeans = factors.flatMap(f => data[f].means)
  const globalMin = Math.min(...allMeans)
  const globalMax = Math.max(...allMeans)
  const range = globalMax - globalMin
  const padding = range * 0.1
  const yMin = globalMin - padding
  const yMax = globalMax + padding
  const yRange = yMax - yMin

  // Scale function for y-axis
  const yScale = (value) => innerHeight - ((value - yMin) / yRange) * innerHeight

  return (
    <div className="bg-slate-700/50 rounded-lg p-6">
      <h4 className="text-gray-100 font-semibold mb-4">{title}</h4>
      <p className="text-gray-300 text-sm mb-4">
        Steeper slopes indicate larger main effects. Horizontal lines indicate no effect.
      </p>

      <div className="flex justify-center">
        <div className="grid" style={{
          gridTemplateColumns: `repeat(${plotsPerRow}, ${plotWidth}px)`,
          gap: '20px'
        }}>
          {factors.map((factor, factorIndex) => {
            const levels = data[factor].levels
            const means = data[factor].means
            const numLevels = levels.length

            // X-scale for this subplot
            const xScale = (index) => (index / (numLevels - 1)) * innerWidth

            return (
              <svg key={factorIndex} width={plotWidth} height={plotHeight}>
                <g transform={`translate(${margin.left}, ${margin.top})`}>
                  {/* Grid lines */}
                  {[0, 0.25, 0.5, 0.75, 1].map((fraction, i) => (
                    <line
                      key={i}
                      x1={0}
                      y1={innerHeight * fraction}
                      x2={innerWidth}
                      y2={innerHeight * fraction}
                      stroke="#475569"
                      strokeWidth={1}
                      strokeDasharray="2"
                      opacity={0.3}
                    />
                  ))}

                  {/* Axes */}
                  <line
                    x1={0}
                    y1={innerHeight}
                    x2={innerWidth}
                    y2={innerHeight}
                    stroke="#64748b"
                    strokeWidth={2}
                  />
                  <line
                    x1={0}
                    y1={0}
                    x2={0}
                    y2={innerHeight}
                    stroke="#64748b"
                    strokeWidth={2}
                  />

                  {/* Line connecting means */}
                  <path
                    d={means.map((mean, i) => {
                      const x = xScale(i)
                      const y = yScale(mean)
                      return `${i === 0 ? 'M' : 'L'} ${x} ${y}`
                    }).join(' ')}
                    fill="none"
                    stroke="#10b981"
                    strokeWidth={3}
                    strokeLinecap="round"
                  />

                  {/* Points */}
                  {means.map((mean, i) => (
                    <g key={i}>
                      <circle
                        cx={xScale(i)}
                        cy={yScale(mean)}
                        r={6}
                        fill="#10b981"
                        stroke="#047857"
                        strokeWidth={2}
                      />

                      {/* Value label */}
                      <text
                        x={xScale(i)}
                        y={yScale(mean) - 12}
                        textAnchor="middle"
                        fill="#e2e8f0"
                        fontSize="11"
                        fontWeight="600"
                      >
                        {mean.toFixed(2)}
                      </text>
                    </g>
                  ))}

                  {/* X-axis labels (levels) */}
                  {levels.map((level, i) => (
                    <text
                      key={i}
                      x={xScale(i)}
                      y={innerHeight + 15}
                      textAnchor="middle"
                      fill="#e2e8f0"
                      fontSize="12"
                    >
                      {level}
                    </text>
                  ))}

                  {/* Y-axis labels */}
                  {[0, 0.5, 1].map((fraction, i) => {
                    const value = yMax - fraction * yRange
                    return (
                      <text
                        key={i}
                        x={-8}
                        y={innerHeight * fraction + 4}
                        textAnchor="end"
                        fill="#94a3b8"
                        fontSize="11"
                      >
                        {value.toFixed(1)}
                      </text>
                    )
                  })}

                  {/* Factor name as title */}
                  <text
                    x={innerWidth / 2}
                    y={-10}
                    textAnchor="middle"
                    fill="#e2e8f0"
                    fontSize="14"
                    fontWeight="600"
                  >
                    {factor}
                  </text>

                  {/* Y-axis label */}
                  <text
                    x={innerWidth / 2}
                    y={innerHeight + 35}
                    textAnchor="middle"
                    fill="#94a3b8"
                    fontSize="11"
                  >
                    {factor} Level
                  </text>
                </g>
              </svg>
            )
          })}
        </div>
      </div>

      <div className="mt-4 bg-slate-800/50 rounded-lg p-4">
        <p className="text-gray-300 text-sm">
          <strong className="text-gray-100">Interpretation:</strong> Each plot shows how the mean response changes
          across levels of one factor. A steep slope indicates that factor has a large effect on the response.
          Parallel lines across subplots suggest factors act independently (no interaction).
        </p>
      </div>
    </div>
  )
}

export default MainEffectsPlot
