const BoxPlot = ({ data }) => {
  if (!data || data.length === 0) return null

  // Find min and max across all box plots for scaling
  const allValues = data.flatMap(d => [d.min, d.q1, d.median, d.q3, d.max, ...d.outliers])
  const globalMin = Math.min(...allValues)
  const globalMax = Math.max(...allValues)
  const range = globalMax - globalMin
  const padding = range * 0.1

  const chartMin = globalMin - padding
  const chartMax = globalMax + padding
  const chartRange = chartMax - chartMin

  // SVG dimensions
  const width = 600
  const height = 300
  const margin = { top: 20, right: 40, bottom: 60, left: 60 }
  const plotWidth = width - margin.left - margin.right
  const plotHeight = height - margin.top - margin.bottom

  // Scale function
  const yScale = (value) => {
    return plotHeight - ((value - chartMin) / chartRange) * plotHeight
  }

  // Box width
  const boxWidth = Math.min(60, plotWidth / (data.length * 2))
  const spacing = plotWidth / data.length

  return (
    <div className="bg-slate-700/50 rounded-lg p-6">
      <h4 className="text-gray-100 font-semibold mb-4">Box Plot</h4>
      <div className="flex justify-center">
        <svg width={width} height={height} className="overflow-visible">
          {/* Background grid */}
          <g transform={`translate(${margin.left}, ${margin.top})`}>
            {/* Horizontal grid lines */}
            {[0, 0.25, 0.5, 0.75, 1].map((fraction, i) => {
              const y = plotHeight * fraction
              const value = chartMax - fraction * chartRange
              return (
                <g key={i}>
                  <line
                    x1={0}
                    y1={y}
                    x2={plotWidth}
                    y2={y}
                    stroke="#475569"
                    strokeWidth={1}
                    strokeDasharray="4"
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

            {/* Box plots */}
            {data.map((box, index) => {
              const x = spacing * index + spacing / 2

              return (
                <g key={index}>
                  {/* Whiskers (vertical lines) */}
                  <line
                    x1={x}
                    y1={yScale(box.min)}
                    x2={x}
                    y2={yScale(box.max)}
                    stroke="#94a3b8"
                    strokeWidth={2}
                  />

                  {/* Min whisker cap */}
                  <line
                    x1={x - boxWidth / 4}
                    y1={yScale(box.min)}
                    x2={x + boxWidth / 4}
                    y2={yScale(box.min)}
                    stroke="#94a3b8"
                    strokeWidth={2}
                  />

                  {/* Max whisker cap */}
                  <line
                    x1={x - boxWidth / 4}
                    y1={yScale(box.max)}
                    x2={x + boxWidth / 4}
                    y2={yScale(box.max)}
                    stroke="#94a3b8"
                    strokeWidth={2}
                  />

                  {/* Box (Q1 to Q3) */}
                  <rect
                    x={x - boxWidth / 2}
                    y={yScale(box.q3)}
                    width={boxWidth}
                    height={yScale(box.q1) - yScale(box.q3)}
                    fill="#3b82f6"
                    fillOpacity={0.6}
                    stroke="#60a5fa"
                    strokeWidth={2}
                  />

                  {/* Median line */}
                  <line
                    x1={x - boxWidth / 2}
                    y1={yScale(box.median)}
                    x2={x + boxWidth / 2}
                    y2={yScale(box.median)}
                    stroke="#f1f5f9"
                    strokeWidth={3}
                  />

                  {/* Outliers */}
                  {box.outliers.map((outlier, outIndex) => (
                    <circle
                      key={outIndex}
                      cx={x}
                      cy={yScale(outlier)}
                      r={4}
                      fill="#ef4444"
                      opacity={0.7}
                    />
                  ))}

                  {/* Label */}
                  <text
                    x={x}
                    y={plotHeight + 20}
                    textAnchor="middle"
                    fill="#e2e8f0"
                    fontSize="14"
                    fontWeight="500"
                  >
                    {box.label}
                  </text>

                  {/* Stats tooltip area */}
                  <g className="group">
                    <rect
                      x={x - boxWidth / 2}
                      y={yScale(box.max)}
                      width={boxWidth}
                      height={yScale(box.min) - yScale(box.max)}
                      fill="transparent"
                      className="cursor-pointer"
                    />

                    {/* Tooltip */}
                    <g className="opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                      <rect
                        x={x + boxWidth / 2 + 10}
                        y={yScale(box.q3) - 10}
                        width="120"
                        height="110"
                        fill="#1e293b"
                        stroke="#475569"
                        strokeWidth="1"
                        rx="4"
                      />
                      <text x={x + boxWidth / 2 + 20} y={yScale(box.q3) + 10} fill="#e2e8f0" fontSize="11">
                        <tspan x={x + boxWidth / 2 + 20} dy="0">Max: {box.max.toFixed(2)}</tspan>
                        <tspan x={x + boxWidth / 2 + 20} dy="15">Q3: {box.q3.toFixed(2)}</tspan>
                        <tspan x={x + boxWidth / 2 + 20} dy="15">Median: {box.median.toFixed(2)}</tspan>
                        <tspan x={x + boxWidth / 2 + 20} dy="15">Q1: {box.q1.toFixed(2)}</tspan>
                        <tspan x={x + boxWidth / 2 + 20} dy="15">Min: {box.min.toFixed(2)}</tspan>
                        {box.outliers.length > 0 && (
                          <tspan x={x + boxWidth / 2 + 20} dy="15">Outliers: {box.outliers.length}</tspan>
                        )}
                      </text>
                    </g>
                  </g>
                </g>
              )
            })}
          </g>

          {/* Y-axis label */}
          <text
            x={15}
            y={height / 2}
            textAnchor="middle"
            fill="#94a3b8"
            fontSize="14"
            transform={`rotate(-90, 15, ${height / 2})`}
          >
            Value
          </text>
        </svg>
      </div>

      {/* Legend */}
      <div className="mt-4 flex flex-wrap gap-4 justify-center text-sm text-gray-300">
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-blue-500 opacity-60 border-2 border-blue-400"></div>
          <span>Interquartile Range (Q1-Q3)</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-8 h-0.5 bg-gray-100"></div>
          <span>Median</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 rounded-full bg-red-500 opacity-70"></div>
          <span>Outliers</span>
        </div>
      </div>
    </div>
  )
}

export default BoxPlot
