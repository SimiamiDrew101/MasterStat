const MeansPlot = ({ data }) => {
  if (!data || Object.keys(data).length === 0) return null

  const groups = Object.keys(data)
  const means = groups.map(g => data[g].mean)
  const lowers = groups.map(g => data[g].lower)
  const uppers = groups.map(g => data[g].upper)

  const allValues = [...means, ...lowers, ...uppers]
  const minVal = Math.min(...allValues)
  const maxVal = Math.max(...allValues)
  const range = maxVal - minVal
  const padding = range * 0.15

  const chartMin = minVal - padding
  const chartMax = maxVal + padding
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

  const spacing = plotWidth / groups.length
  const pointRadius = 6

  return (
    <div className="bg-slate-700/50 rounded-lg p-6">
      <h4 className="text-gray-100 font-semibold mb-4">Group Means with 95% Confidence Intervals</h4>
      <div className="flex justify-center">
        <svg width={width} height={height} className="overflow-visible">
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
                    {value.toFixed(2)}
                  </text>
                </g>
              )
            })}

            {/* Y-axis */}
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

            {/* Plot each group */}
            {groups.map((group, index) => {
              const x = spacing * index + spacing / 2
              const mean = data[group].mean
              const lower = data[group].lower
              const upper = data[group].upper

              return (
                <g key={group}>
                  {/* Confidence interval line */}
                  <line
                    x1={x}
                    y1={yScale(lower)}
                    x2={x}
                    y2={yScale(upper)}
                    stroke="#3b82f6"
                    strokeWidth={3}
                  />

                  {/* Lower cap */}
                  <line
                    x1={x - 8}
                    y1={yScale(lower)}
                    x2={x + 8}
                    y2={yScale(lower)}
                    stroke="#3b82f6"
                    strokeWidth={3}
                  />

                  {/* Upper cap */}
                  <line
                    x1={x - 8}
                    y1={yScale(upper)}
                    x2={x + 8}
                    y2={yScale(upper)}
                    stroke="#3b82f6"
                    strokeWidth={3}
                  />

                  {/* Mean point */}
                  <circle
                    cx={x}
                    cy={yScale(mean)}
                    r={pointRadius}
                    fill="#10b981"
                    stroke="#059669"
                    strokeWidth={2}
                  />

                  {/* Group label */}
                  <text
                    x={x}
                    y={plotHeight + 20}
                    textAnchor="middle"
                    fill="#e2e8f0"
                    fontSize="13"
                    fontWeight="500"
                  >
                    {group}
                  </text>

                  {/* Hover tooltip */}
                  <g className="group">
                    <rect
                      x={x - 15}
                      y={yScale(upper) - 10}
                      width={30}
                      height={yScale(lower) - yScale(upper) + 20}
                      fill="transparent"
                      className="cursor-pointer"
                    />

                    <g className="opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                      <rect
                        x={x + 15}
                        y={yScale(mean) - 30}
                        width="110"
                        height="65"
                        fill="#1e293b"
                        stroke="#475569"
                        strokeWidth="1"
                        rx="4"
                      />
                      <text x={x + 25} y={yScale(mean) - 10} fill="#e2e8f0" fontSize="11">
                        <tspan x={x + 25} dy="0">Mean: {mean.toFixed(3)}</tspan>
                        <tspan x={x + 25} dy="15">95% CI:</tspan>
                        <tspan x={x + 25} dy="15">[{lower.toFixed(3)},</tspan>
                        <tspan x={x + 25} dy="15"> {upper.toFixed(3)}]</tspan>
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
            Mean Value
          </text>
        </svg>
      </div>

      {/* Legend */}
      <div className="mt-4 flex flex-wrap gap-4 justify-center text-sm text-gray-300">
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 rounded-full bg-green-500 border-2 border-green-600"></div>
          <span>Group Mean</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-8 h-1 bg-blue-500"></div>
          <span>95% Confidence Interval</span>
        </div>
      </div>
    </div>
  )
}

export default MeansPlot
