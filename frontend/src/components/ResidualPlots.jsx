const ResidualPlots = ({ residuals, fittedValues, standardizedResiduals }) => {
  if (!residuals || residuals.length === 0) return null

  // Calculate normal quantiles for Q-Q plot
  const calculateNormalQuantiles = (data) => {
    const sorted = [...data].sort((a, b) => a - b)
    const n = sorted.length
    return sorted.map((value, i) => {
      const p = (i + 0.5) / n
      // Inverse normal CDF approximation
      const z = Math.sqrt(2) * inverseErf(2 * p - 1)
      return { theoretical: z, sample: value }
    })
  }

  // Inverse error function approximation
  const inverseErf = (x) => {
    const a = 0.147
    const b = 2 / (Math.PI * a) + Math.log(1 - x * x) / 2
    const sign = x < 0 ? -1 : 1
    return sign * Math.sqrt(Math.sqrt(b * b - Math.log(1 - x * x) / a) - b)
  }

  const qqData = calculateNormalQuantiles(standardizedResiduals)

  // Calculate histogram bins
  const calculateHistogram = (data, numBins = 15) => {
    const min = Math.min(...data)
    const max = Math.max(...data)
    const binWidth = (max - min) / numBins
    const bins = Array(numBins).fill(0)
    const binEdges = Array(numBins + 1).fill(0).map((_, i) => min + i * binWidth)

    data.forEach(value => {
      const binIndex = Math.min(Math.floor((value - min) / binWidth), numBins - 1)
      bins[binIndex]++
    })

    return { bins, binEdges, binWidth }
  }

  const histogram = calculateHistogram(standardizedResiduals)

  // Common SVG dimensions
  const width = 400
  const height = 300
  const margin = { top: 20, right: 20, bottom: 50, left: 50 }
  const plotWidth = width - margin.left - margin.right
  const plotHeight = height - margin.top - margin.bottom

  // Q-Q Plot
  const QQPlot = () => {
    const theoreticalVals = qqData.map(d => d.theoretical)
    const sampleVals = qqData.map(d => d.sample)
    const minTheoretical = Math.min(...theoreticalVals)
    const maxTheoretical = Math.max(...theoreticalVals)
    const minSample = Math.min(...sampleVals)
    const maxSample = Math.max(...sampleVals)

    const xScale = (val) => ((val - minTheoretical) / (maxTheoretical - minTheoretical)) * plotWidth
    const yScale = (val) => plotHeight - ((val - minSample) / (maxSample - minSample)) * plotHeight

    // Reference line (y = x in standardized space)
    const refLineStart = Math.max(minTheoretical, minSample)
    const refLineEnd = Math.min(maxTheoretical, maxSample)

    return (
      <svg width={width} height={height}>
        <g transform={`translate(${margin.left}, ${margin.top})`}>
          {/* Grid */}
          {[0, 0.25, 0.5, 0.75, 1].map((fraction, i) => (
            <line
              key={i}
              x1={0}
              y1={plotHeight * fraction}
              x2={plotWidth}
              y2={plotHeight * fraction}
              stroke="#475569"
              strokeWidth={1}
              strokeDasharray="2"
            />
          ))}

          {/* Reference line */}
          <line
            x1={xScale(refLineStart)}
            y1={yScale(refLineStart)}
            x2={xScale(refLineEnd)}
            y2={yScale(refLineEnd)}
            stroke="#ef4444"
            strokeWidth={2}
            strokeDasharray="4"
          />

          {/* Points */}
          {qqData.map((point, i) => (
            <circle
              key={i}
              cx={xScale(point.theoretical)}
              cy={yScale(point.sample)}
              r={3}
              fill="#3b82f6"
              opacity={0.6}
            />
          ))}

          {/* Axes */}
          <line x1={0} y1={plotHeight} x2={plotWidth} y2={plotHeight} stroke="#64748b" strokeWidth={2} />
          <line x1={0} y1={0} x2={0} y2={plotHeight} stroke="#64748b" strokeWidth={2} />

          {/* Labels */}
          <text x={plotWidth / 2} y={plotHeight + 35} textAnchor="middle" fill="#94a3b8" fontSize="12">
            Theoretical Quantiles
          </text>
          <text
            x={-plotHeight / 2}
            y={-35}
            textAnchor="middle"
            fill="#94a3b8"
            fontSize="12"
            transform={`rotate(-90, -${plotHeight / 2}, -35)`}
          >
            Sample Quantiles
          </text>
        </g>
      </svg>
    )
  }

  // Residuals vs Fitted Plot
  const ResidualsVsFittedPlot = () => {
    const minFitted = Math.min(...fittedValues)
    const maxFitted = Math.max(...fittedValues)
    const minResidual = Math.min(...residuals)
    const maxResidual = Math.max(...residuals)
    const maxAbsResidual = Math.max(Math.abs(minResidual), Math.abs(maxResidual))

    const xScale = (val) => ((val - minFitted) / (maxFitted - minFitted)) * plotWidth
    const yScale = (val) => plotHeight / 2 - (val / maxAbsResidual) * (plotHeight / 2)

    return (
      <svg width={width} height={height}>
        <g transform={`translate(${margin.left}, ${margin.top})`}>
          {/* Grid */}
          {[-1, -0.5, 0, 0.5, 1].map((fraction, i) => (
            <line
              key={i}
              x1={0}
              y1={yScale(fraction * maxAbsResidual)}
              x2={plotWidth}
              y2={yScale(fraction * maxAbsResidual)}
              stroke="#475569"
              strokeWidth={1}
              strokeDasharray="2"
            />
          ))}

          {/* Zero line */}
          <line
            x1={0}
            y1={yScale(0)}
            x2={plotWidth}
            y2={yScale(0)}
            stroke="#ef4444"
            strokeWidth={2}
          />

          {/* Points */}
          {residuals.map((residual, i) => (
            <circle
              key={i}
              cx={xScale(fittedValues[i])}
              cy={yScale(residual)}
              r={3}
              fill="#3b82f6"
              opacity={0.6}
            />
          ))}

          {/* Axes */}
          <line x1={0} y1={plotHeight} x2={plotWidth} y2={plotHeight} stroke="#64748b" strokeWidth={2} />
          <line x1={0} y1={0} x2={0} y2={plotHeight} stroke="#64748b" strokeWidth={2} />

          {/* Labels */}
          <text x={plotWidth / 2} y={plotHeight + 35} textAnchor="middle" fill="#94a3b8" fontSize="12">
            Fitted Values
          </text>
          <text
            x={-plotHeight / 2}
            y={-35}
            textAnchor="middle"
            fill="#94a3b8"
            fontSize="12"
            transform={`rotate(-90, -${plotHeight / 2}, -35)`}
          >
            Residuals
          </text>
        </g>
      </svg>
    )
  }

  // Histogram of Residuals
  const HistogramPlot = () => {
    const maxCount = Math.max(...histogram.bins)
    const xScale = (val) => ((val - histogram.binEdges[0]) / (histogram.binEdges[histogram.binEdges.length - 1] - histogram.binEdges[0])) * plotWidth
    const yScale = (count) => plotHeight - (count / maxCount) * plotHeight

    return (
      <svg width={width} height={height}>
        <g transform={`translate(${margin.left}, ${margin.top})`}>
          {/* Grid */}
          {[0, 0.25, 0.5, 0.75, 1].map((fraction, i) => (
            <line
              key={i}
              x1={0}
              y1={plotHeight * fraction}
              x2={plotWidth}
              y2={plotHeight * fraction}
              stroke="#475569"
              strokeWidth={1}
              strokeDasharray="2"
            />
          ))}

          {/* Bars */}
          {histogram.bins.map((count, i) => {
            const x = xScale(histogram.binEdges[i])
            const barWidth = xScale(histogram.binEdges[i + 1]) - x
            const barHeight = plotHeight - yScale(count)

            return (
              <rect
                key={i}
                x={x}
                y={yScale(count)}
                width={barWidth}
                height={barHeight}
                fill="#3b82f6"
                opacity={0.7}
                stroke="#2563eb"
                strokeWidth={1}
              />
            )
          })}

          {/* Axes */}
          <line x1={0} y1={plotHeight} x2={plotWidth} y2={plotHeight} stroke="#64748b" strokeWidth={2} />
          <line x1={0} y1={0} x2={0} y2={plotHeight} stroke="#64748b" strokeWidth={2} />

          {/* Labels */}
          <text x={plotWidth / 2} y={plotHeight + 35} textAnchor="middle" fill="#94a3b8" fontSize="12">
            Standardized Residuals
          </text>
          <text
            x={-plotHeight / 2}
            y={-35}
            textAnchor="middle"
            fill="#94a3b8"
            fontSize="12"
            transform={`rotate(-90, -${plotHeight / 2}, -35)`}
          >
            Frequency
          </text>
        </g>
      </svg>
    )
  }

  return (
    <div className="bg-slate-700/50 rounded-lg p-6">
      <h4 className="text-gray-100 font-semibold mb-6">Residual Diagnostic Plots</h4>
      <p className="text-gray-300 text-sm mb-6">
        These plots help verify ANOVA assumptions: normality of residuals and homogeneity of variance.
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Q-Q Plot */}
        <div className="bg-slate-800/50 rounded-lg p-4">
          <h5 className="text-gray-200 font-medium mb-3 text-center">Normal Q-Q Plot</h5>
          <p className="text-gray-400 text-xs mb-3 text-center">
            Points should follow the red line if residuals are normally distributed
          </p>
          <div className="flex justify-center">
            <QQPlot />
          </div>
        </div>

        {/* Residuals vs Fitted */}
        <div className="bg-slate-800/50 rounded-lg p-4">
          <h5 className="text-gray-200 font-medium mb-3 text-center">Residuals vs Fitted Values</h5>
          <p className="text-gray-400 text-xs mb-3 text-center">
            Points should be randomly scattered around zero line
          </p>
          <div className="flex justify-center">
            <ResidualsVsFittedPlot />
          </div>
        </div>

        {/* Histogram */}
        <div className="bg-slate-800/50 rounded-lg p-4 lg:col-span-2">
          <h5 className="text-gray-200 font-medium mb-3 text-center">Histogram of Standardized Residuals</h5>
          <p className="text-gray-400 text-xs mb-3 text-center">
            Should approximate a normal (bell-shaped) distribution
          </p>
          <div className="flex justify-center">
            <HistogramPlot />
          </div>
        </div>
      </div>

      {/* Interpretation Guide */}
      <div className="mt-6 bg-slate-800/50 rounded-lg p-4">
        <h5 className="text-gray-200 font-medium mb-2">Interpretation Guide:</h5>
        <ul className="text-gray-300 text-sm space-y-1 list-disc list-inside">
          <li><strong>Q-Q Plot:</strong> Deviations from the red line suggest non-normality</li>
          <li><strong>Residuals vs Fitted:</strong> Funnel shapes or patterns indicate unequal variances</li>
          <li><strong>Histogram:</strong> Should be symmetric and bell-shaped for normality</li>
        </ul>
      </div>
    </div>
  )
}

export default ResidualPlots
