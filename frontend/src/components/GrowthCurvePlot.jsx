import { TrendingUp, Info, Users } from 'lucide-react'
import { useState } from 'react'

const GrowthCurvePlot = ({ individualTrajectories, populationCurve, timeVar = "Time", responseVar = "Response" }) => {
  const [showIndividual, setShowIndividual] = useState(true)
  const [showPopulation, setShowPopulation] = useState(true)
  const [showCI, setShowCI] = useState(true)

  if (!individualTrajectories && !populationCurve) return null

  // Prepare data for plotting
  const prepareData = () => {
    let allTimePoints = []
    let allValues = []

    // Collect all data points to determine axis ranges
    if (individualTrajectories && individualTrajectories.length > 0) {
      individualTrajectories.forEach(trajectory => {
        trajectory.observed.forEach(point => {
          allTimePoints.push(point.time)
          allValues.push(point.value)
        })
      })
    }

    if (populationCurve && populationCurve.time_points) {
      allTimePoints.push(...populationCurve.time_points)
      allValues.push(...populationCurve.predicted)
      if (populationCurve.ci_lower && populationCurve.ci_upper) {
        allValues.push(...populationCurve.ci_lower, ...populationCurve.ci_upper)
      }
    }

    const minTime = Math.min(...allTimePoints)
    const maxTime = Math.max(...allTimePoints)
    const minValue = Math.min(...allValues)
    const maxValue = Math.max(...allValues)

    // Add 10% padding
    const timePadding = (maxTime - minTime) * 0.1
    const valuePadding = (maxValue - minValue) * 0.1

    return {
      minTime: minTime - timePadding,
      maxTime: maxTime + timePadding,
      minValue: minValue - valuePadding,
      maxValue: maxValue + valuePadding
    }
  }

  const { minTime, maxTime, minValue, maxValue } = prepareData()

  // SVG Spaghetti Plot
  const SpaghettiPlot = () => {
    const plotWidth = 700
    const plotHeight = 500
    const marginLeft = 70
    const marginRight = 40
    const marginTop = 40
    const marginBottom = 70

    const xScale = (val) => {
      return marginLeft + ((val - minTime) / (maxTime - minTime)) * (plotWidth - marginLeft - marginRight)
    }

    const yScale = (val) => {
      return plotHeight - marginBottom - ((val - minValue) / (maxValue - minValue)) * (plotHeight - marginTop - marginBottom)
    }

    // Generate path from points
    const generatePath = (points, accessor = 'value') => {
      if (!points || points.length === 0) return ''
      return points.map((point, i) => {
        const x = xScale(point.time)
        const y = yScale(point[accessor])
        return `${i === 0 ? 'M' : 'L'} ${x} ${y}`
      }).join(' ')
    }

    // Color palette for individual trajectories
    const colors = [
      '#8b5cf6', '#3b82f6', '#10b981', '#f59e0b', '#ef4444',
      '#ec4899', '#14b8a6', '#6366f1', '#f97316', '#84cc16'
    ]

    return (
      <svg width={plotWidth} height={plotHeight} className="bg-slate-900/30 rounded-lg">
        {/* Grid lines */}
        {[...Array(6)].map((_, i) => {
          const timeValue = minTime + (i * (maxTime - minTime)) / 5
          const valueValue = minValue + (i * (maxValue - minValue)) / 5
          const x = xScale(timeValue)
          const y = yScale(valueValue)

          return (
            <g key={i}>
              {/* Vertical grid */}
              <line
                x1={x}
                y1={marginTop}
                x2={x}
                y2={plotHeight - marginBottom}
                stroke="currentColor"
                className="text-slate-700"
                strokeWidth="1"
                opacity="0.3"
              />
              {/* Horizontal grid */}
              <line
                x1={marginLeft}
                y1={y}
                x2={plotWidth - marginRight}
                y2={y}
                stroke="currentColor"
                className="text-slate-700"
                strokeWidth="1"
                opacity="0.3"
              />
              {/* X-axis labels */}
              <text
                x={x}
                y={plotHeight - marginBottom + 25}
                fill="currentColor"
                className="text-gray-400"
                fontSize="11"
                textAnchor="middle"
              >
                {timeValue.toFixed(1)}
              </text>
              {/* Y-axis labels */}
              <text
                x={marginLeft - 15}
                y={y}
                fill="currentColor"
                className="text-gray-400"
                fontSize="11"
                textAnchor="end"
                dominantBaseline="middle"
              >
                {valueValue.toFixed(1)}
              </text>
            </g>
          )
        })}

        {/* Population CI band */}
        {showPopulation && showCI && populationCurve && populationCurve.time_points && populationCurve.ci_lower && (
          <path
            d={
              populationCurve.time_points.map((t, i) => {
                const x = xScale(t)
                const yLower = yScale(populationCurve.ci_lower[i])
                return `${i === 0 ? 'M' : 'L'} ${x} ${yLower}`
              }).join(' ') + ' ' +
              populationCurve.time_points.slice().reverse().map((t, i) => {
                const x = xScale(t)
                const idx = populationCurve.time_points.length - 1 - i
                const yUpper = yScale(populationCurve.ci_upper[idx])
                return `L ${x} ${yUpper}`
              }).join(' ') + ' Z'
            }
            fill="#f59e0b"
            opacity="0.15"
          />
        )}

        {/* Individual trajectories (spaghetti lines) */}
        {showIndividual && individualTrajectories && individualTrajectories.map((trajectory, idx) => (
          <g key={idx}>
            {/* Observed points */}
            {trajectory.observed.map((point, i) => (
              <circle
                key={`obs-${i}`}
                cx={xScale(point.time)}
                cy={yScale(point.value)}
                r="3"
                fill={colors[idx % colors.length]}
                opacity="0.6"
              />
            ))}
            {/* Predicted line */}
            <path
              d={generatePath(trajectory.predicted)}
              stroke={colors[idx % colors.length]}
              strokeWidth="1.5"
              fill="none"
              opacity="0.4"
            />
          </g>
        ))}

        {/* Population average curve */}
        {showPopulation && populationCurve && populationCurve.time_points && (
          <path
            d={populationCurve.time_points.map((t, i) => {
              const x = xScale(t)
              const y = yScale(populationCurve.predicted[i])
              return `${i === 0 ? 'M' : 'L'} ${x} ${y}`
            }).join(' ')}
            stroke="#f59e0b"
            strokeWidth="3"
            fill="none"
            opacity="1"
          />
        )}

        {/* Axes */}
        <line
          x1={marginLeft}
          y1={plotHeight - marginBottom}
          x2={plotWidth - marginRight}
          y2={plotHeight - marginBottom}
          stroke="currentColor"
          className="text-gray-500"
          strokeWidth="2"
        />
        <line
          x1={marginLeft}
          y1={marginTop}
          x2={marginLeft}
          y2={plotHeight - marginBottom}
          stroke="currentColor"
          className="text-gray-500"
          strokeWidth="2"
        />

        {/* Axis labels */}
        <text
          x={plotWidth / 2}
          y={plotHeight - 15}
          fill="currentColor"
          className="text-gray-300"
          fontSize="13"
          fontWeight="500"
          textAnchor="middle"
        >
          {timeVar}
        </text>
        <text
          x={20}
          y={plotHeight / 2}
          fill="currentColor"
          className="text-gray-300"
          fontSize="13"
          fontWeight="500"
          textAnchor="middle"
          transform={`rotate(-90, 20, ${plotHeight / 2})`}
        >
          {responseVar}
        </text>

        {/* Title */}
        <text
          x={plotWidth / 2}
          y={20}
          fill="currentColor"
          className="text-gray-200"
          fontSize="14"
          fontWeight="bold"
          textAnchor="middle"
        >
          Growth Curves: Individual Trajectories & Population Average
        </text>
      </svg>
    )
  }

  return (
    <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <TrendingUp className="w-6 h-6 text-amber-400" />
          <h3 className="text-xl font-bold text-gray-100">Growth Curve Visualization</h3>
        </div>
        <div className="flex items-center space-x-2">
          <Users className="w-5 h-5 text-gray-400" />
          <span className="text-sm text-gray-400">
            {individualTrajectories?.length || 0} subjects
          </span>
        </div>
      </div>

      <p className="text-gray-400 text-sm mb-6">
        Spaghetti plot showing individual trajectories (colored lines) and population-average growth curve (thick orange line).
      </p>

      {/* Toggle controls */}
      <div className="flex space-x-4 mb-6">
        <label className="flex items-center space-x-2 cursor-pointer">
          <input
            type="checkbox"
            checked={showIndividual}
            onChange={(e) => setShowIndividual(e.target.checked)}
            className="w-4 h-4 rounded border-gray-600 text-purple-500 focus:ring-purple-500"
          />
          <span className="text-sm text-gray-300">Show Individual Trajectories</span>
        </label>
        <label className="flex items-center space-x-2 cursor-pointer">
          <input
            type="checkbox"
            checked={showPopulation}
            onChange={(e) => setShowPopulation(e.target.checked)}
            className="w-4 h-4 rounded border-gray-600 text-amber-500 focus:ring-amber-500"
          />
          <span className="text-sm text-gray-300">Show Population Average</span>
        </label>
        <label className="flex items-center space-x-2 cursor-pointer">
          <input
            type="checkbox"
            checked={showCI}
            onChange={(e) => setShowCI(e.target.checked)}
            className="w-4 h-4 rounded border-gray-600 text-amber-500 focus:ring-amber-500"
          />
          <span className="text-sm text-gray-300">Show Confidence Band</span>
        </label>
      </div>

      {/* Spaghetti Plot */}
      <div className="flex justify-center mb-6">
        <SpaghettiPlot />
      </div>

      {/* Legend */}
      <div className="bg-slate-700/30 rounded-lg p-4">
        <h5 className="font-semibold text-gray-200 mb-3 flex items-center">
          <Info className="w-4 h-4 mr-2" />
          Interpreting Growth Curves
        </h5>
        <div className="space-y-2 text-xs text-gray-400">
          <p>
            <strong className="text-gray-300">Individual trajectories (colored):</strong> Each line shows one subject's observed data points and predicted trajectory.
            Wide spread indicates high between-subject variability.
          </p>
          <p>
            <strong className="text-gray-300">Population average (orange):</strong> Represents the typical growth pattern across all subjects.
            The confidence band shows uncertainty in the population average.
          </p>
          <p>
            <strong className="text-gray-300">Parallel lines:</strong> Subjects differ in initial status but have similar growth rates (random intercepts only).
          </p>
          <p>
            <strong className="text-gray-300">Diverging/converging lines:</strong> Subjects have different growth rates over time (random slopes).
          </p>
        </div>
      </div>
    </div>
  )
}

export default GrowthCurvePlot
