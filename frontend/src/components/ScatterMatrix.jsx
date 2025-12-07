import { useState } from 'react'
import Plot from 'react-plotly.js'
import { Grid2x2 } from 'lucide-react'

const ScatterMatrix = ({
  data,
  variableNames = null,
  colorByVariable = null,
  colorByValues = null,
  title = 'Scatter Matrix',
  showDiagonal = true,
  diagonalType = 'histogram'
}) => {
  const [diagType, setDiagType] = useState(diagonalType)
  const [showDiag, setShowDiag] = useState(showDiagonal)

  if (!data || Object.keys(data).length === 0) {
    return (
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <div className="flex items-center gap-3 mb-4">
          <Grid2x2 className="w-6 h-6 text-green-400" />
          <h3 className="text-xl font-bold text-gray-100">Scatter Matrix</h3>
        </div>
        <p className="text-gray-400">No data available for scatter matrix</p>
      </div>
    )
  }

  // Extract variable names and data
  const varNames = variableNames || Object.keys(data)
  const numVars = varNames.length

  if (numVars < 2) {
    return (
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <div className="flex items-center gap-3 mb-4">
          <Grid2x2 className="w-6 h-6 text-green-400" />
          <h3 className="text-xl font-bold text-gray-100">Scatter Matrix</h3>
        </div>
        <p className="text-gray-400">At least 2 variables required for scatter matrix</p>
      </div>
    )
  }

  // Prepare data for splom (scatter plot matrix)
  const dimensions = varNames.map(name => ({
    label: name,
    values: data[name]
  }))

  // Prepare marker colors if colorByVariable is specified
  let marker = {
    size: 4,
    line: {
      width: 0.5,
      color: 'rgba(255, 255, 255, 0.3)'
    }
  }

  if (colorByVariable && colorByValues) {
    // Create color mapping for categorical variables
    const uniqueValues = [...new Set(colorByValues)]
    const colorScale = [
      '#3b82f6', // blue
      '#ef4444', // red
      '#10b981', // green
      '#f59e0b', // amber
      '#8b5cf6', // purple
      '#ec4899', // pink
      '#14b8a6', // teal
      '#f97316'  // orange
    ]

    marker = {
      ...marker,
      color: colorByValues.map(val => {
        const index = uniqueValues.indexOf(val)
        return colorScale[index % colorScale.length]
      }),
      colorscale: 'Viridis',
      showscale: false
    }
  } else {
    marker.color = 'rgba(59, 130, 246, 0.6)'
  }

  const plotData = [{
    type: 'splom',
    dimensions,
    marker,
    diagonal: {
      visible: showDiag
    },
    showupperhalf: true,
    showlowerhalf: true,
    text: colorByVariable && colorByValues ? colorByValues : undefined,
    hovertemplate: colorByVariable ?
      '<b>%{text}</b><br>%{xaxis.title.text}: %{x}<br>%{yaxis.title.text}: %{y}<extra></extra>' :
      '%{xaxis.title.text}: %{x}<br>%{yaxis.title.text}: %{y}<extra></extra>'
  }]

  // Calculate plot height based on number of variables
  const plotHeight = Math.max(500, Math.min(900, numVars * 150))

  // Create axis layout dynamically
  const axisLayout = {}
  for (let i = 1; i <= numVars; i++) {
    const axisName = i === 1 ? 'xaxis' : `xaxis${i}`
    axisLayout[axisName] = {
      showgrid: true,
      gridcolor: 'rgba(148, 163, 184, 0.1)',
      zerolinecolor: 'rgba(148, 163, 184, 0.2)',
      tickfont: { size: 9, color: '#cbd5e1' }
    }

    const yaxisName = i === 1 ? 'yaxis' : `yaxis${i}`
    axisLayout[yaxisName] = {
      showgrid: true,
      gridcolor: 'rgba(148, 163, 184, 0.1)',
      zerolinecolor: 'rgba(148, 163, 184, 0.2)',
      tickfont: { size: 9, color: '#cbd5e1' }
    }
  }

  return (
    <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <Grid2x2 className="w-6 h-6 text-green-400" />
          <h3 className="text-xl font-bold text-gray-100">{title}</h3>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-4">
          {/* Diagonal Type Selector */}
          <div className="flex items-center gap-2">
            <label className="text-gray-300 text-sm">Diagonal:</label>
            <select
              value={diagType}
              onChange={(e) => setDiagType(e.target.value)}
              disabled={!showDiag}
              className="bg-slate-700 text-gray-100 px-3 py-1 rounded-lg text-sm border border-slate-600 focus:outline-none focus:ring-2 focus:ring-green-500 disabled:opacity-50"
            >
              <option value="histogram">Histogram</option>
              <option value="box">Box Plot</option>
            </select>
          </div>

          {/* Toggle Diagonal */}
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={showDiag}
              onChange={(e) => setShowDiag(e.target.checked)}
              className="w-4 h-4 text-green-500 bg-slate-700 border-slate-600 rounded focus:ring-green-500"
            />
            <span className="text-gray-300 text-sm">Show Diagonal</span>
          </label>
        </div>
      </div>

      {/* Info Banner */}
      <div className="mb-4 bg-blue-900/20 rounded-lg p-3 border border-blue-700/30">
        <p className="text-blue-200 text-sm">
          <strong>Scatter Matrix:</strong> Each cell shows the relationship between two variables.
          {colorByVariable && ` Colored by: ${colorByVariable}`}
        </p>
      </div>

      {/* Plot */}
      <Plot
        data={plotData}
        layout={{
          autosize: true,
          height: plotHeight,
          plot_bgcolor: 'rgba(15, 23, 42, 0.5)',
          paper_bgcolor: 'rgba(15, 23, 42, 0)',
          font: { color: '#e2e8f0', family: 'Inter, system-ui, sans-serif' },
          ...axisLayout,
          margin: { l: 80, r: 40, t: 40, b: 80 },
          hovermode: 'closest',
          dragmode: 'select'
        }}
        config={{
          displayModeBar: true,
          displaylogo: false,
          modeBarButtonsToRemove: ['lasso2d'],
          toImageButtonOptions: {
            format: 'png',
            filename: `scatter_matrix_${new Date().toISOString().slice(0, 10)}`,
            height: 1000,
            width: 1000,
            scale: 2
          }
        }}
        style={{ width: '100%' }}
        useResizeHandler={true}
      />

      {/* Summary Statistics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4">
        <div className="bg-slate-700/30 rounded-lg p-3">
          <p className="text-gray-400 text-xs">Variables</p>
          <p className="text-gray-100 font-semibold">{numVars}</p>
        </div>
        <div className="bg-slate-700/30 rounded-lg p-3">
          <p className="text-gray-400 text-xs">Scatter Plots</p>
          <p className="text-gray-100 font-semibold">{numVars * (numVars - 1)}</p>
        </div>
        <div className="bg-slate-700/30 rounded-lg p-3">
          <p className="text-gray-400 text-xs">Observations</p>
          <p className="text-gray-100 font-semibold">{data[varNames[0]]?.length || 0}</p>
        </div>
        {colorByVariable && (
          <div className="bg-slate-700/30 rounded-lg p-3">
            <p className="text-gray-400 text-xs">Color Variable</p>
            <p className="text-gray-100 font-semibold truncate">{colorByVariable}</p>
          </div>
        )}
      </div>

      {/* Usage Hint */}
      <div className="mt-4 bg-slate-700/30 rounded-lg p-3">
        <p className="text-gray-400 text-xs">
          <strong>Tip:</strong> Click and drag to select points. Use the camera icon to save the plot.
          {colorByVariable && ' Colors indicate different groups or levels.'}
        </p>
      </div>
    </div>
  )
}

export default ScatterMatrix
