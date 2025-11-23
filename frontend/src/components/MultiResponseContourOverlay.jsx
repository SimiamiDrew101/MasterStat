import { useState } from 'react'
import Plot from 'react-plotly.js'
import { Eye, EyeOff, Layers, Grid, Maximize2 } from 'lucide-react'

const MultiResponseContourOverlay = ({
  surfacesData,
  responseConfigs,
  factor1,
  factor2,
  experimentalData = null,
  viewMode = 'side-by-side' // 'side-by-side', 'overlay', 'single'
}) => {
  const [localViewMode, setLocalViewMode] = useState(viewMode)
  const [visibleResponses, setVisibleResponses] = useState(
    Object.keys(responseConfigs).reduce((acc, name) => {
      acc[name] = responseConfigs[name].visible !== false
      return acc
    }, {})
  )
  const [selectedResponse, setSelectedResponse] = useState(Object.keys(responseConfigs)[0])

  // Color schemes for different responses
  const colorSchemes = ['Viridis', 'Plasma', 'Inferno', 'Cividis', 'Turbo']

  // Helper: Convert surface data to Plotly contour format
  const convertToContourData = (surfacePoints) => {
    const xValues = [...new Set(surfacePoints.map(d => d.x))].sort((a, b) => a - b)
    const yValues = [...new Set(surfacePoints.map(d => d.y))].sort((a, b) => a - b)

    const zGrid = []
    for (let i = 0; i < yValues.length; i++) {
      zGrid[i] = []
      for (let j = 0; j < xValues.length; j++) {
        const point = surfacePoints.find(d => d.x === xValues[j] && d.y === yValues[i])
        zGrid[i][j] = point ? point.z : 0
      }
    }

    return { xValues, yValues, zGrid }
  }

  // Generate single contour plot
  const createSinglePlot = (responseName, surfacePoints, config) => {
    const { xValues, yValues, zGrid } = convertToContourData(surfacePoints)

    const traces = [{
      type: 'contour',
      x: xValues,
      y: yValues,
      z: zGrid,
      colorscale: config.colorScale || 'Viridis',
      contours: {
        coloring: 'heatmap',
        showlabels: true,
        labelfont: {
          size: 10,
          color: '#ffffff'
        }
      },
      colorbar: {
        title: {
          text: responseName,
          font: { color: '#e2e8f0' }
        },
        tickfont: { color: '#e2e8f0' }
      },
      hovertemplate: `${factor1}: %{x:.2f}<br>${factor2}: %{y:.2f}<br>${responseName}: %{z:.4f}<extra></extra>`
    }]

    // Add experimental data points if available
    if (experimentalData && experimentalData[responseName]) {
      traces.push({
        type: 'scatter',
        mode: 'markers',
        x: experimentalData[responseName].map(d => d[factor1]),
        y: experimentalData[responseName].map(d => d[factor2]),
        marker: {
          size: 10,
          color: '#ffffff',
          line: { color: '#000000', width: 2 }
        },
        name: 'Experimental Data',
        hovertemplate: `${factor1}: %{x:.2f}<br>${factor2}: %{y:.2f}<extra></extra>`
      })
    }

    const layout = {
      title: {
        text: responseName,
        font: { size: 16, color: '#f1f5f9' }
      },
      xaxis: {
        title: factor1,
        gridcolor: '#475569',
        color: '#e2e8f0'
      },
      yaxis: {
        title: factor2,
        gridcolor: '#475569',
        color: '#e2e8f0'
      },
      paper_bgcolor: '#334155',
      plot_bgcolor: '#1e293b',
      font: { color: '#e2e8f0' },
      margin: { l: 60, r: 100, b: 60, t: 60 },
      height: 400
    }

    return { data: traces, layout }
  }

  // Generate overlay plot
  const createOverlayPlot = () => {
    const traces = []
    const responseNames = Object.keys(surfacesData).filter(name => visibleResponses[name])

    responseNames.forEach((responseName, index) => {
      const surfacePoints = surfacesData[responseName]
      const config = responseConfigs[responseName]
      const { xValues, yValues, zGrid } = convertToContourData(surfacePoints)

      traces.push({
        type: 'contour',
        x: xValues,
        y: yValues,
        z: zGrid,
        colorscale: config.colorScale || colorSchemes[index % colorSchemes.length],
        opacity: config.opacity || 0.7,
        contours: {
          coloring: 'lines',
          showlabels: true,
          labelfont: {
            size: 9,
            color: '#ffffff'
          }
        },
        showscale: index === 0,
        colorbar: index === 0 ? {
          title: {
            text: 'Value',
            font: { color: '#e2e8f0' }
          },
          tickfont: { color: '#e2e8f0' },
          x: 1.02
        } : undefined,
        name: responseName,
        hovertemplate: `${factor1}: %{x:.2f}<br>${factor2}: %{y:.2f}<br>${responseName}: %{z:.4f}<extra></extra>`
      })
    })

    const layout = {
      title: {
        text: 'Multi-Response Overlay',
        font: { size: 18, color: '#f1f5f9' }
      },
      xaxis: {
        title: factor1,
        gridcolor: '#475569',
        color: '#e2e8f0'
      },
      yaxis: {
        title: factor2,
        gridcolor: '#475569',
        color: '#e2e8f0'
      },
      paper_bgcolor: '#334155',
      plot_bgcolor: '#1e293b',
      font: { color: '#e2e8f0' },
      margin: { l: 60, r: 120, b: 60, t: 80 },
      height: 500,
      showlegend: true,
      legend: {
        x: 1.05,
        y: 1,
        bgcolor: '#334155',
        bordercolor: '#475569',
        borderwidth: 1,
        font: { color: '#e2e8f0' }
      }
    }

    return { data: traces, layout }
  }

  const config = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d']
  }

  const toggleResponseVisibility = (responseName) => {
    setVisibleResponses(prev => ({
      ...prev,
      [responseName]: !prev[responseName]
    }))
  }

  const responseNames = Object.keys(surfacesData)

  return (
    <div className="space-y-6">
      {/* View Mode Selector */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl p-4 border border-slate-700/50">
        <div className="flex flex-wrap items-center justify-between gap-4">
          {/* View Mode Buttons */}
          <div className="flex gap-2">
            <button
              onClick={() => setLocalViewMode('side-by-side')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                localViewMode === 'side-by-side'
                  ? 'bg-indigo-600 text-white shadow-lg'
                  : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
              }`}
            >
              <Grid className="w-4 h-4" />
              Side-by-Side
            </button>
            <button
              onClick={() => setLocalViewMode('overlay')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                localViewMode === 'overlay'
                  ? 'bg-indigo-600 text-white shadow-lg'
                  : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
              }`}
            >
              <Layers className="w-4 h-4" />
              Overlay
            </button>
            <button
              onClick={() => setLocalViewMode('single')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                localViewMode === 'single'
                  ? 'bg-indigo-600 text-white shadow-lg'
                  : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
              }`}
            >
              <Maximize2 className="w-4 h-4" />
              Single
            </button>
          </div>

          {/* Response Toggle Controls (for overlay mode) */}
          {localViewMode === 'overlay' && (
            <div className="flex flex-wrap gap-2">
              {responseNames.map((name) => (
                <button
                  key={name}
                  onClick={() => toggleResponseVisibility(name)}
                  className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-all ${
                    visibleResponses[name]
                      ? 'bg-green-600/20 border border-green-600 text-green-300'
                      : 'bg-slate-700/50 border border-slate-600 text-gray-400'
                  }`}
                >
                  {visibleResponses[name] ? (
                    <Eye className="w-4 h-4" />
                  ) : (
                    <EyeOff className="w-4 h-4" />
                  )}
                  <span className="font-mono text-sm">{name}</span>
                </button>
              ))}
            </div>
          )}

          {/* Single Response Selector */}
          {localViewMode === 'single' && (
            <select
              value={selectedResponse}
              onChange={(e) => setSelectedResponse(e.target.value)}
              className="px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-gray-100 focus:outline-none focus:border-indigo-500"
            >
              {responseNames.map(name => (
                <option key={name} value={name}>{name}</option>
              ))}
            </select>
          )}
        </div>
      </div>

      {/* Visualization Area */}
      <div className="bg-slate-800/30 backdrop-blur-lg rounded-xl border border-slate-700/50 p-6">
        {localViewMode === 'side-by-side' && (
          <div className={`grid ${responseNames.length === 2 ? 'grid-cols-2' : responseNames.length === 3 ? 'grid-cols-3' : 'grid-cols-2'} gap-4`}>
            {responseNames.map((responseName) => {
              const plot = createSinglePlot(
                responseName,
                surfacesData[responseName],
                responseConfigs[responseName]
              )
              return (
                <div key={responseName} className="bg-slate-700/30 rounded-lg p-2">
                  <Plot
                    data={plot.data}
                    layout={plot.layout}
                    config={config}
                    style={{ width: '100%' }}
                  />
                </div>
              )
            })}
          </div>
        )}

        {localViewMode === 'overlay' && (
          <div className="bg-slate-700/30 rounded-lg p-2">
            {(() => {
              const plot = createOverlayPlot()
              return (
                <Plot
                  data={plot.data}
                  layout={plot.layout}
                  config={config}
                  style={{ width: '100%' }}
                />
              )
            })()}
          </div>
        )}

        {localViewMode === 'single' && (
          <div className="bg-slate-700/30 rounded-lg p-2">
            {(() => {
              const plot = createSinglePlot(
                selectedResponse,
                surfacesData[selectedResponse],
                responseConfigs[selectedResponse]
              )
              return (
                <Plot
                  data={plot.data}
                  layout={plot.layout}
                  config={config}
                  style={{ width: '100%' }}
                />
              )
            })()}
          </div>
        )}
      </div>

      {/* Legend/Info */}
      <div className="bg-blue-900/20 border border-blue-700/50 rounded-lg p-4">
        <h4 className="text-blue-300 font-semibold mb-2">Multi-Response Visualization</h4>
        <ul className="text-sm text-blue-100 space-y-1">
          <li><strong>Side-by-Side:</strong> Compare responses individually with full detail</li>
          <li><strong>Overlay:</strong> See correlations and conflicts between responses</li>
          <li><strong>Single:</strong> Focus on one response at a time</li>
          <li><strong>Tip:</strong> Use overlay mode to identify trade-off regions for multi-objective optimization</li>
        </ul>
      </div>
    </div>
  )
}

export default MultiResponseContourOverlay
