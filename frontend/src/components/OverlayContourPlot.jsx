import { useState } from 'react'
import Plot from 'react-plotly.js'
import { X, Target, MapPin, Layers, Info, CheckCircle, AlertTriangle } from 'lucide-react'
import { getPlotlyConfig } from '../utils/plotlyConfig'

const OverlayContourPlot = ({
  contourData,
  onClose,
  sweetSpot = null
}) => {
  const [showFeasibleRegion, setShowFeasibleRegion] = useState(true)
  const [visibleResponses, setVisibleResponses] = useState(
    contourData?.contours?.reduce((acc, c) => {
      acc[c.response_name] = true
      return acc
    }, {}) || {}
  )

  if (!contourData) return null

  const { grid, contours, feasible_region, sweet_spot, factors, interpretation } = contourData

  // Color schemes for different responses
  const colorSchemes = [
    [[0, '#0000ff'], [0.5, '#00ffff'], [1, '#00ff00']],  // Blue to Green
    [[0, '#ff0000'], [0.5, '#ff8800'], [1, '#ffff00']],  // Red to Yellow
    [[0, '#8800ff'], [0.5, '#ff00ff'], [1, '#ff88ff']],  // Purple to Pink
    [[0, '#00ffff'], [0.5, '#0088ff'], [1, '#0000ff']],  // Cyan to Blue
    [[0, '#00ff00'], [0.5, '#888800'], [1, '#ff8800']]   // Green to Orange
  ]

  const toggleResponse = (name) => {
    setVisibleResponses(prev => ({
      ...prev,
      [name]: !prev[name]
    }))
  }

  // Build Plotly traces
  const traces = []

  // Add contour traces for each visible response
  contours.forEach((contour, idx) => {
    if (!visibleResponses[contour.response_name]) return

    traces.push({
      type: 'contour',
      x: grid.x,
      y: grid.y,
      z: contour.Z,
      colorscale: colorSchemes[idx % colorSchemes.length],
      opacity: 0.7,
      contours: {
        coloring: 'lines',
        showlabels: true,
        labelfont: {
          size: 10,
          color: '#ffffff'
        }
      },
      line: {
        width: 2
      },
      showscale: idx === 0,
      colorbar: idx === 0 ? {
        title: {
          text: 'Response Value',
          font: { color: '#e2e8f0' }
        },
        tickfont: { color: '#e2e8f0' },
        x: 1.02
      } : undefined,
      name: contour.response_name,
      hovertemplate: `${factors[0]}: %{x:.2f}<br>${factors[1]}: %{y:.2f}<br>${contour.response_name}: %{z:.3f}<extra></extra>`
    })
  })

  // Add feasible region if available and enabled
  if (showFeasibleRegion && feasible_region && feasible_region.length > 0) {
    traces.push({
      type: 'scatter',
      mode: 'markers',
      x: feasible_region.map(p => p.x),
      y: feasible_region.map(p => p.y),
      marker: {
        size: 6,
        color: 'rgba(34, 197, 94, 0.4)',
        symbol: 'square'
      },
      name: 'Feasible Region',
      hovertemplate: `Feasible Point<br>${factors[0]}: %{x:.2f}<br>${factors[1]}: %{y:.2f}<extra></extra>`
    })
  }

  // Add sweet spot marker
  if (sweet_spot) {
    traces.push({
      type: 'scatter',
      mode: 'markers+text',
      x: [sweet_spot.x],
      y: [sweet_spot.y],
      marker: {
        size: 20,
        color: '#ffd700',
        symbol: 'star',
        line: {
          color: '#ffffff',
          width: 2
        }
      },
      text: ['Sweet Spot'],
      textposition: 'top center',
      textfont: {
        color: '#ffd700',
        size: 12,
        family: 'Arial Black'
      },
      name: 'Sweet Spot',
      hovertemplate: `<b>Sweet Spot</b><br>${factors[0]}: ${sweet_spot.x.toFixed(3)}<br>${factors[1]}: ${sweet_spot.y.toFixed(3)}<extra></extra>`
    })
  }

  const layout = {
    title: {
      text: 'Multi-Response Overlay Contour Plot',
      font: { size: 18, color: '#f1f5f9' }
    },
    xaxis: {
      title: {
        text: factors[0],
        font: { color: '#e2e8f0' }
      },
      gridcolor: '#475569',
      zerolinecolor: '#64748b',
      color: '#e2e8f0'
    },
    yaxis: {
      title: {
        text: factors[1],
        font: { color: '#e2e8f0' }
      },
      gridcolor: '#475569',
      zerolinecolor: '#64748b',
      color: '#e2e8f0',
      scaleanchor: 'x'
    },
    paper_bgcolor: '#1e293b',
    plot_bgcolor: '#0f172a',
    font: { color: '#e2e8f0' },
    showlegend: true,
    legend: {
      x: 1.15,
      y: 1,
      bgcolor: 'rgba(30, 41, 59, 0.9)',
      bordercolor: '#475569',
      borderwidth: 1,
      font: { color: '#e2e8f0' }
    },
    margin: { l: 80, r: 180, b: 80, t: 80 },
    height: 600
  }

  const config = getPlotlyConfig('overlay-contour', {
    modeBarButtonsToRemove: ['lasso2d', 'select2d']
  })

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-slate-800 rounded-2xl border border-slate-700 shadow-2xl w-full max-w-6xl max-h-[95vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-700 sticky top-0 bg-slate-800 z-10">
          <div className="flex items-center gap-3">
            <Layers className="w-6 h-6 text-indigo-400" />
            <h2 className="text-2xl font-bold text-gray-100">Multi-Response Overlay Contours</h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-gray-400" />
          </button>
        </div>

        {/* Controls */}
        <div className="p-4 border-b border-slate-700 bg-slate-800/50">
          <div className="flex flex-wrap items-center gap-4">
            {/* Response toggles */}
            <div className="flex flex-wrap gap-2">
              {contours.map((contour, idx) => (
                <button
                  key={contour.response_name}
                  onClick={() => toggleResponse(contour.response_name)}
                  className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-all text-sm ${
                    visibleResponses[contour.response_name]
                      ? 'bg-indigo-600/30 border border-indigo-500 text-indigo-200'
                      : 'bg-slate-700/50 border border-slate-600 text-gray-400'
                  }`}
                >
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{
                      backgroundColor: visibleResponses[contour.response_name]
                        ? colorSchemes[idx % colorSchemes.length][1][1]
                        : '#64748b'
                    }}
                  />
                  {contour.response_name}
                  <span className={`text-xs px-1.5 py-0.5 rounded ${
                    contour.goal === 'maximize' ? 'bg-green-900/50 text-green-300' :
                    contour.goal === 'minimize' ? 'bg-blue-900/50 text-blue-300' :
                    'bg-purple-900/50 text-purple-300'
                  }`}>
                    {contour.goal}
                  </span>
                </button>
              ))}
            </div>

            {/* Feasible region toggle */}
            <button
              onClick={() => setShowFeasibleRegion(!showFeasibleRegion)}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-all text-sm ${
                showFeasibleRegion
                  ? 'bg-green-600/30 border border-green-500 text-green-200'
                  : 'bg-slate-700/50 border border-slate-600 text-gray-400'
              }`}
            >
              <CheckCircle className="w-4 h-4" />
              Feasible Region
            </button>
          </div>
        </div>

        {/* Plot */}
        <div className="p-6">
          <div className="bg-slate-900/50 rounded-xl p-4">
            <Plot
              data={traces}
              layout={layout}
              config={config}
              style={{ width: '100%' }}
              useResizeHandler={true}
            />
          </div>
        </div>

        {/* Interpretation Panel */}
        <div className="p-6 pt-0 space-y-4">
          {/* Sweet Spot Info */}
          {sweet_spot && (
            <div className="bg-gradient-to-r from-yellow-900/30 to-amber-900/30 rounded-xl p-4 border border-yellow-700/50">
              <div className="flex items-center gap-3 mb-3">
                <Target className="w-5 h-5 text-yellow-400" />
                <h3 className="text-lg font-semibold text-yellow-200">Sweet Spot Identified</h3>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-slate-800/50 rounded-lg p-3">
                  <p className="text-gray-400 text-xs">{factors[0]}</p>
                  <p className="text-2xl font-bold text-yellow-300">{sweet_spot.x.toFixed(4)}</p>
                </div>
                <div className="bg-slate-800/50 rounded-lg p-3">
                  <p className="text-gray-400 text-xs">{factors[1]}</p>
                  <p className="text-2xl font-bold text-yellow-300">{sweet_spot.y.toFixed(4)}</p>
                </div>
              </div>
              <p className="text-yellow-100/80 text-sm mt-3">
                The sweet spot represents the center of the feasible region where all response constraints are satisfied.
              </p>
            </div>
          )}

          {/* Feasible Region Stats */}
          {interpretation && (
            <div className={`rounded-xl p-4 border ${
              interpretation.has_sweet_spot
                ? 'bg-green-900/20 border-green-700/50'
                : 'bg-orange-900/20 border-orange-700/50'
            }`}>
              <div className="flex items-center gap-3 mb-3">
                {interpretation.has_sweet_spot ? (
                  <CheckCircle className="w-5 h-5 text-green-400" />
                ) : (
                  <AlertTriangle className="w-5 h-5 text-orange-400" />
                )}
                <h3 className={`text-lg font-semibold ${
                  interpretation.has_sweet_spot ? 'text-green-200' : 'text-orange-200'
                }`}>
                  Feasibility Analysis
                </h3>
              </div>
              <div className="grid grid-cols-2 gap-4 mb-3">
                <div className="bg-slate-800/50 rounded-lg p-3">
                  <p className="text-gray-400 text-xs">Feasible Points</p>
                  <p className="text-xl font-bold text-gray-100">{interpretation.feasible_points}</p>
                </div>
                <div className="bg-slate-800/50 rounded-lg p-3">
                  <p className="text-gray-400 text-xs">Status</p>
                  <p className={`text-xl font-bold ${
                    interpretation.has_sweet_spot ? 'text-green-300' : 'text-orange-300'
                  }`}>
                    {interpretation.has_sweet_spot ? 'Solution Found' : 'No Solution'}
                  </p>
                </div>
              </div>
              <p className={`text-sm ${
                interpretation.has_sweet_spot ? 'text-green-100/80' : 'text-orange-100/80'
              }`}>
                {interpretation.recommendation}
              </p>
            </div>
          )}

          {/* Legend */}
          <div className="bg-blue-900/20 border border-blue-700/50 rounded-xl p-4">
            <div className="flex items-center gap-2 mb-3">
              <Info className="w-5 h-5 text-blue-400" />
              <h3 className="text-lg font-semibold text-blue-200">How to Interpret</h3>
            </div>
            <ul className="text-sm text-blue-100/80 space-y-2">
              <li><strong>Contour Lines:</strong> Each color represents a different response. Lines connect points with equal predicted values.</li>
              <li><strong>Feasible Region:</strong> Green squares show factor combinations where ALL response constraints are satisfied.</li>
              <li><strong>Sweet Spot:</strong> The gold star marks the center of the feasible region - your optimal compromise point.</li>
              <li><strong>Trade-offs:</strong> Where contours cross or run parallel indicates potential trade-offs between responses.</li>
            </ul>
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-3 p-6 border-t border-slate-700">
          <button
            onClick={onClose}
            className="px-6 py-2 bg-slate-700 hover:bg-slate-600 text-gray-200 rounded-lg transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  )
}

export default OverlayContourPlot
