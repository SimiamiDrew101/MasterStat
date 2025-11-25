import { useState } from 'react'
import Plot from 'react-plotly.js'
import { Info, Eye, Layers, MapPin } from 'lucide-react'
import {
  generateDesignPreview,
  getDesignStatistics,
  getDesignCoverageInfo,
  getPointTypeLegend
} from '../utils/designPreview'

const DesignPreview = ({ wizardData }) => {
  const { selectedDesign, nFactors, factorNames } = wizardData
  const [showLegend, setShowLegend] = useState(true)

  if (!selectedDesign) {
    return (
      <div className="bg-yellow-900/20 border border-yellow-700/50 rounded-lg p-6 text-center">
        <Info className="w-12 h-12 text-yellow-400 mx-auto mb-3" />
        <p className="text-yellow-200 font-semibold">No design selected</p>
        <p className="text-yellow-300 text-sm mt-1">
          Select a design to see the preview visualization
        </p>
      </div>
    )
  }

  const previewData = generateDesignPreview(selectedDesign.type, nFactors)
  const stats = getDesignStatistics(selectedDesign.type, nFactors)
  const coverageInfo = getDesignCoverageInfo(selectedDesign.type)
  const legend = getPointTypeLegend()

  // Get factor names or use defaults
  const getFactorName = (index) => {
    return factorNames[index] || `Factor ${index + 1}`
  }

  // Render 2D scatter plot for 2 factors
  const render2DPlot = () => {
    // Group points by type for separate traces
    const pointsByType = {}
    previewData.forEach(point => {
      if (!pointsByType[point.type]) {
        pointsByType[point.type] = []
      }
      pointsByType[point.type].push(point)
    })

    const traces = Object.entries(pointsByType).map(([type, points]) => ({
      x: points.map(p => p.x),
      y: points.map(p => p.y),
      mode: 'markers',
      type: 'scatter',
      name: type,
      marker: {
        size: 12,
        color: points[0].color,
        symbol: 'circle',
        line: {
          color: '#1e293b',
          width: 2
        }
      },
      hovertemplate: `<b>${type} Point</b><br>` +
        `${getFactorName(0)}: %{x:.2f}<br>` +
        `${getFactorName(1)}: %{y:.2f}<br>` +
        '<extra></extra>'
    }))

    const layout = {
      title: {
        text: `${stats.designFamily} - Design Space Coverage`,
        font: { size: 16, color: '#f1f5f9' }
      },
      xaxis: {
        title: getFactorName(0),
        range: [-2, 2],
        gridcolor: '#334155',
        zerolinecolor: '#475569',
        color: '#cbd5e1'
      },
      yaxis: {
        title: getFactorName(1),
        range: [-2, 2],
        gridcolor: '#334155',
        zerolinecolor: '#475569',
        color: '#cbd5e1'
      },
      plot_bgcolor: '#0f172a',
      paper_bgcolor: '#1e293b',
      font: { color: '#e2e8f0' },
      showlegend: showLegend,
      legend: {
        x: 1.02,
        y: 1,
        bgcolor: 'rgba(30, 41, 59, 0.9)',
        bordercolor: '#475569',
        borderwidth: 1
      },
      margin: { l: 60, r: 150, t: 50, b: 60 }
    }

    return (
      <Plot
        data={traces}
        layout={layout}
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: '100%', height: '500px' }}
      />
    )
  }

  // Render 3D scatter plot for 3 factors
  const render3DPlot = () => {
    const pointsByType = {}
    previewData.forEach(point => {
      if (!pointsByType[point.type]) {
        pointsByType[point.type] = []
      }
      pointsByType[point.type].push(point)
    })

    const traces = Object.entries(pointsByType).map(([type, points]) => ({
      x: points.map(p => p.x),
      y: points.map(p => p.y),
      z: points.map(p => p.z),
      mode: 'markers',
      type: 'scatter3d',
      name: type,
      marker: {
        size: 8,
        color: points[0].color,
        symbol: 'circle',
        line: {
          color: '#1e293b',
          width: 2
        }
      },
      hovertemplate: `<b>${type} Point</b><br>` +
        `${getFactorName(0)}: %{x:.2f}<br>` +
        `${getFactorName(1)}: %{y:.2f}<br>` +
        `${getFactorName(2)}: %{z:.2f}<br>` +
        '<extra></extra>'
    }))

    const layout = {
      title: {
        text: `${stats.designFamily} - 3D Design Space`,
        font: { size: 16, color: '#f1f5f9' }
      },
      scene: {
        xaxis: {
          title: getFactorName(0),
          range: [-2, 2],
          gridcolor: '#334155',
          color: '#cbd5e1'
        },
        yaxis: {
          title: getFactorName(1),
          range: [-2, 2],
          gridcolor: '#334155',
          color: '#cbd5e1'
        },
        zaxis: {
          title: getFactorName(2),
          range: [-2, 2],
          gridcolor: '#334155',
          color: '#cbd5e1'
        },
        bgcolor: '#0f172a'
      },
      plot_bgcolor: '#0f172a',
      paper_bgcolor: '#1e293b',
      font: { color: '#e2e8f0' },
      showlegend: showLegend,
      legend: {
        x: 1.02,
        y: 1,
        bgcolor: 'rgba(30, 41, 59, 0.9)',
        bordercolor: '#475569',
        borderwidth: 1
      },
      margin: { l: 0, r: 150, t: 50, b: 0 }
    }

    return (
      <Plot
        data={traces}
        layout={layout}
        config={{ responsive: true, displayModeBar: true }}
        style={{ width: '100%', height: '600px' }}
      />
    )
  }

  // Render multiple 2D projections for 4+ factors
  const renderMultiPlots = () => {
    return (
      <div className="space-y-4">
        <div className="bg-blue-900/20 border border-blue-700/50 rounded-lg p-4">
          <Info className="w-5 h-5 text-blue-400 inline mr-2" />
          <span className="text-blue-200 text-sm">
            For {nFactors} factors, showing 2D projections of the design space.
            Each plot shows how two factors interact in the design.
          </span>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {previewData.slice(0, 4).map((projection, idx) => {
            const pointsByType = {}
            projection.points.forEach(point => {
              if (!pointsByType[point.type]) {
                pointsByType[point.type] = []
              }
              pointsByType[point.type].push(point)
            })

            const traces = Object.entries(pointsByType).map(([type, points]) => ({
              x: points.map(p => p.x),
              y: points.map(p => p.y),
              mode: 'markers',
              type: 'scatter',
              name: type,
              marker: {
                size: 10,
                color: points[0].color,
                line: { color: '#1e293b', width: 2 }
              },
              showlegend: idx === 0
            }))

            const layout = {
              title: {
                text: `${projection.factorName1} vs ${projection.factorName2}`,
                font: { size: 14, color: '#f1f5f9' }
              },
              xaxis: {
                title: projection.factorName1,
                range: [-1.5, 1.5],
                gridcolor: '#334155',
                color: '#cbd5e1'
              },
              yaxis: {
                title: projection.factorName2,
                range: [-1.5, 1.5],
                gridcolor: '#334155',
                color: '#cbd5e1'
              },
              plot_bgcolor: '#0f172a',
              paper_bgcolor: '#1e293b',
              font: { color: '#e2e8f0' },
              showlegend: idx === 0 && showLegend,
              margin: { l: 50, r: 20, t: 40, b: 50 },
              height: 350
            }

            return (
              <div key={idx} className="bg-slate-800/50 rounded-lg p-2">
                <Plot
                  data={traces}
                  layout={layout}
                  config={{ responsive: true, displayModeBar: false }}
                  style={{ width: '100%' }}
                />
              </div>
            )
          })}
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-2xl font-bold text-gray-100 flex items-center gap-2">
            <Eye className="w-7 h-7 text-blue-400" />
            Design Space Preview
          </h3>
          <p className="text-gray-300 text-sm mt-1">
            Visualize your experimental design before generation
          </p>
        </div>
        <button
          onClick={() => setShowLegend(!showLegend)}
          className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-gray-200 rounded-lg text-sm font-medium transition-colors"
        >
          {showLegend ? 'Hide Legend' : 'Show Legend'}
        </button>
      </div>

      {/* Design Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gradient-to-br from-red-900/30 to-red-800/20 border border-red-700/50 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <MapPin className="w-5 h-5 text-red-400" />
            <p className="text-red-200 text-sm font-semibold">Corner Points</p>
          </div>
          <p className="text-3xl font-bold text-red-100">{stats.cornerPoints}</p>
        </div>

        {stats.centerPoints > 0 && (
          <div className="bg-gradient-to-br from-green-900/30 to-green-800/20 border border-green-700/50 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <MapPin className="w-5 h-5 text-green-400" />
              <p className="text-green-200 text-sm font-semibold">Center Points</p>
            </div>
            <p className="text-3xl font-bold text-green-100">{stats.centerPoints}</p>
          </div>
        )}

        {stats.axialPoints > 0 && (
          <div className="bg-gradient-to-br from-blue-900/30 to-blue-800/20 border border-blue-700/50 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <MapPin className="w-5 h-5 text-blue-400" />
              <p className="text-blue-200 text-sm font-semibold">Axial Points</p>
            </div>
            <p className="text-3xl font-bold text-blue-100">{stats.axialPoints}</p>
          </div>
        )}

        {stats.edgePoints > 0 && (
          <div className="bg-gradient-to-br from-orange-900/30 to-orange-800/20 border border-orange-700/50 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <MapPin className="w-5 h-5 text-orange-400" />
              <p className="text-orange-200 text-sm font-semibold">Edge Points</p>
            </div>
            <p className="text-3xl font-bold text-orange-100">{stats.edgePoints}</p>
          </div>
        )}

        <div className="bg-gradient-to-br from-purple-900/30 to-purple-800/20 border border-purple-700/50 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <Layers className="w-5 h-5 text-purple-400" />
            <p className="text-purple-200 text-sm font-semibold">Total Runs</p>
          </div>
          <p className="text-3xl font-bold text-purple-100">{stats.totalRuns}</p>
        </div>
      </div>

      {/* Coverage Information */}
      <div
        className="border rounded-lg p-4"
        style={{
          backgroundColor: `${coverageInfo.color}15`,
          borderColor: `${coverageInfo.color}80`
        }}
      >
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 flex-shrink-0 mt-0.5" style={{ color: coverageInfo.color }} />
          <div>
            <p className="font-semibold mb-1" style={{ color: coverageInfo.color }}>
              {stats.designFamily} - {coverageInfo.coverage} Coverage
            </p>
            <p className="text-gray-300 text-sm">{stats.description}</p>
            <p className="text-gray-400 text-sm mt-1">{coverageInfo.description}</p>
          </div>
        </div>
      </div>

      {/* Visualization */}
      <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700/50">
        {nFactors === 2 && render2DPlot()}
        {nFactors === 3 && render3DPlot()}
        {nFactors > 3 && renderMultiPlots()}
      </div>

      {/* Point Type Legend (when hidden from plot) */}
      {!showLegend && (
        <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
          <h4 className="text-gray-100 font-semibold mb-3">Point Types</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {legend.map((item) => (
              <div key={item.type} className="flex items-center gap-3">
                <div
                  className="w-4 h-4 rounded-full"
                  style={{ backgroundColor: item.color, border: '2px solid #1e293b' }}
                />
                <div>
                  <p className="text-gray-200 text-sm font-medium">{item.type}</p>
                  <p className="text-gray-400 text-xs">{item.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Educational Note */}
      <div className="bg-blue-900/20 border border-blue-700/50 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-blue-200">
            <p className="font-semibold mb-1">Understanding the Preview</p>
            <ul className="space-y-1 list-disc list-inside text-blue-300">
              <li><strong>Corner points</strong> test extreme combinations of factors</li>
              <li><strong>Center points</strong> estimate curvature and experimental error</li>
              <li><strong>Axial points</strong> (star points) extend beyond the cube for quadratic modeling</li>
              <li><strong>Edge points</strong> test intermediate levels while avoiding extremes</li>
              <li>Coordinates shown are in <strong>coded units</strong> (-1, 0, +1) for standard designs</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

export default DesignPreview
