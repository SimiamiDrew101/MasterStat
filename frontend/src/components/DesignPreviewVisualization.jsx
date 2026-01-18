import Plot from 'react-plotly.js'
import { getPlotlyConfig } from '../utils/plotlyConfig'

/**
 * DesignPreviewVisualization component for visualizing experimental designs
 * Shows design points in 2D or 3D space with color coding by point type
 * Supports all design types: CCD, Box-Behnken, DSD, Plackett-Burman, etc.
 */
const DesignPreviewVisualization = ({ design, factors, designType = 'Unknown' }) => {
  if (!design || design.length === 0 || !factors || factors.length === 0) {
    return (
      <div className="bg-slate-700/50 rounded-lg p-6">
        <p className="text-gray-400 text-center">No design data to visualize</p>
      </div>
    )
  }

  const numFactors = factors.length

  // Classify points by type (factorial, axial, center, other)
  const classifyPoint = (point) => {
    const values = factors.map(f => Math.abs(point[f]))
    const maxVal = Math.max(...values)
    const isAllZero = values.every(v => v < 0.1)
    const isAllOne = values.every(v => Math.abs(v - 1) < 0.1)
    const hasAlpha = values.some(v => v > 1.1) // Alpha > 1 for rotatable CCD

    if (isAllZero) return 'center'
    if (hasAlpha) return 'axial'
    if (isAllOne) return 'factorial'
    return 'other'
  }

  const pointTypes = design.map(classifyPoint)

  // Color map for point types
  const colorMap = {
    'factorial': '#3b82f6',    // Blue - factorial points
    'axial': '#10b981',        // Green - axial/star points
    'center': '#f59e0b',       // Amber - center points
    'other': '#8b5cf6'         // Purple - other points (DSD, PB, etc.)
  }

  const typeNames = {
    'factorial': 'Factorial',
    'axial': 'Axial (Star)',
    'center': 'Center',
    'other': 'Design Points'
  }

  // For 2 factors: 2D scatter plot
  if (numFactors === 2) {
    const traces = Object.keys(colorMap).map(type => {
      const indices = pointTypes.map((t, i) => t === type ? i : -1).filter(i => i >= 0)
      if (indices.length === 0) return null

      return {
        type: 'scatter',
        mode: 'markers',
        x: indices.map(i => design[i][factors[0]]),
        y: indices.map(i => design[i][factors[1]]),
        marker: {
          size: 14,
          color: colorMap[type],
          line: {
            color: '#f1f5f9',
            width: 2
          }
        },
        name: typeNames[type],
        hovertemplate: `<b>${typeNames[type]}</b><br>` +
          `${factors[0]}: %{x:.2f}<br>` +
          `${factors[1]}: %{y:.2f}<br>` +
          '<extra></extra>'
      }
    }).filter(t => t !== null)

    const layout = {
      title: {
        text: `${designType} - Design Space (${design.length} runs)`,
        font: {
          size: 18,
          color: '#f1f5f9'
        }
      },
      xaxis: {
        title: factors[0],
        gridcolor: '#475569',
        zerolinecolor: '#64748b',
        color: '#e2e8f0'
      },
      yaxis: {
        title: factors[1],
        gridcolor: '#475569',
        zerolinecolor: '#64748b',
        color: '#e2e8f0'
      },
      paper_bgcolor: '#334155',
      plot_bgcolor: '#1e293b',
      font: {
        color: '#e2e8f0'
      },
      margin: {
        l: 60,
        r: 60,
        b: 60,
        t: 80
      },
      showlegend: true,
      legend: {
        x: 1.05,
        y: 1,
        bgcolor: 'rgba(51, 65, 85, 0.8)',
        bordercolor: '#475569',
        borderwidth: 1
      }
    }

    const config = getPlotlyConfig('design-preview-2d', {
      modeBarButtonsToRemove: ['lasso2d', 'select2d', 'zoom2d', 'pan2d']
    })

    return (
      <div className="bg-slate-700/50 rounded-lg p-6">
        <div className="mb-4">
          <h4 className="text-gray-100 font-semibold text-lg">Design Preview</h4>
          <p className="text-gray-400 text-sm mt-1">
            2D visualization of {design.length} design points in the factor space
          </p>
        </div>

        <div className="flex justify-center bg-slate-800/50 rounded-lg p-4">
          <Plot
            data={traces}
            layout={layout}
            config={config}
            style={{ width: '100%', height: '500px' }}
            useResizeHandler={true}
          />
        </div>

        <div className="mt-4 bg-slate-800/50 rounded-lg p-4">
          <p className="text-gray-300 text-sm">
            <strong className="text-gray-100">Interpretation:</strong> Each point represents one experimental run.
            Different colors indicate different point types in the design. The spatial distribution shows
            how the design explores the factor space.
          </p>
        </div>
      </div>
    )
  }

  // For 3+ factors: 3D scatter plot (using first 3 factors)
  if (numFactors >= 3) {
    const f1 = factors[0]
    const f2 = factors[1]
    const f3 = factors[2]

    const traces = Object.keys(colorMap).map(type => {
      const indices = pointTypes.map((t, i) => t === type ? i : -1).filter(i => i >= 0)
      if (indices.length === 0) return null

      return {
        type: 'scatter3d',
        mode: 'markers',
        x: indices.map(i => design[i][f1]),
        y: indices.map(i => design[i][f2]),
        z: indices.map(i => design[i][f3]),
        marker: {
          size: 8,
          color: colorMap[type],
          line: {
            color: '#f1f5f9',
            width: 1
          }
        },
        name: typeNames[type],
        hovertemplate: `<b>${typeNames[type]}</b><br>` +
          `${f1}: %{x:.2f}<br>` +
          `${f2}: %{y:.2f}<br>` +
          `${f3}: %{z:.2f}<br>` +
          '<extra></extra>'
      }
    }).filter(t => t !== null)

    // Add wireframe cube for design space boundaries
    const edges = [
      // Bottom square
      [[-1, 1], [-1, -1], [-1, -1]],
      [[1, 1], [-1, 1], [-1, -1]],
      [[1, -1], [1, 1], [-1, -1]],
      [[-1, -1], [1, -1], [-1, -1]],
      // Top square
      [[-1, 1], [-1, -1], [1, 1]],
      [[1, 1], [-1, 1], [1, 1]],
      [[1, -1], [1, 1], [1, 1]],
      [[-1, -1], [1, -1], [1, 1]],
      // Vertical edges
      [[-1, -1], [-1, -1], [-1, 1]],
      [[1, 1], [-1, -1], [-1, 1]],
      [[1, 1], [1, 1], [-1, 1]],
      [[-1, -1], [1, 1], [-1, 1]]
    ]

    const edgeTraces = edges.map((edge, idx) => ({
      type: 'scatter3d',
      mode: 'lines',
      x: edge[0],
      y: edge[1],
      z: edge[2],
      line: {
        color: '#64748b',
        width: 2,
        dash: 'dot'
      },
      hoverinfo: 'skip',
      showlegend: false,
      name: ''
    }))

    const plotData = [...edgeTraces, ...traces]

    const layout = {
      title: {
        text: `${designType} - Design Space (${design.length} runs)`,
        font: {
          size: 18,
          color: '#f1f5f9'
        }
      },
      autosize: true,
      scene: {
        xaxis: {
          title: f1,
          backgroundcolor: '#1e293b',
          gridcolor: '#475569',
          showbackground: true,
          zerolinecolor: '#64748b',
          color: '#e2e8f0'
        },
        yaxis: {
          title: f2,
          backgroundcolor: '#1e293b',
          gridcolor: '#475569',
          showbackground: true,
          zerolinecolor: '#64748b',
          color: '#e2e8f0'
        },
        zaxis: {
          title: f3,
          backgroundcolor: '#1e293b',
          gridcolor: '#475569',
          showbackground: true,
          zerolinecolor: '#64748b',
          color: '#e2e8f0'
        },
        camera: {
          eye: {
            x: 1.7,
            y: 1.7,
            z: 1.3
          }
        }
      },
      paper_bgcolor: '#334155',
      plot_bgcolor: '#1e293b',
      font: {
        color: '#e2e8f0'
      },
      margin: {
        l: 0,
        r: 0,
        b: 0,
        t: 60
      },
      showlegend: true,
      legend: {
        x: 0.85,
        y: 0.95,
        bgcolor: 'rgba(51, 65, 85, 0.8)',
        bordercolor: '#475569',
        borderwidth: 1
      }
    }

    const config = getPlotlyConfig('design-preview-3d', {
      modeBarButtonsToRemove: ['lasso2d', 'select2d']
    })

    return (
      <div className="bg-slate-700/50 rounded-lg p-6">
        <div className="mb-4">
          <h4 className="text-gray-100 font-semibold text-lg">Design Preview - 3D Projection</h4>
          <p className="text-gray-400 text-sm mt-1">
            Showing {design.length} design points (first 3 factors: {f1}, {f2}, {f3})
            {numFactors > 3 && ` - ${numFactors - 3} additional factor${numFactors > 4 ? 's' : ''} not shown`}
          </p>
        </div>

        <div className="flex justify-center bg-slate-800/50 rounded-lg p-4">
          <Plot
            data={plotData}
            layout={layout}
            config={config}
            style={{ width: '100%', height: '600px' }}
            useResizeHandler={true}
          />
        </div>

        <div className="mt-4 bg-slate-800/50 rounded-lg p-4">
          <p className="text-gray-300 text-sm mb-3">
            <strong className="text-gray-100">Interpretation:</strong> This 3D visualization shows how the design
            explores the experimental space. The dotted cube represents the standard design region (±1 coded units).
            Different point types serve different purposes in the design.
          </p>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
            {Object.keys(typeNames).map(type => {
              const count = pointTypes.filter(t => t === type).length
              if (count === 0) return null

              return (
                <div key={type} className="bg-slate-700/50 rounded p-2">
                  <div className="flex items-center gap-2">
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: colorMap[type] }}
                    />
                    <span className="text-gray-300">{typeNames[type]}</span>
                  </div>
                  <div className="text-gray-400 ml-5">{count} point{count !== 1 ? 's' : ''}</div>
                </div>
              )
            })}
          </div>

          <div className="mt-3 grid grid-cols-3 gap-3 text-xs">
            <div className="bg-slate-700/50 rounded p-2">
              <span className="text-gray-400">Rotate:</span>
              <span className="text-gray-200 ml-1 font-medium">Click & drag</span>
            </div>
            <div className="bg-slate-700/50 rounded p-2">
              <span className="text-gray-400">Zoom:</span>
              <span className="text-gray-200 ml-1 font-medium">Scroll wheel</span>
            </div>
            <div className="bg-slate-700/50 rounded p-2">
              <span className="text-gray-400">Pan:</span>
              <span className="text-gray-200 ml-1 font-medium">Shift + drag</span>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-slate-700/50 rounded-lg p-6">
      <p className="text-gray-400 text-center">
        Design visualization requires at least 2 factors
      </p>
    </div>
  )
}

export default DesignPreviewVisualization
