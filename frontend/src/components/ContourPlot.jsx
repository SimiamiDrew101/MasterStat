import Plot from 'react-plotly.js'
import { Target } from 'lucide-react'
import { getPlotlyConfig } from '../utils/plotlyConfig'

const ContourPlot = ({
  surfaceData,
  factor1,
  factor2,
  responseName,
  experimentalData = null,
  optimizationResult = null,
  canonicalResult = null,
  steepestAscentResult = null,
  ridgeAnalysisResult = null
}) => {
  if (!surfaceData || surfaceData.length === 0) return null

  // Extract unique x and y values
  const xValues = [...new Set(surfaceData.map(d => d.x))].sort((a, b) => a - b)
  const yValues = [...new Set(surfaceData.map(d => d.y))].sort((a, b) => a - b)

  // Create 2D grid of z values
  const zGrid = []
  for (let i = 0; i < yValues.length; i++) {
    zGrid[i] = []
    for (let j = 0; j < xValues.length; j++) {
      const point = surfaceData.find(d => d.x === xValues[j] && d.y === yValues[i])
      zGrid[i][j] = point ? point.z : 0
    }
  }

  // Prepare plot data
  const traces = []

  // 1. Contour plot (base layer)
  traces.push({
    type: 'contour',
    x: xValues,
    y: yValues,
    z: zGrid,
    colorscale: [
      [0, '#0050ff'],      // Blue (low values)
      [0.25, '#00d4ff'],   // Cyan
      [0.5, '#64ff96'],    // Green
      [0.75, '#ffff00'],   // Yellow
      [1, '#ff0000']       // Red (high values)
    ],
    contours: {
      coloring: 'heatmap',
      showlabels: true,
      labelfont: {
        size: 10,
        color: 'white'
      }
    },
    colorbar: {
      title: {
        text: responseName,
        side: 'right'
      },
      thickness: 20,
      len: 0.7
    },
    hovertemplate: `${factor1}: %{x:.2f}<br>${factor2}: %{y:.2f}<br>${responseName}: %{z:.2f}<extra></extra>`
  })

  // 2. Experimental data points (if provided)
  if (experimentalData && experimentalData.length > 0) {
    const expX = experimentalData.map(d => d[factor1])
    const expY = experimentalData.map(d => d[factor2])
    const expZ = experimentalData.map(d => d[responseName])

    traces.push({
      type: 'scatter',
      mode: 'markers',
      x: expX,
      y: expY,
      text: expZ.map(z => `${responseName}: ${z.toFixed(2)}`),
      marker: {
        size: 10,
        color: 'white',
        symbol: 'circle',
        line: {
          color: '#1e293b',
          width: 2
        }
      },
      name: 'Experimental Points',
      hovertemplate: `${factor1}: %{x:.2f}<br>${factor2}: %{y:.2f}<br>%{text}<extra></extra>`
    })
  }

  // 3. Optimization path (if provided)
  if (optimizationResult && optimizationResult.optimal_point) {
    const optimalX = optimizationResult.optimal_point[factor1]
    const optimalY = optimizationResult.optimal_point[factor2]

    // Path from origin/center to optimal point
    traces.push({
      type: 'scatter',
      mode: 'lines+markers',
      x: [0, optimalX],
      y: [0, optimalY],
      line: {
        color: '#22c55e',
        width: 3,
        dash: 'dash'
      },
      marker: {
        size: [8, 14],
        color: ['#3b82f6', '#22c55e'],
        symbol: ['circle', 'star']
      },
      name: 'Path to Optimum',
      hovertemplate: `${factor1}: %{x:.2f}<br>${factor2}: %{y:.2f}<extra></extra>`
    })
  }

  // 4. Stationary point (if canonical analysis available)
  if (canonicalResult && canonicalResult.stationary_point) {
    const statX = canonicalResult.stationary_point[factor1]
    const statY = canonicalResult.stationary_point[factor2]

    traces.push({
      type: 'scatter',
      mode: 'markers',
      x: [statX],
      y: [statY],
      marker: {
        size: 16,
        color: '#a855f7',
        symbol: 'x',
        line: {
          color: 'white',
          width: 2
        }
      },
      name: 'Stationary Point',
      hovertemplate: `${factor1}: ${statX}<br>${factor2}: ${statY}<br>Type: ${canonicalResult.surface_type}<extra></extra>`
    })
  }

  // 5. Steepest ascent/descent path (if available)
  if (steepestAscentResult && steepestAscentResult.path && steepestAscentResult.path.length > 0) {
    const pathX = steepestAscentResult.path.map(point => point[factor1])
    const pathY = steepestAscentResult.path.map(point => point[factor2])

    // Only plot if we have valid coordinates for both factors
    if (pathX.length > 0 && pathY.length > 0 && pathX.every(x => x !== undefined) && pathY.every(y => y !== undefined)) {
      traces.push({
        type: 'scatter',
        mode: 'lines+markers',
        x: pathX,
        y: pathY,
        line: {
          color: '#f59e0b',
          width: 3
        },
        marker: {
          size: 8,
          color: '#f59e0b',
          symbol: 'circle',
          line: {
            color: 'white',
            width: 1
          }
        },
        name: 'Steepest Ascent Path',
        hovertemplate: `Step: %{text}<br>${factor1}: %{x:.2f}<br>${factor2}: %{y:.2f}<extra></extra>`,
        text: steepestAscentResult.path.map(point => point.step || 0)
      })

      // Add arrow to show direction
      if (pathX.length >= 2) {
        traces.push({
          type: 'scatter',
          mode: 'markers',
          x: [pathX[pathX.length - 1]],
          y: [pathY[pathY.length - 1]],
          marker: {
            size: 14,
            color: '#f59e0b',
            symbol: 'triangle-up',
            line: {
              color: 'white',
              width: 2
            }
          },
          name: 'Path Direction',
          showlegend: false,
          hoverinfo: 'skip'
        })
      }
    }
  }

  // 6. Ridge analysis contour (if available)
  if (ridgeAnalysisResult && ridgeAnalysisResult.ridge_points && ridgeAnalysisResult.ridge_points.length > 0) {
    const ridgeX = ridgeAnalysisResult.ridge_points.map(point => point[factor1])
    const ridgeY = ridgeAnalysisResult.ridge_points.map(point => point[factor2])

    // Only plot if we have valid coordinates
    if (ridgeX.length > 0 && ridgeY.length > 0 && ridgeX.every(x => x !== undefined) && ridgeY.every(y => y !== undefined)) {
      // Close the contour by adding first point at the end
      ridgeX.push(ridgeX[0])
      ridgeY.push(ridgeY[0])

      traces.push({
        type: 'scatter',
        mode: 'lines',
        x: ridgeX,
        y: ridgeY,
        line: {
          color: '#a855f7',
          width: 3,
          dash: 'dot'
        },
        name: `Ridge (Y=${ridgeAnalysisResult.target_response})`,
        hovertemplate: `${factor1}: %{x:.2f}<br>${factor2}: %{y:.2f}<br>Response: ${ridgeAnalysisResult.target_response}<extra></extra>`
      })
    }
  }

  const layout = {
    title: {
      text: `${responseName} Contour Plot`,
      font: {
        size: 20,
        color: '#f1f5f9'
      }
    },
    autosize: true,
    xaxis: {
      title: factor1,
      gridcolor: '#475569',
      zerolinecolor: '#64748b'
    },
    yaxis: {
      title: factor2,
      gridcolor: '#475569',
      zerolinecolor: '#64748b',
      scaleanchor: 'x'
    },
    paper_bgcolor: '#334155',
    plot_bgcolor: '#1e293b',
    font: {
      color: '#e2e8f0'
    },
    showlegend: true,
    legend: {
      x: 1.02,
      y: 1,
      bgcolor: 'rgba(30, 41, 59, 0.8)',
      bordercolor: '#475569',
      borderwidth: 1
    },
    margin: {
      l: 80,
      r: 150,
      b: 80,
      t: 80
    }
  }

  const config = getPlotlyConfig('contour-plot')

  return (
    <div className="bg-slate-700/50 rounded-lg p-6">
      <div className="mb-4">
        <h4 className="text-gray-100 font-semibold text-lg">Interactive Contour Plot</h4>
        <div className="flex items-center gap-4 mt-2 text-sm">
          {experimentalData && (
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-white border-2 border-slate-800"></div>
              <span className="text-gray-300">Experimental Points</span>
            </div>
          )}
          {optimizationResult && (
            <div className="flex items-center gap-2">
              <Target className="w-4 h-4 text-green-400" />
              <span className="text-gray-300">Path to Optimum</span>
            </div>
          )}
          {canonicalResult && (
            <div className="flex items-center gap-2">
              <span className="text-purple-400 text-xl">×</span>
              <span className="text-gray-300">Stationary Point</span>
            </div>
          )}
        </div>
      </div>

      <div className="flex justify-center bg-slate-800/50 rounded-lg p-4">
        <Plot
          data={traces}
          layout={layout}
          config={config}
          style={{ width: '100%', height: '700px' }}
          useResizeHandler={true}
        />
      </div>

      <div className="mt-4 bg-slate-800/50 rounded-lg p-4">
        <p className="text-gray-300 text-sm">
          <strong className="text-gray-100">Interpretation:</strong> Contour lines connect points with equal response values. The color gradient shows the response surface with red indicating higher values and blue indicating lower values.
          {experimentalData && ' White dots show your actual experimental data points.'}
          {optimizationResult && ' The green dashed line shows the path from the design center to the optimal settings.'}
          {canonicalResult && ` The purple X marks the stationary point (${canonicalResult.surface_type}).`}
        </p>
      </div>
    </div>
  )
}

export default ContourPlot
