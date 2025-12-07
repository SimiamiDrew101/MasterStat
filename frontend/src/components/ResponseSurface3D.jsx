import Plot from 'react-plotly.js'
import { getPlotlyConfig } from '../utils/plotlyConfig'

const ResponseSurface3D = ({ surfaceData, factor1, factor2, responseName }) => {
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

  // Prepare data for Plotly
  const plotData = [
    {
      type: 'surface',
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
        z: {
          show: true,
          usecolormap: true,
          highlightcolor: '#42f462',
          project: { z: true }
        }
      },
      colorbar: {
        title: {
          text: responseName,
          side: 'right'
        },
        thickness: 20,
        len: 0.7
      }
    }
  ]

  const layout = {
    title: {
      text: `${responseName} Response Surface`,
      font: {
        size: 20,
        color: '#f1f5f9'
      }
    },
    autosize: true,
    scene: {
      xaxis: {
        title: factor1,
        backgroundcolor: '#1e293b',
        gridcolor: '#475569',
        showbackground: true,
        zerolinecolor: '#64748b'
      },
      yaxis: {
        title: factor2,
        backgroundcolor: '#1e293b',
        gridcolor: '#475569',
        showbackground: true,
        zerolinecolor: '#64748b'
      },
      zaxis: {
        title: responseName,
        backgroundcolor: '#1e293b',
        gridcolor: '#475569',
        showbackground: true,
        zerolinecolor: '#64748b'
      },
      camera: {
        eye: {
          x: 1.5,
          y: 1.5,
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
      t: 50
    }
  }

  const config = getPlotlyConfig('response-surface', {
    modeBarButtonsToAdd: ['hoverclosest', 'hovercompare'],
    modeBarButtonsToRemove: ['lasso2d', 'select2d']
  })

  return (
    <div className="bg-slate-700/50 rounded-lg p-6">
      <div className="mb-4">
        <h4 className="text-gray-100 font-semibold text-lg">Interactive 3D Response Surface</h4>
        <p className="text-gray-400 text-sm mt-1">
          Click and drag to rotate • Scroll to zoom • Shift+drag to pan
        </p>
      </div>

      <div className="flex justify-center bg-slate-800/50 rounded-lg p-4">
        <Plot
          data={plotData}
          layout={layout}
          config={config}
          style={{ width: '100%', height: '700px' }}
          useResizeHandler={true}
        />
      </div>

      <div className="mt-4 bg-slate-800/50 rounded-lg p-4">
        <p className="text-gray-300 text-sm">
          <strong className="text-gray-100">Interpretation:</strong> This interactive 3D surface plot visualizes the response as a function of two factors. The color gradient represents the response magnitude - red indicates higher values, blue indicates lower values. The shape of the surface reveals the nature of the optimization problem (convex, concave, or saddle point). Use your mouse to rotate, zoom, and pan the plot for better visualization.
        </p>
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

export default ResponseSurface3D
