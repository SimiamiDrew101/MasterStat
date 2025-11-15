import Plot from 'react-plotly.js'

/**
 * CubePlot component for visualizing 2^3 and 2^4 factorial designs using Plotly
 * Shows response values at each corner of the design space in an interactive 3D plot
 */
const CubePlot = ({ data, factors, responseName = 'Response' }) => {
  if (!data || data.length === 0 || !factors) return null

  const numFactors = factors.length

  // For 2^3 designs, create a 3D scatter plot with cube edges
  if (numFactors === 3) {
    // Create cube edges
    const edges = [
      // Bottom square
      [[0, 1], [0, 0], [0, 0]],
      [[1, 1], [0, 1], [0, 0]],
      [[1, 0], [1, 1], [0, 0]],
      [[0, 0], [1, 0], [0, 0]],
      // Top square
      [[0, 1], [0, 0], [1, 1]],
      [[1, 1], [0, 1], [1, 1]],
      [[1, 0], [1, 1], [1, 1]],
      [[0, 0], [1, 0], [1, 1]],
      // Vertical edges
      [[0, 0], [0, 0], [0, 1]],
      [[1, 1], [0, 0], [0, 1]],
      [[1, 1], [1, 1], [0, 1]],
      [[0, 0], [1, 1], [0, 1]]
    ]

    const edgeTraces = edges.map((edge, idx) => ({
      type: 'scatter3d',
      mode: 'lines',
      x: edge[0],
      y: edge[1],
      z: edge[2],
      line: {
        color: '#64748b',
        width: 3
      },
      hoverinfo: 'skip',
      showlegend: false,
      name: ''
    }))

    // Create scatter plot for response values at corners
    const cornerTrace = {
      type: 'scatter3d',
      mode: 'markers+text',
      x: data.map(d => d.x),
      y: data.map(d => d.y),
      z: data.map(d => d.z),
      text: data.map(d => `${d.response.toFixed(2)}`),
      textposition: 'top center',
      textfont: {
        color: '#e2e8f0',
        size: 14,
        family: 'monospace',
        weight: 'bold'
      },
      marker: {
        size: 16,
        color: data.map(d => d.response),
        colorscale: [
          [0, '#0050ff'],      // Blue (low values)
          [0.25, '#00d4ff'],   // Cyan
          [0.5, '#64ff96'],    // Green
          [0.75, '#ffff00'],   // Yellow
          [1, '#ff0000']       // Red (high values)
        ],
        showscale: true,
        colorbar: {
          title: {
            text: responseName,
            side: 'right'
          },
          thickness: 20,
          len: 0.7
        },
        line: {
          color: '#f1f5f9',
          width: 2
        }
      },
      hovertemplate: '<b>Response: %{text}</b><br>' +
        `${factors[0]}: %{x}<br>` +
        `${factors[1]}: %{y}<br>` +
        `${factors[2]}: %{z}<br>` +
        '<extra></extra>',
      name: 'Design Points'
    }

    const plotData = [...edgeTraces, cornerTrace]

    const layout = {
      title: {
        text: `2³ Factorial Design Cube Plot`,
        font: {
          size: 20,
          color: '#f1f5f9'
        }
      },
      autosize: true,
      scene: {
        xaxis: {
          title: factors[0],
          backgroundcolor: '#1e293b',
          gridcolor: '#475569',
          showbackground: true,
          zerolinecolor: '#64748b',
          tickvals: [0, 1],
          ticktext: ['Low', 'High']
        },
        yaxis: {
          title: factors[1],
          backgroundcolor: '#1e293b',
          gridcolor: '#475569',
          showbackground: true,
          zerolinecolor: '#64748b',
          tickvals: [0, 1],
          ticktext: ['Low', 'High']
        },
        zaxis: {
          title: factors[2],
          backgroundcolor: '#1e293b',
          gridcolor: '#475569',
          showbackground: true,
          zerolinecolor: '#64748b',
          tickvals: [0, 1],
          ticktext: ['Low', 'High']
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
        t: 50
      }
    }

    const config = {
      responsive: true,
      displayModeBar: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d'],
      toImageButtonOptions: {
        format: 'png',
        filename: `cube-plot-${new Date().toISOString().split('T')[0]}`,
        height: 1000,
        width: 1200,
        scale: 2
      }
    }

    return (
      <div className="bg-slate-700/50 rounded-lg p-6">
        <div className="mb-4">
          <h4 className="text-gray-100 font-semibold text-lg">2³ Design Cube Plot</h4>
          <p className="text-gray-400 text-sm mt-1">
            Each corner represents a treatment combination. Numbers show response values.
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
          <p className="text-gray-300 text-sm">
            <strong className="text-gray-100">Interpretation:</strong> This cube plot visualizes all 8 treatment combinations of a 2³ factorial design.
            Each vertex of the cube represents a unique combination of factor levels (Low/High). The color and numerical values indicate the response at each combination.
            Red corners indicate higher responses, blue indicates lower responses. Look for patterns to identify main effects and interactions.
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
  } else if (numFactors === 4) {
    // For 4 factors, show two 3D cubes for different levels of the 4th factor
    return (
      <div className="bg-slate-700/50 rounded-lg p-6">
        <div className="mb-4">
          <h4 className="text-gray-100 font-semibold text-lg">2⁴ Design Visualization</h4>
          <p className="text-gray-400 text-sm mt-1">
            Showing two 3D cubes representing {factors[3]} at Low and High levels
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {[0, 1].map(level4 => {
            const filteredData = data.filter(d => {
              // Check if data has 'w' property for 4th dimension
              if ('w' in d) return d.w === level4
              // Otherwise try to infer from the data structure
              return false
            })

            if (filteredData.length === 0) return null

            // Create cube edges
            const edges = [
              [[0, 1], [0, 0], [0, 0]],
              [[1, 1], [0, 1], [0, 0]],
              [[1, 0], [1, 1], [0, 0]],
              [[0, 0], [1, 0], [0, 0]],
              [[0, 1], [0, 0], [1, 1]],
              [[1, 1], [0, 1], [1, 1]],
              [[1, 0], [1, 1], [1, 1]],
              [[0, 0], [1, 0], [1, 1]],
              [[0, 0], [0, 0], [0, 1]],
              [[1, 1], [0, 0], [0, 1]],
              [[1, 1], [1, 1], [0, 1]],
              [[0, 0], [1, 1], [0, 1]]
            ]

            const edgeTraces = edges.map((edge, idx) => ({
              type: 'scatter3d',
              mode: 'lines',
              x: edge[0],
              y: edge[1],
              z: edge[2],
              line: {
                color: '#64748b',
                width: 2
              },
              hoverinfo: 'skip',
              showlegend: false
            }))

            const cornerTrace = {
              type: 'scatter3d',
              mode: 'markers+text',
              x: filteredData.map(d => d.x),
              y: filteredData.map(d => d.y),
              z: filteredData.map(d => d.z),
              text: filteredData.map(d => `${d.response.toFixed(1)}`),
              textposition: 'top center',
              textfont: {
                color: '#e2e8f0',
                size: 12,
                family: 'monospace'
              },
              marker: {
                size: 12,
                color: filteredData.map(d => d.response),
                colorscale: [
                  [0, '#0050ff'],
                  [0.25, '#00d4ff'],
                  [0.5, '#64ff96'],
                  [0.75, '#ffff00'],
                  [1, '#ff0000']
                ],
                showscale: level4 === 1,
                colorbar: level4 === 1 ? {
                  title: {
                    text: responseName,
                    side: 'right'
                  },
                  thickness: 15,
                  len: 0.7
                } : undefined,
                line: {
                  color: '#f1f5f9',
                  width: 2
                }
              },
              hovertemplate: '<b>%{text}</b><br>' +
                `${factors[0]}: %{x}<br>` +
                `${factors[1]}: %{y}<br>` +
                `${factors[2]}: %{z}<br>` +
                `${factors[3]}: ${level4 === 0 ? 'Low' : 'High'}<br>` +
                '<extra></extra>'
            }

            const plotData = [...edgeTraces, cornerTrace]

            const layout = {
              title: {
                text: `${factors[3]} = ${level4 === 0 ? 'Low' : 'High'}`,
                font: {
                  size: 16,
                  color: '#f1f5f9'
                }
              },
              autosize: true,
              scene: {
                xaxis: {
                  title: factors[0],
                  backgroundcolor: '#1e293b',
                  gridcolor: '#475569',
                  showbackground: true,
                  tickvals: [0, 1],
                  ticktext: ['L', 'H']
                },
                yaxis: {
                  title: factors[1],
                  backgroundcolor: '#1e293b',
                  gridcolor: '#475569',
                  showbackground: true,
                  tickvals: [0, 1],
                  ticktext: ['L', 'H']
                },
                zaxis: {
                  title: factors[2],
                  backgroundcolor: '#1e293b',
                  gridcolor: '#475569',
                  showbackground: true,
                  tickvals: [0, 1],
                  ticktext: ['L', 'H']
                },
                camera: {
                  eye: { x: 1.5, y: 1.5, z: 1.2 }
                }
              },
              paper_bgcolor: '#334155',
              plot_bgcolor: '#1e293b',
              font: { color: '#e2e8f0', size: 10 },
              margin: { l: 0, r: 0, b: 0, t: 40 }
            }

            const config = {
              responsive: true,
              displayModeBar: true,
              displaylogo: false,
              modeBarButtonsToRemove: ['lasso2d', 'select2d'],
              toImageButtonOptions: {
                format: 'png',
                filename: `cube-plot-${factors[3]}-${level4 === 0 ? 'low' : 'high'}-${new Date().toISOString().split('T')[0]}`,
                height: 800,
                width: 800,
                scale: 2
              }
            }

            return (
              <div key={level4} className="bg-slate-800/50 rounded-lg p-3">
                <Plot
                  data={plotData}
                  layout={layout}
                  config={config}
                  style={{ width: '100%', height: '400px' }}
                  useResizeHandler={true}
                />
              </div>
            )
          })}
        </div>

        <div className="mt-4 bg-slate-800/50 rounded-lg p-4">
          <p className="text-gray-300 text-sm">
            <strong className="text-gray-100">Interpretation:</strong> For 2⁴ designs with 16 treatment combinations, we show two 3D cubes - one for each level of the 4th factor ({factors[3]}).
            Compare patterns across both cubes to understand how the 4th factor interacts with others.
          </p>
        </div>
      </div>
    )
  }

  return null
}

export default CubePlot
