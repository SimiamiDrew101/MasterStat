import Plot from 'react-plotly.js'

/**
 * FactorialInteractionPlots component using Plotly
 * Displays all 2-way interaction plots for factorial designs
 */
const FactorialInteractionPlots = ({ interactionData, factors }) => {
  if (!interactionData || Object.keys(interactionData).length === 0) return null

  const interactionNames = Object.keys(interactionData)

  return (
    <div className="space-y-6">
      {interactionNames.map((interactionName, idx) => {
        const plotInfo = interactionData[interactionName]

        // Create traces for each line (one per level of line_factor)
        const traces = plotInfo.lines.map((line, lineIdx) => ({
          type: 'scatter',
          mode: 'lines+markers',
          x: plotInfo.x_levels,
          y: line.values,
          name: `${plotInfo.line_factor} = ${line.label}`,
          line: {
            width: 3,
            shape: 'linear'
          },
          marker: {
            size: 10,
            line: {
              color: '#1e293b',
              width: 2
            }
          },
          hovertemplate: `<b>${plotInfo.line_factor} = ${line.label}</b><br>` +
            `${plotInfo.x_factor}: %{x}<br>` +
            `Mean Response: %{y:.4f}<br>` +
            '<extra></extra>'
        }))

        const layout = {
          title: {
            text: `${interactionName} Interaction`,
            font: {
              size: 18,
              color: '#f1f5f9'
            }
          },
          xaxis: {
            title: plotInfo.x_factor,
            gridcolor: '#475569',
            zerolinecolor: '#64748b',
            color: '#e2e8f0',
            categoryorder: 'array',
            categoryarray: plotInfo.x_levels
          },
          yaxis: {
            title: 'Mean Response',
            gridcolor: '#475569',
            zerolinecolor: '#64748b',
            color: '#e2e8f0'
          },
          paper_bgcolor: '#334155',
          plot_bgcolor: '#1e293b',
          font: {
            color: '#e2e8f0'
          },
          hovermode: 'closest',
          showlegend: true,
          legend: {
            x: 1.02,
            y: 1,
            xanchor: 'left',
            bgcolor: 'rgba(30, 41, 59, 0.8)',
            bordercolor: '#64748b',
            borderwidth: 1
          },
          margin: {
            l: 60,
            r: 120,
            t: 60,
            b: 60
          }
        }

        const config = {
          responsive: true,
          displayModeBar: true,
          displaylogo: false,
          modeBarButtonsToRemove: ['lasso2d', 'select2d'],
          toImageButtonOptions: {
            format: 'png',
            filename: `interaction-${interactionName.replace(/×/g, 'x')}-${new Date().toISOString().split('T')[0]}`,
            height: 600,
            width: 900,
            scale: 2
          }
        }

        // Check if lines are parallel (no interaction) or crossed (interaction present)
        const isParallel = traces.length === 2 && (() => {
          const line1 = traces[0].y
          const line2 = traces[1].y
          if (!line1 || !line2 || line1.length < 2) return true

          const slope1 = line1[1] - line1[0]
          const slope2 = line2[1] - line2[0]

          // Check if slopes are similar (within 10% tolerance)
          const tolerance = Math.abs(Math.max(slope1, slope2)) * 0.1
          return Math.abs(slope1 - slope2) < tolerance
        })()

        return (
          <div key={idx} className="bg-slate-700/50 rounded-lg p-6">
            <div className="mb-4">
              <h4 className="text-gray-100 font-semibold text-lg">{interactionName} Interaction Plot</h4>
              <p className="text-gray-400 text-sm mt-1">
                {is Parallel
                  ? "Lines are approximately parallel - interaction likely not significant"
                  : "Lines are non-parallel or crossing - interaction may be significant"}
              </p>
            </div>

            <div className="bg-slate-800/50 rounded-lg p-4">
              <Plot
                data={traces}
                layout={layout}
                config={config}
                style={{ width: '100%', height: '400px' }}
                useResizeHandler={true}
              />
            </div>

            <div className="mt-4 bg-slate-800/50 rounded-lg p-4">
              <p className="text-gray-300 text-sm">
                <strong className="text-gray-100">Interpretation:</strong> This plot shows how the effect of {plotInfo.x_factor} depends on the level of {plotInfo.line_factor}.
                {isParallel ? (
                  <span className="text-green-400"> Parallel lines suggest the two factors act independently (no interaction).</span>
                ) : (
                  <span className="text-yellow-400"> Non-parallel or crossing lines indicate an interaction - the effect of one factor changes depending on the level of the other factor.</span>
                )}
              </p>
            </div>
          </div>
        )
      })}
    </div>
  )
}

export default FactorialInteractionPlots
