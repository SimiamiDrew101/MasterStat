import Plot from 'react-plotly.js'
import { getPlotlyConfig } from '../utils/plotlyConfig'

/**
 * FactorialInteractionPlots component using Plotly
 * Displays main effects and 2-way interaction plots for factorial designs in a grid
 */
const FactorialInteractionPlots = ({ mainEffectsData, interactionData, factors }) => {
  if (!interactionData || Object.keys(interactionData).length === 0) return null

  const interactionNames = Object.keys(interactionData)
  const hasMainEffects = mainEffectsData && Object.keys(mainEffectsData).length > 0

  // Render a main effects plot
  const renderMainEffectPlot = (factor, data, idx) => {
    const trace = {
      type: 'scatter',
      mode: 'lines+markers',
      x: data.levels,
      y: data.means,
      name: factor,
      line: {
        width: 3,
        color: '#3b82f6'
      },
      marker: {
        size: 12,
        color: '#3b82f6',
        line: {
          color: '#1e293b',
          width: 2
        }
      },
      hovertemplate: `<b>${factor}</b><br>` +
        `Level: %{x}<br>` +
        `Mean Response: %{y:.4f}<br>` +
        '<extra></extra>'
    }

    const layout = {
      title: {
        text: `${factor} Main Effect`,
        font: {
          size: 16,
          color: '#f1f5f9'
        }
      },
      xaxis: {
        title: factor,
        gridcolor: '#475569',
        zerolinecolor: '#64748b',
        color: '#e2e8f0',
        categoryorder: 'array',
        categoryarray: data.levels
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
      showlegend: false,
      margin: {
        l: 60,
        r: 40,
        t: 60,
        b: 60
      }
    }

    const config = getPlotlyConfig(`main-effect-${factor}`, {
      modeBarButtonsToRemove: ['lasso2d', 'select2d']
    })

    return (
      <div key={idx} className="bg-slate-800/50 rounded-lg p-4">
        <Plot
          data={[trace]}
          layout={layout}
          config={config}
          style={{ width: '100%', height: '350px' }}
          useResizeHandler={true}
        />
      </div>
    )
  }

  // Render an interaction plot
  const renderInteractionPlot = (interactionName, idx) => {
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

    const interactionLayout = {
      title: {
        text: `${interactionName}`,
        font: {
          size: 16,
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
        x: 0.5,
        y: -0.2,
        xanchor: 'center',
        yanchor: 'top',
        orientation: 'h',
        bgcolor: 'rgba(30, 41, 59, 0.8)',
        bordercolor: '#64748b',
        borderwidth: 1
      },
      margin: {
        l: 60,
        r: 40,
        t: 60,
        b: 100
      }
    }

    const interactionConfig = getPlotlyConfig(`interaction-${interactionName.replace(/×/g, 'x')}`, {
      modeBarButtonsToRemove: ['lasso2d', 'select2d']
    })

    return (
      <div key={idx} className="bg-slate-800/50 rounded-lg p-4">
        <Plot
          data={traces}
          layout={interactionLayout}
          config={interactionConfig}
          style={{ width: '100%', height: '350px' }}
          useResizeHandler={true}
        />
      </div>
    )
  }

  return (
    <div className="bg-slate-700/50 rounded-lg p-6">
      <h3 className="text-xl font-bold text-gray-100 mb-4">Main Effects and Interaction Plots</h3>
      <p className="text-gray-300 text-sm mb-6">
        Main effects show the average response at each factor level. Interaction plots show how the effect of one factor depends on the level of another.
      </p>

      {/* Grid Layout for all plots */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {/* Main Effects Plots */}
        {hasMainEffects && Object.entries(mainEffectsData).map(([factor, data], idx) =>
          renderMainEffectPlot(factor, data, `main-${idx}`)
        )}

        {/* Interaction Plots */}
        {interactionNames.map((interactionName, idx) =>
          renderInteractionPlot(interactionName, `interaction-${idx}`)
        )}
      </div>
    </div>
  )
}

export default FactorialInteractionPlots
