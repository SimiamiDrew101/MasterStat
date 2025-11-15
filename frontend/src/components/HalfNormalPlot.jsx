import Plot from 'react-plotly.js'

/**
 * HalfNormalPlot component for effect screening using Lenth's method
 * Helps identify significant effects in unreplicated factorial designs
 */
const HalfNormalPlot = ({ lenthsData }) => {
  if (!lenthsData || !lenthsData.half_normal_plot_data) return null

  const plotData = lenthsData.half_normal_plot_data

  // Separate insignificant and significant effects
  const insignificantEffects = plotData.filter(d => !d.is_significant_me)
  const significantME = plotData.filter(d => d.is_significant_me && !d.is_significant_sme)
  const significantSME = plotData.filter(d => d.is_significant_sme)

  // Create traces for each category
  const traces = []

  // Insignificant effects (gray)
  if (insignificantEffects.length > 0) {
    traces.push({
      type: 'scatter',
      mode: 'markers+text',
      x: insignificantEffects.map(d => d.half_normal_quantile),
      y: insignificantEffects.map(d => d.abs_effect),
      text: insignificantEffects.map(d => d.name),
      textposition: 'top center',
      textfont: {
        size: 10,
        color: '#94a3b8'
      },
      marker: {
        size: 8,
        color: '#64748b',
        line: {
          color: '#475569',
          width: 1
        }
      },
      name: 'Insignificant',
      hovertemplate: '<b>%{text}</b><br>' +
        'Absolute Effect: %{y:.4f}<br>' +
        'Half-Normal Quantile: %{x:.3f}<br>' +
        '<extra></extra>'
    })
  }

  // Significant at ME level (yellow)
  if (significantME.length > 0) {
    traces.push({
      type: 'scatter',
      mode: 'markers+text',
      x: significantME.map(d => d.half_normal_quantile),
      y: significantME.map(d => d.abs_effect),
      text: significantME.map(d => d.name),
      textposition: 'top center',
      textfont: {
        size: 11,
        color: '#fbbf24',
        weight: 'bold'
      },
      marker: {
        size: 12,
        color: '#fbbf24',
        symbol: 'diamond',
        line: {
          color: '#f59e0b',
          width: 2
        }
      },
      name: 'Significant (ME)',
      hovertemplate: '<b>%{text}</b><br>' +
        'Absolute Effect: %{y:.4f}<br>' +
        'Half-Normal Quantile: %{x:.3f}<br>' +
        'Exceeds ME threshold<br>' +
        '<extra></extra>'
    })
  }

  // Significant at SME level (red)
  if (significantSME.length > 0) {
    traces.push({
      type: 'scatter',
      mode: 'markers+text',
      x: significantSME.map(d => d.half_normal_quantile),
      y: significantSME.map(d => d.abs_effect),
      text: significantSME.map(d => d.name),
      textposition: 'top center',
      textfont: {
        size: 12,
        color: '#ef4444',
        weight: 'bold'
      },
      marker: {
        size: 14,
        color: '#ef4444',
        symbol: 'diamond',
        line: {
          color: '#dc2626',
          width: 2
        }
      },
      name: 'Highly Significant (SME)',
      hovertemplate: '<b>%{text}</b><br>' +
        'Absolute Effect: %{y:.4f}<br>' +
        'Half-Normal Quantile: %{x:.3f}<br>' +
        'Exceeds SME threshold<br>' +
        '<extra></extra>'
    })
  }

  // Add reference line through origin
  const maxQuantile = Math.max(...plotData.map(d => d.half_normal_quantile))
  traces.push({
    type: 'scatter',
    mode: 'lines',
    x: [0, maxQuantile],
    y: [0, maxQuantile * lenthsData.pse],
    line: {
      color: '#64748b',
      width: 2,
      dash: 'dash'
    },
    name: 'Reference Line',
    hoverinfo: 'skip',
    showlegend: false
  })

  // Add ME threshold line
  traces.push({
    type: 'scatter',
    mode: 'lines',
    x: [0, maxQuantile],
    y: [lenthsData.me, lenthsData.me],
    line: {
      color: '#fbbf24',
      width: 2,
      dash: 'dot'
    },
    name: `ME = ${lenthsData.me.toFixed(4)}`,
    hovertemplate: `Margin of Error (ME)<br>Threshold: ${lenthsData.me.toFixed(4)}<extra></extra>`
  })

  // Add SME threshold line
  traces.push({
    type: 'scatter',
    mode: 'lines',
    x: [0, maxQuantile],
    y: [lenthsData.sme, lenthsData.sme],
    line: {
      color: '#ef4444',
      width: 2,
      dash: 'dot'
    },
    name: `SME = ${lenthsData.sme.toFixed(4)}`,
    hovertemplate: `Simultaneous Margin of Error (SME)<br>Threshold: ${lenthsData.sme.toFixed(4)}<extra></extra>`
  })

  const layout = {
    title: {
      text: "Half-Normal Plot (Lenth's Method)",
      font: {
        size: 20,
        color: '#f1f5f9'
      }
    },
    xaxis: {
      title: 'Half-Normal Quantile',
      gridcolor: '#475569',
      zerolinecolor: '#64748b',
      color: '#e2e8f0'
    },
    yaxis: {
      title: 'Absolute Effect',
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
      x: 0.02,
      y: 0.98,
      bgcolor: 'rgba(30, 41, 59, 0.8)',
      bordercolor: '#64748b',
      borderwidth: 1
    },
    margin: {
      l: 60,
      r: 20,
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
      filename: `half-normal-plot-${new Date().toISOString().split('T')[0]}`,
      height: 800,
      width: 1200,
      scale: 2
    }
  }

  return (
    <div className="bg-slate-700/50 rounded-lg p-6">
      <div className="mb-4">
        <h4 className="text-gray-100 font-semibold text-lg">Half-Normal Plot for Effect Screening</h4>
        <p className="text-gray-400 text-sm mt-1">
          Using Lenth's method to identify significant effects without replication
        </p>
      </div>

      <div className="bg-slate-800/50 rounded-lg p-4 mb-4">
        <Plot
          data={traces}
          layout={layout}
          config={config}
          style={{ width: '100%', height: '500px' }}
          useResizeHandler={true}
        />
      </div>

      {/* Statistics Summary */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <div className="bg-slate-800/50 rounded-lg p-4">
          <h5 className="text-gray-100 font-semibold text-sm mb-3">Lenth's Statistics</h5>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">PSE (Pseudo Standard Error):</span>
              <span className="text-gray-100 font-mono">{lenthsData.pse.toFixed(4)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">ME (Margin of Error):</span>
              <span className="text-yellow-400 font-mono">{lenthsData.me.toFixed(4)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">SME (Simultaneous ME):</span>
              <span className="text-red-400 font-mono">{lenthsData.sme.toFixed(4)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Number of Effects:</span>
              <span className="text-gray-100 font-mono">{lenthsData.n_effects}</span>
            </div>
          </div>
        </div>

        <div className="bg-slate-800/50 rounded-lg p-4">
          <h5 className="text-gray-100 font-semibold text-sm mb-3">Significant Effects</h5>
          <div className="space-y-2 text-sm">
            {lenthsData.significant_effects_sme.length > 0 ? (
              <div>
                <span className="text-red-400 font-semibold">Highly Significant (SME):</span>
                <div className="mt-1 flex flex-wrap gap-1">
                  {lenthsData.significant_effects_sme.map((effect, idx) => (
                    <span key={idx} className="px-2 py-1 bg-red-900/30 text-red-300 rounded text-xs font-mono border border-red-700/30">
                      {effect}
                    </span>
                  ))}
                </div>
              </div>
            ) : null}

            {lenthsData.significant_effects_me.filter(e => !lenthsData.significant_effects_sme.includes(e)).length > 0 ? (
              <div className="mt-2">
                <span className="text-yellow-400 font-semibold">Potentially Significant (ME):</span>
                <div className="mt-1 flex flex-wrap gap-1">
                  {lenthsData.significant_effects_me
                    .filter(e => !lenthsData.significant_effects_sme.includes(e))
                    .map((effect, idx) => (
                      <span key={idx} className="px-2 py-1 bg-yellow-900/30 text-yellow-300 rounded text-xs font-mono border border-yellow-700/30">
                        {effect}
                      </span>
                    ))}
                </div>
              </div>
            ) : null}

            {lenthsData.significant_effects_sme.length === 0 && lenthsData.significant_effects_me.length === 0 ? (
              <p className="text-gray-400 italic">No significant effects detected</p>
            ) : null}
          </div>
        </div>
      </div>

      {/* Interpretation Guide */}
      <div className="bg-slate-800/50 rounded-lg p-4">
        <h5 className="text-gray-100 font-semibold text-sm mb-2">Interpretation Guide</h5>
        <div className="text-gray-300 text-sm space-y-2">
          <p>
            <strong className="text-gray-100">How to read this plot:</strong> In an unreplicated factorial design, we can't estimate pure error from replicates.
            Lenth's method uses a robust estimate of error based on the effects themselves.
          </p>
          <ul className="list-disc list-inside space-y-1 ml-2">
            <li>
              <span className="text-gray-400">Gray points</span> near the reference line represent inactive/insignificant effects (noise)
            </li>
            <li>
              <span className="text-yellow-400">Yellow points</span> (diamonds) exceed the ME threshold - potentially active effects (± = {lenthsData.alpha})
            </li>
            <li>
              <span className="text-red-400">Red points</span> (diamonds) exceed the SME threshold - highly significant effects with multiple comparison adjustment
            </li>
            <li>
              Points far from the reference line indicate <strong>real factor effects</strong>, not random variation
            </li>
          </ul>
          <p className="mt-2">
            <strong className="text-purple-400">PSE = {lenthsData.pse.toFixed(4)}</strong> is the pseudo standard error,
            calculated as 1.5 × median of small effects, providing a robust estimate of noise in unreplicated experiments.
          </p>
        </div>
      </div>
    </div>
  )
}

export default HalfNormalPlot
