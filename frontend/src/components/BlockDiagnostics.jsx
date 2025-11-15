import { AlertCircle, CheckCircle, TrendingUp } from 'lucide-react'
import Plot from 'react-plotly.js'

/**
 * BlockDiagnostics component displays block-specific diagnostic tests and plots
 * Includes normality test, homogeneity test, and block-treatment interaction plot
 */
const BlockDiagnostics = ({ normalityTest, homogeneityTest, interactionMeans, blockType }) => {
  if (!normalityTest && !homogeneityTest && !interactionMeans) return null

  // Prepare interaction plot data
  const hasInteractionData = interactionMeans && Object.keys(interactionMeans).length > 0
  let interactionPlotData = null

  if (hasInteractionData) {
    // Group data by block and treatment
    const blocks = [...new Set(Object.values(interactionMeans).map(d => d.block))]
    const treatments = [...new Set(Object.values(interactionMeans).map(d => d.treatment))]

    // Create traces for each block
    interactionPlotData = blocks.map(block => {
      const blockData = Object.values(interactionMeans).filter(d => d.block === block)
      const sortedData = treatments.map(treatment => {
        const point = blockData.find(d => d.treatment === treatment)
        return point ? point.mean : null
      })

      return {
        type: 'scatter',
        mode: 'lines+markers',
        name: `Block ${block}`,
        x: treatments,
        y: sortedData,
        line: { width: 3 },
        marker: { size: 10, line: { color: '#1e293b', width: 2 } },
        hovertemplate: `Block ${block}<br>Treatment: %{x}<br>Mean: %{y:.4f}<extra></extra>`
      }
    })
  }

  return (
    <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
      <div className="flex items-center gap-2 mb-6">
        <AlertCircle className="w-6 h-6 text-orange-400" />
        <h3 className="text-xl font-bold text-gray-100">Block-Specific Diagnostics</h3>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Normality Test */}
        {normalityTest && !normalityTest.error && (
          <div className="bg-slate-700/30 rounded-lg p-5">
            <div className="flex items-center gap-2 mb-3">
              {normalityTest.interpretation === 'Normal' ? (
                <CheckCircle className="w-5 h-5 text-green-400" />
              ) : (
                <AlertCircle className="w-5 h-5 text-orange-400" />
              )}
              <h4 className="font-semibold text-gray-100">{normalityTest.test}</h4>
            </div>

            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">Test Statistic:</span>
                <span className="text-gray-200 font-mono">{normalityTest.statistic}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">p-value:</span>
                <span className={`font-mono font-semibold ${
                  normalityTest.p_value < 0.05 ? 'text-orange-400' : 'text-green-400'
                }`}>
                  {normalityTest.p_value.toFixed(6)}
                </span>
              </div>
              <div className="flex justify-between items-center mt-3 pt-3 border-t border-slate-600">
                <span className="text-gray-400">Interpretation:</span>
                <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                  normalityTest.interpretation === 'Normal'
                    ? 'bg-green-500/20 text-green-400'
                    : 'bg-orange-500/20 text-orange-400'
                }`}>
                  {normalityTest.interpretation}
                </span>
              </div>
            </div>

            <div className="mt-4 bg-slate-800/50 rounded p-3">
              <p className="text-xs text-gray-400">
                <strong className="text-gray-300">Note:</strong> Tests if residuals follow a normal distribution.
                Normal residuals indicate the ANOVA assumptions are met (p &gt; 0.05 is good).
              </p>
            </div>
          </div>
        )}

        {/* Homogeneity Test */}
        {homogeneityTest && !homogeneityTest.error && (
          <div className="bg-slate-700/30 rounded-lg p-5">
            <div className="flex items-center gap-2 mb-3">
              {homogeneityTest.interpretation === 'Homogeneous' ? (
                <CheckCircle className="w-5 h-5 text-green-400" />
              ) : (
                <AlertCircle className="w-5 h-5 text-orange-400" />
              )}
              <h4 className="font-semibold text-gray-100">{homogeneityTest.test}</h4>
            </div>

            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">Test Statistic:</span>
                <span className="text-gray-200 font-mono">{homogeneityTest.statistic}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">p-value:</span>
                <span className={`font-mono font-semibold ${
                  homogeneityTest.p_value < 0.05 ? 'text-orange-400' : 'text-green-400'
                }`}>
                  {homogeneityTest.p_value.toFixed(6)}
                </span>
              </div>
              <div className="flex justify-between items-center mt-3 pt-3 border-t border-slate-600">
                <span className="text-gray-400">Interpretation:</span>
                <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                  homogeneityTest.interpretation === 'Homogeneous'
                    ? 'bg-green-500/20 text-green-400'
                    : 'bg-orange-500/20 text-orange-400'
                }`}>
                  {homogeneityTest.interpretation}
                </span>
              </div>
            </div>

            <div className="mt-4 bg-slate-800/50 rounded p-3">
              <p className="text-xs text-gray-400">
                <strong className="text-gray-300">Note:</strong> Tests if variance is equal across blocks.
                Homogeneous variance validates the blocking structure (p &gt; 0.05 is good).
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Block-Treatment Interaction Plot */}
      {hasInteractionData && (
        <div className="mt-6 bg-slate-700/30 rounded-lg p-5">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="w-5 h-5 text-cyan-400" />
            <h4 className="font-semibold text-gray-100">Block × Treatment Interaction Plot</h4>
          </div>

          <p className="text-sm text-gray-400 mb-4">
            This plot shows how treatment effects vary across blocks. Parallel lines indicate no block-treatment interaction.
            Crossing or diverging lines suggest interaction effects.
          </p>

          <Plot
            data={interactionPlotData}
            layout={{
              paper_bgcolor: '#334155',
              plot_bgcolor: '#1e293b',
              font: { color: '#e2e8f0' },
              xaxis: {
                title: 'Treatment',
                gridcolor: '#475569',
                zerolinecolor: '#64748b',
                color: '#e2e8f0'
              },
              yaxis: {
                title: 'Mean Response',
                gridcolor: '#475569',
                zerolinecolor: '#64748b',
                color: '#e2e8f0'
              },
              showlegend: true,
              legend: {
                bgcolor: 'rgba(30, 41, 59, 0.8)',
                bordercolor: '#64748b',
                borderwidth: 1,
                x: 1,
                xanchor: 'right',
                y: 1
              },
              hovermode: 'closest',
              height: 400,
              margin: { l: 60, r: 40, b: 60, t: 40 }
            }}
            config={{
              responsive: true,
              displayModeBar: true,
              displaylogo: false,
              modeBarButtonsToRemove: ['lasso2d', 'select2d'],
              toImageButtonOptions: {
                format: 'png',
                filename: `block-treatment-interaction-${new Date().toISOString().split('T')[0]}`,
                height: 400,
                width: 700,
                scale: 2
              }
            }}
            style={{ width: '100%' }}
          />

          <div className="mt-3 bg-slate-800/50 rounded p-3">
            <p className="text-xs text-gray-300 mb-2">
              <strong>Interpretation Guide:</strong>
            </p>
            <ul className="text-xs text-gray-400 space-y-1 ml-4">
              <li>• <strong>Parallel lines:</strong> No interaction - blocking is effective</li>
              <li>• <strong>Slight divergence:</strong> Minor interaction - acceptable for most cases</li>
              <li>• <strong>Crossing lines:</strong> Strong interaction - blocking may mask treatment effects</li>
            </ul>
          </div>
        </div>
      )}

      {/* Overall Summary */}
      <div className="mt-6 bg-blue-900/20 border border-blue-700/50 rounded-lg p-4">
        <h5 className="font-semibold text-blue-200 mb-2">Diagnostic Summary</h5>
        <p className="text-sm text-blue-100/90">
          These diagnostics help validate the assumptions of your {blockType === 'random' ? 'random' : 'fixed'} blocks design:
        </p>
        <ul className="text-sm text-blue-100/80 mt-2 space-y-1 ml-4">
          <li>• <strong>Normality:</strong> Residuals should be normally distributed</li>
          <li>• <strong>Homogeneity:</strong> Variance should be equal across blocks</li>
          <li>• <strong>No Interaction:</strong> Treatment effects should be consistent across blocks</li>
        </ul>
      </div>
    </div>
  )
}

export default BlockDiagnostics
