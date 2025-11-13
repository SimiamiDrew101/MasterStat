import Plot from 'react-plotly.js'
import { AlertCircle, CheckCircle, HelpCircle } from 'lucide-react'

const AssumptionsPanel = ({ assumptions, alpha = 0.05 }) => {
  if (!assumptions) return null

  const { normality_tests, variance_test, qq_plot_data, check } = assumptions

  return (
    <div className="space-y-6">
      {/* Assumptions Summary */}
      <div className={`rounded-2xl p-6 border ${
        check?.all_assumptions_met
          ? 'bg-green-900/20 border-green-700/50'
          : 'bg-orange-900/20 border-orange-700/50'
      }`}>
        <div className="flex items-center gap-3 mb-4">
          {check?.all_assumptions_met ? (
            <CheckCircle className="w-6 h-6 text-green-400" />
          ) : (
            <AlertCircle className="w-6 h-6 text-orange-400" />
          )}
          <h3 className="text-xl font-bold text-gray-100">
            Assumptions Check
          </h3>
        </div>

        <div className="space-y-3">
          {check?.violations && check.violations.length > 0 ? (
            <div className="space-y-2">
              <p className="text-gray-200 font-semibold">Violations Detected:</p>
              <ul className="list-disc list-inside space-y-1">
                {check.violations.map((violation, idx) => (
                  <li key={idx} className="text-orange-200 text-sm">{violation}</li>
                ))}
              </ul>
            </div>
          ) : (
            <p className="text-green-200 font-semibold">✓ All assumptions met</p>
          )}

          {check?.recommendations && check.recommendations.length > 0 && (
            <div className="mt-4 space-y-2">
              <p className="text-gray-200 font-semibold">Recommendations:</p>
              <ul className="list-disc list-inside space-y-1">
                {check.recommendations.map((rec, idx) => (
                  <li key={idx} className="text-gray-300 text-sm">{rec}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>

      {/* Normality Tests */}
      {normality_tests && Object.keys(normality_tests).length > 0 && (
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
          <div className="flex items-center gap-2 mb-4">
            <HelpCircle className="w-5 h-5 text-purple-400" />
            <h3 className="text-xl font-bold text-gray-100">Normality Tests</h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(normality_tests).map(([sampleName, test]) => {
              if (!test) return null

              const passed = test.is_normal
              const sampleLabel = sampleName === 'sample1' ? 'Sample 1' : 'Sample 2'

              return (
                <div
                  key={sampleName}
                  className={`rounded-lg p-4 border ${
                    passed ? 'bg-green-900/20 border-green-700/30' : 'bg-red-900/20 border-red-700/30'
                  }`}
                >
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="font-semibold text-gray-100">{sampleLabel}</h4>
                    {passed ? (
                      <CheckCircle className="w-5 h-5 text-green-400" />
                    ) : (
                      <AlertCircle className="w-5 h-5 text-red-400" />
                    )}
                  </div>

                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-300">Test:</span>
                      <span className="text-gray-100 font-mono">{test.test_name}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">W Statistic:</span>
                      <span className="text-gray-100 font-mono">{test.statistic.toFixed(4)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">p-value:</span>
                      <span className={`font-mono ${passed ? 'text-green-300' : 'text-red-300'}`}>
                        {test.p_value.toFixed(4)}
                      </span>
                    </div>
                    <div className="mt-3 pt-3 border-t border-gray-600">
                      <p className={`text-xs ${passed ? 'text-green-200' : 'text-red-200'}`}>
                        {passed
                          ? `✓ Cannot reject normality (p > ${alpha})`
                          : `✗ Evidence against normality (p < ${alpha})`}
                      </p>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>

          <div className="mt-4 bg-blue-900/20 rounded-lg p-4 border border-blue-700/30">
            <p className="text-blue-200 text-sm">
              <strong>Shapiro-Wilk Test:</strong> Tests the null hypothesis that the data was drawn from a normal distribution.
              A p-value greater than α ({alpha}) suggests the data is consistent with normality.
            </p>
          </div>
        </div>
      )}

      {/* Variance Equality Test */}
      {variance_test && (
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
          <div className="flex items-center gap-2 mb-4">
            <HelpCircle className="w-5 h-5 text-indigo-400" />
            <h3 className="text-xl font-bold text-gray-100">Variance Equality Test</h3>
          </div>

          <div className={`rounded-lg p-4 border ${
            variance_test.equal_variances
              ? 'bg-green-900/20 border-green-700/30'
              : 'bg-red-900/20 border-red-700/30'
          }`}>
            <div className="flex items-center justify-between mb-3">
              <h4 className="font-semibold text-gray-100">{variance_test.test_name}</h4>
              {variance_test.equal_variances ? (
                <CheckCircle className="w-5 h-5 text-green-400" />
              ) : (
                <AlertCircle className="w-5 h-5 text-red-400" />
              )}
            </div>

            <div className="grid grid-cols-2 gap-4 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-300">Statistic:</span>
                <span className="text-gray-100 font-mono">{variance_test.statistic.toFixed(4)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">p-value:</span>
                <span className={`font-mono ${variance_test.equal_variances ? 'text-green-300' : 'text-red-300'}`}>
                  {variance_test.p_value.toFixed(4)}
                </span>
              </div>
            </div>

            <div className="mt-3 pt-3 border-t border-gray-600">
              <p className={`text-xs ${variance_test.equal_variances ? 'text-green-200' : 'text-red-200'}`}>
                {variance_test.equal_variances
                  ? `✓ Equal variances assumption met (p > ${alpha})`
                  : `✗ Variances may be unequal (p < ${alpha}) - consider Welch's t-test`}
              </p>
            </div>
          </div>

          <div className="mt-4 bg-blue-900/20 rounded-lg p-4 border border-blue-700/30">
            <p className="text-blue-200 text-sm">
              <strong>Levene's Test:</strong> Tests the null hypothesis that population variances are equal.
              A p-value greater than α ({alpha}) suggests equal variances.
            </p>
          </div>
        </div>
      )}

      {/* Q-Q Plots */}
      {qq_plot_data && Object.keys(qq_plot_data).length > 0 && (
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
          <div className="flex items-center gap-2 mb-4">
            <HelpCircle className="w-5 h-5 text-cyan-400" />
            <h3 className="text-xl font-bold text-gray-100">Q-Q Plots (Normality Assessment)</h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {Object.entries(qq_plot_data).map(([sampleName, data]) => {
              if (!data) return null

              const sampleLabel = sampleName === 'sample1' ? 'Sample 1' : 'Sample 2'

              // Q-Q plot traces
              const traces = [
                // Actual Q-Q points
                {
                  type: 'scatter',
                  mode: 'markers',
                  x: data.theoretical,
                  y: data.observed,
                  marker: {
                    size: 8,
                    color: '#3b82f6',
                    opacity: 0.7
                  },
                  name: 'Sample quantiles',
                  hovertemplate: 'Theoretical: %{x:.2f}<br>Observed: %{y:.2f}<extra></extra>'
                },
                // Reference line (fitted line from normal distribution)
                {
                  type: 'scatter',
                  mode: 'lines',
                  x: data.theoretical,
                  y: data.fit_line
                    ? data.theoretical.map(x => data.fit_line.slope * x + data.fit_line.intercept)
                    : data.theoretical,
                  line: {
                    color: '#ef4444',
                    width: 2,
                    dash: 'dash'
                  },
                  name: 'Normal fit',
                  hoverinfo: 'skip'
                }
              ]

              const layout = {
                title: {
                  text: sampleLabel,
                  font: {
                    size: 14,
                    color: '#f1f5f9'
                  }
                },
                xaxis: {
                  title: 'Theoretical Quantiles',
                  gridcolor: '#475569',
                  color: '#e2e8f0'
                },
                yaxis: {
                  title: 'Sample Quantiles',
                  gridcolor: '#475569',
                  color: '#e2e8f0'
                },
                paper_bgcolor: '#334155',
                plot_bgcolor: '#1e293b',
                font: {
                  color: '#e2e8f0'
                },
                margin: { l: 60, r: 20, b: 60, t: 40 },
                height: 350,
                showlegend: false
              }

              const config = {
                responsive: true,
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['lasso2d', 'select2d']
              }

              return (
                <div key={sampleName}>
                  <Plot
                    data={traces}
                    layout={layout}
                    config={config}
                    style={{ width: '100%' }}
                  />
                </div>
              )
            })}
          </div>

          <div className="mt-4 bg-blue-900/20 rounded-lg p-4 border border-blue-700/30">
            <p className="text-blue-200 text-sm">
              <strong>Q-Q Plot Interpretation:</strong> Points should fall close to the red reference line if data is normally distributed.
              The line represents the expected pattern for a normal distribution fitted to your data.
              Systematic deviations (S-curves, heavy tails, or skewness) indicate departures from normality.
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

export default AssumptionsPanel
