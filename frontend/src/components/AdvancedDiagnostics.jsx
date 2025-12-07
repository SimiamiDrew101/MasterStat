import { useState, useEffect } from 'react'
import Plot from 'react-plotly.js'
import { AlertTriangle, CheckCircle, Info, TrendingUp } from 'lucide-react'
import { getPlotlyConfig } from '../utils/plotlyConfig'

const AdvancedDiagnostics = ({ diagnosticsData }) => {
  const [activeView, setActiveView] = useState('summary') // 'summary', 'leverage', 'cooks', 'dffits', 'vif', 'press'

  if (!diagnosticsData) {
    return (
      <div className="bg-slate-700/30 rounded-lg p-6 text-center">
        <Info className="w-12 h-12 text-gray-400 mx-auto mb-3" />
        <p className="text-gray-400">Fit a model to view advanced diagnostics</p>
      </div>
    )
  }

  const { diagnostics, press, summary, recommendations, interpretation } = diagnosticsData

  // Render summary cards
  const renderSummaryCards = () => (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <div className={`rounded-lg p-4 ${
        summary.n_high_leverage > 0 ? 'bg-yellow-900/30 border border-yellow-700/50' : 'bg-green-900/30 border border-green-700/50'
      }`}>
        <p className="text-gray-300 text-sm">High Leverage Points</p>
        <p className="text-3xl font-bold text-gray-100">{summary.n_high_leverage}</p>
        <p className="text-xs text-gray-400 mt-1">Threshold: {summary.leverage_threshold}</p>
      </div>

      <div className={`rounded-lg p-4 ${
        summary.n_influential_cooks > 0 ? 'bg-orange-900/30 border border-orange-700/50' : 'bg-green-900/30 border border-green-700/50'
      }`}>
        <p className="text-gray-300 text-sm">Influential (Cook's D)</p>
        <p className="text-3xl font-bold text-gray-100">{summary.n_influential_cooks}</p>
        <p className="text-xs text-gray-400 mt-1">Threshold: {summary.cooks_threshold}</p>
      </div>

      <div className={`rounded-lg p-4 ${
        summary.n_influential_dffits > 0 ? 'bg-red-900/30 border border-red-700/50' : 'bg-green-900/30 border border-green-700/50'
      }`}>
        <p className="text-gray-300 text-sm">Influential (DFFITS)</p>
        <p className="text-3xl font-bold text-gray-100">{summary.n_influential_dffits}</p>
        <p className="text-xs text-gray-400 mt-1">Threshold: {summary.dffits_threshold}</p>
      </div>

      <div className={`rounded-lg p-4 ${
        summary.n_multicollinearity_issues > 0 ? 'bg-purple-900/30 border border-purple-700/50' : 'bg-green-900/30 border border-green-700/50'
      }`}>
        <p className="text-gray-300 text-sm">Multicollinearity Issues</p>
        <p className="text-3xl font-bold text-gray-100">{summary.n_multicollinearity_issues}</p>
        <p className="text-xs text-gray-400 mt-1">VIF &gt; 5</p>
      </div>
    </div>
  )

  // Render PRESS statistics
  const renderPRESS = () => (
    <div className="bg-gradient-to-r from-indigo-900/30 to-purple-900/30 rounded-lg p-6 border border-indigo-700/50">
      <h4 className="text-xl font-bold text-gray-100 mb-4">PRESS Statistics (Cross-Validation)</h4>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-slate-700/50 rounded-lg p-4">
          <p className="text-gray-400 text-sm">PRESS Value</p>
          <p className="text-3xl font-bold text-indigo-300">{press.value}</p>
          <p className="text-xs text-gray-400 mt-2">Prediction Error Sum of Squares</p>
        </div>
        <div className="bg-slate-700/50 rounded-lg p-4">
          <p className="text-gray-400 text-sm">R² Prediction</p>
          <p className="text-3xl font-bold text-purple-300">{press.r2_prediction}</p>
          <p className="text-xs text-gray-400 mt-2">{press.interpretation}</p>
        </div>
      </div>
      <div className="mt-4 bg-slate-700/30 rounded-lg p-4">
        <p className="text-gray-300 text-sm">
          <strong>Interpretation:</strong> R² prediction measures how well the model predicts new observations using leave-one-out cross-validation.
          Values &gt; 0.9 are excellent, 0.7-0.9 are good, &lt; 0.5 suggest poor predictive ability.
        </p>
      </div>
    </div>
  )

  // Render Leverage diagnostics
  const renderLeverage = () => {
    const leverageValues = diagnostics.leverage.map(item => item.leverage)
    const observations = diagnostics.leverage.map(item => item.observation)
    const colors = diagnostics.leverage.map(item => {
      if (item.status === 'high') return 'red'
      if (item.status === 'moderate') return 'orange'
      return 'green'
    })

    return (
      <div className="space-y-4">
        <div className="bg-slate-700/30 rounded-lg p-4">
          <h4 className="text-xl font-bold text-gray-100 mb-2">Leverage (Hat Values)</h4>
          <p className="text-gray-300 text-sm mb-4">
            Leverage measures how far an observation's factor values are from the center of the design space.
            High leverage points have greater potential to influence the model fit.
          </p>
          <Plot
            data={[
              {
                type: 'bar',
                x: observations,
                y: leverageValues,
                marker: { color: colors },
                text: leverageValues.map(v => v.toFixed(4)),
                textposition: 'auto',
                hovertemplate: 'Observation %{x}<br>Leverage: %{y:.6f}<extra></extra>'
              },
              {
                type: 'scatter',
                mode: 'lines',
                x: [0, observations.length + 1],
                y: [summary.leverage_threshold, summary.leverage_threshold],
                line: { color: 'orange', dash: 'dash', width: 2 },
                name: 'Moderate Threshold',
                showlegend: true
              },
              {
                type: 'scatter',
                mode: 'lines',
                x: [0, observations.length + 1],
                y: [summary.high_leverage_threshold, summary.high_leverage_threshold],
                line: { color: 'red', dash: 'dash', width: 2 },
                name: 'High Threshold',
                showlegend: true
              }
            ]}
            layout={{
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              font: { color: '#e5e7eb' },
              xaxis: { title: 'Observation', gridcolor: '#374151' },
              yaxis: { title: 'Leverage (Hat Value)', gridcolor: '#374151' },
              showlegend: true,
              legend: { x: 1, xanchor: 'right', y: 1 },
              margin: { l: 60, r: 40, t: 40, b: 60 }
            }}
            config={getPlotlyConfig('leverage-plot')}
            className="w-full"
          />
        </div>

        {/* Leverage Table */}
        <div className="bg-slate-700/30 rounded-lg overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="bg-slate-700/70">
                <th className="px-4 py-2 text-left text-gray-100 font-semibold border-b border-slate-600">Observation</th>
                <th className="px-4 py-2 text-right text-gray-100 font-semibold border-b border-slate-600">Leverage</th>
                <th className="px-4 py-2 text-center text-gray-100 font-semibold border-b border-slate-600">Status</th>
              </tr>
            </thead>
            <tbody>
              {diagnostics.leverage
                .filter(item => item.status !== 'normal')
                .map((item, idx) => (
                  <tr key={idx} className="border-b border-slate-700/30 hover:bg-slate-600/10">
                    <td className="px-4 py-2 text-gray-100">{item.observation}</td>
                    <td className="px-4 py-2 text-right text-gray-100">{item.leverage}</td>
                    <td className="px-4 py-2 text-center">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        item.status === 'high' ? 'bg-red-900/50 text-red-200' : 'bg-orange-900/50 text-orange-200'
                      }`}>
                        {item.status.toUpperCase()}
                      </span>
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
          {diagnostics.leverage.filter(item => item.status !== 'normal').length === 0 && (
            <div className="p-4 text-center text-gray-400">
              All observations have normal leverage
            </div>
          )}
        </div>
      </div>
    )
  }

  // Render Cook's Distance
  const renderCooksDistance = () => {
    const cooksValues = diagnostics.cooks_distance.map(item => item.cooks_distance)
    const observations = diagnostics.cooks_distance.map(item => item.observation)
    const colors = diagnostics.cooks_distance.map(item => {
      if (item.status === 'highly_influential') return 'red'
      if (item.status === 'influential') return 'orange'
      return 'green'
    })

    return (
      <div className="space-y-4">
        <div className="bg-slate-700/30 rounded-lg p-4">
          <h4 className="text-xl font-bold text-gray-100 mb-2">Cook's Distance</h4>
          <p className="text-gray-300 text-sm mb-4">
            Cook's Distance measures the overall influence of each observation on the model.
            Values &gt; 1 are highly concerning, values &gt; 4/n warrant investigation.
          </p>
          <Plot
            data={[
              {
                type: 'bar',
                x: observations,
                y: cooksValues,
                marker: { color: colors },
                text: cooksValues.map(v => v.toFixed(6)),
                textposition: 'auto',
                hovertemplate: 'Observation %{x}<br>Cook\'s D: %{y:.6f}<extra></extra>'
              },
              {
                type: 'scatter',
                mode: 'lines',
                x: [0, observations.length + 1],
                y: [summary.cooks_threshold, summary.cooks_threshold],
                line: { color: 'orange', dash: 'dash', width: 2 },
                name: 'Influential Threshold',
                showlegend: true
              },
              {
                type: 'scatter',
                mode: 'lines',
                x: [0, observations.length + 1],
                y: [1, 1],
                line: { color: 'red', dash: 'dash', width: 2 },
                name: 'Highly Influential',
                showlegend: true
              }
            ]}
            layout={{
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              font: { color: '#e5e7eb' },
              xaxis: { title: 'Observation', gridcolor: '#374151' },
              yaxis: { title: 'Cook\'s Distance', gridcolor: '#374151' },
              showlegend: true,
              legend: { x: 1, xanchor: 'right', y: 1 },
              margin: { l: 60, r: 40, t: 40, b: 60 }
            }}
            config={getPlotlyConfig('cooks-distance-plot')}
            className="w-full"
          />
        </div>

        {/* Cook's Distance Table */}
        <div className="bg-slate-700/30 rounded-lg overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="bg-slate-700/70">
                <th className="px-4 py-2 text-left text-gray-100 font-semibold border-b border-slate-600">Observation</th>
                <th className="px-4 py-2 text-right text-gray-100 font-semibold border-b border-slate-600">Cook's Distance</th>
                <th className="px-4 py-2 text-center text-gray-100 font-semibold border-b border-slate-600">Status</th>
              </tr>
            </thead>
            <tbody>
              {diagnostics.cooks_distance
                .filter(item => item.status !== 'normal')
                .map((item, idx) => (
                  <tr key={idx} className="border-b border-slate-700/30 hover:bg-slate-600/10">
                    <td className="px-4 py-2 text-gray-100">{item.observation}</td>
                    <td className="px-4 py-2 text-right text-gray-100">{item.cooks_distance}</td>
                    <td className="px-4 py-2 text-center">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        item.status === 'highly_influential' ? 'bg-red-900/50 text-red-200' : 'bg-orange-900/50 text-orange-200'
                      }`}>
                        {item.status.replace('_', ' ').toUpperCase()}
                      </span>
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
          {diagnostics.cooks_distance.filter(item => item.status !== 'normal').length === 0 && (
            <div className="p-4 text-center text-gray-400">
              No influential observations detected
            </div>
          )}
        </div>
      </div>
    )
  }

  // Render DFFITS
  const renderDFFITS = () => {
    const dffitsValues = diagnostics.dffits.map(item => item.dffits)
    const observations = diagnostics.dffits.map(item => item.observation)
    const colors = diagnostics.dffits.map(item => item.status === 'influential' ? 'red' : 'green')

    return (
      <div className="space-y-4">
        <div className="bg-slate-700/30 rounded-lg p-4">
          <h4 className="text-xl font-bold text-gray-100 mb-2">DFFITS</h4>
          <p className="text-gray-300 text-sm mb-4">
            DFFITS measures how much the predicted value changes when an observation is removed.
            Values exceeding ±{summary.dffits_threshold} are considered influential.
          </p>
          <Plot
            data={[
              {
                type: 'bar',
                x: observations,
                y: dffitsValues,
                marker: { color: colors },
                text: dffitsValues.map(v => v.toFixed(6)),
                textposition: 'auto',
                hovertemplate: 'Observation %{x}<br>DFFITS: %{y:.6f}<extra></extra>'
              },
              {
                type: 'scatter',
                mode: 'lines',
                x: [0, observations.length + 1],
                y: [summary.dffits_threshold, summary.dffits_threshold],
                line: { color: 'red', dash: 'dash', width: 2 },
                name: 'Upper Threshold',
                showlegend: true
              },
              {
                type: 'scatter',
                mode: 'lines',
                x: [0, observations.length + 1],
                y: [-summary.dffits_threshold, -summary.dffits_threshold],
                line: { color: 'red', dash: 'dash', width: 2 },
                name: 'Lower Threshold',
                showlegend: true
              }
            ]}
            layout={{
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              font: { color: '#e5e7eb' },
              xaxis: { title: 'Observation', gridcolor: '#374151' },
              yaxis: { title: 'DFFITS', gridcolor: '#374151' },
              showlegend: true,
              legend: { x: 1, xanchor: 'right', y: 1 },
              margin: { l: 60, r: 40, t: 40, b: 60 }
            }}
            config={getPlotlyConfig('dffits-plot')}
            className="w-full"
          />
        </div>

        {/* DFFITS Table */}
        <div className="bg-slate-700/30 rounded-lg overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="bg-slate-700/70">
                <th className="px-4 py-2 text-left text-gray-100 font-semibold border-b border-slate-600">Observation</th>
                <th className="px-4 py-2 text-right text-gray-100 font-semibold border-b border-slate-600">DFFITS</th>
                <th className="px-4 py-2 text-center text-gray-100 font-semibold border-b border-slate-600">Status</th>
              </tr>
            </thead>
            <tbody>
              {diagnostics.dffits
                .filter(item => item.status !== 'normal')
                .map((item, idx) => (
                  <tr key={idx} className="border-b border-slate-700/30 hover:bg-slate-600/10">
                    <td className="px-4 py-2 text-gray-100">{item.observation}</td>
                    <td className="px-4 py-2 text-right text-gray-100">{item.dffits}</td>
                    <td className="px-4 py-2 text-center">
                      <span className="px-2 py-1 rounded text-xs font-medium bg-red-900/50 text-red-200">
                        INFLUENTIAL
                      </span>
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
          {diagnostics.dffits.filter(item => item.status !== 'normal').length === 0 && (
            <div className="p-4 text-center text-gray-400">
              No influential observations detected
            </div>
          )}
        </div>
      </div>
    )
  }

  // Render VIF
  const renderVIF = () => (
    <div className="space-y-4">
      <div className="bg-slate-700/30 rounded-lg p-4">
        <h4 className="text-xl font-bold text-gray-100 mb-2">Variance Inflation Factor (VIF)</h4>
        <p className="text-gray-300 text-sm mb-4">
          VIF measures multicollinearity among predictors. VIF &gt; 10 indicates severe multicollinearity,
          VIF 5-10 indicates moderate multicollinearity, VIF &lt; 5 is generally acceptable.
        </p>
      </div>

      <div className="bg-slate-700/30 rounded-lg overflow-hidden">
        <table className="w-full">
          <thead>
            <tr className="bg-slate-700/70">
              <th className="px-4 py-2 text-left text-gray-100 font-semibold border-b border-slate-600">Term</th>
              <th className="px-4 py-2 text-right text-gray-100 font-semibold border-b border-slate-600">VIF</th>
              <th className="px-4 py-2 text-center text-gray-100 font-semibold border-b border-slate-600">Status</th>
            </tr>
          </thead>
          <tbody>
            {diagnostics.vif.map((item, idx) => (
              <tr key={idx} className="border-b border-slate-700/30 hover:bg-slate-600/10">
                <td className="px-4 py-2 text-gray-100 font-mono text-sm">{item.term}</td>
                <td className="px-4 py-2 text-right text-gray-100">{item.vif}</td>
                <td className="px-4 py-2 text-center">
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    item.status === 'severe_multicollinearity' ? 'bg-red-900/50 text-red-200' :
                    item.status === 'moderate_multicollinearity' ? 'bg-orange-900/50 text-orange-200' :
                    item.status === 'low_multicollinearity' ? 'bg-yellow-900/50 text-yellow-200' :
                    item.status === 'excellent' ? 'bg-green-900/50 text-green-200' :
                    'bg-gray-700 text-gray-400'
                  }`}>
                    {item.status.replace(/_/g, ' ').toUpperCase()}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )

  // Render recommendations
  const renderRecommendations = () => (
    <div className="bg-gradient-to-r from-blue-900/30 to-cyan-900/30 rounded-lg p-6 border border-blue-700/50">
      <h4 className="text-xl font-bold text-gray-100 mb-4 flex items-center gap-2">
        <TrendingUp className="w-5 h-5" />
        Recommendations
      </h4>
      <ul className="space-y-2">
        {recommendations.map((rec, idx) => (
          <li key={idx} className="flex items-start gap-2">
            {rec.includes('passed') ? (
              <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
            ) : (
              <AlertTriangle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
            )}
            <span className="text-gray-300">{rec}</span>
          </li>
        ))}
      </ul>
    </div>
  )

  return (
    <div className="space-y-6">
      {/* Summary */}
      {renderSummaryCards()}

      {/* PRESS Statistics */}
      {renderPRESS()}

      {/* Navigation Tabs */}
      <div className="flex flex-wrap gap-2 bg-slate-700/30 p-2 rounded-lg">
        <button
          onClick={() => setActiveView('leverage')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            activeView === 'leverage' ? 'bg-blue-600 text-white' : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
          }`}
        >
          Leverage
        </button>
        <button
          onClick={() => setActiveView('cooks')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            activeView === 'cooks' ? 'bg-blue-600 text-white' : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
          }`}
        >
          Cook's Distance
        </button>
        <button
          onClick={() => setActiveView('dffits')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            activeView === 'dffits' ? 'bg-blue-600 text-white' : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
          }`}
        >
          DFFITS
        </button>
        <button
          onClick={() => setActiveView('vif')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            activeView === 'vif' ? 'bg-blue-600 text-white' : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
          }`}
        >
          VIF
        </button>
      </div>

      {/* Diagnostic Views */}
      {activeView === 'leverage' && renderLeverage()}
      {activeView === 'cooks' && renderCooksDistance()}
      {activeView === 'dffits' && renderDFFITS()}
      {activeView === 'vif' && renderVIF()}

      {/* Recommendations */}
      {renderRecommendations()}

      {/* Interpretation */}
      <div className="bg-slate-700/30 rounded-lg p-4">
        <h4 className="text-gray-100 font-semibold mb-2">Overall Interpretation</h4>
        <p className="text-gray-300 text-sm">{interpretation}</p>
      </div>
    </div>
  )
}

export default AdvancedDiagnostics
