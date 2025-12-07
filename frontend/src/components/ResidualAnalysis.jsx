import Plot from 'react-plotly.js'
import { AlertCircle, CheckCircle, Activity } from 'lucide-react'
import { getPlotlyConfig } from '../utils/plotlyConfig'

const ResidualAnalysis = ({ diagnostics, responseName }) => {
  if (!diagnostics) return null

  const {
    residuals,
    fitted,
    fitted_values = fitted, // Support both field names for backwards compatibility
    standardized_residuals,
    studentized_residuals,
    leverage,
    cooks_distance,
    observed_values,
    tests
  } = diagnostics

  // Prepare data for plots
  const n = residuals.length
  const runOrder = Array.from({ length: n }, (_, i) => i + 1)

  // 1. Residuals vs Fitted Values
  const residualsVsFittedTrace = {
    type: 'scatter',
    mode: 'markers',
    x: fitted_values,
    y: residuals,
    marker: {
      size: 8,
      color: residuals.map(r => Math.abs(r)),
      colorscale: 'Viridis',
      showscale: false,
      line: {
        color: '#1e293b',
        width: 1
      }
    },
    text: runOrder.map((i, idx) => `Run ${i}<br>Fitted: ${fitted_values[idx]}<br>Residual: ${residuals[idx]}`),
    hovertemplate: '%{text}<extra></extra>'
  }

  // Add reference line at y=0
  const zeroLineTrace = {
    type: 'scatter',
    mode: 'lines',
    x: [Math.min(...fitted_values), Math.max(...fitted_values)],
    y: [0, 0],
    line: {
      color: '#ef4444',
      width: 2,
      dash: 'dash'
    },
    showlegend: false,
    hoverinfo: 'skip'
  }

  // 2. Normal Q-Q Plot
  const sortedResiduals = [...standardized_residuals].sort((a, b) => a - b)
  const theoreticalQuantiles = sortedResiduals.map((_, i) => {
    // Calculate theoretical quantiles from standard normal
    const p = (i + 0.5) / n
    // Approximate inverse CDF for standard normal
    return p < 0.5
      ? -Math.sqrt(-2 * Math.log(p))
      : Math.sqrt(-2 * Math.log(1 - p))
  })

  const qqTrace = {
    type: 'scatter',
    mode: 'markers',
    x: theoreticalQuantiles,
    y: sortedResiduals,
    marker: {
      size: 8,
      color: '#3b82f6',
      line: {
        color: '#1e293b',
        width: 1
      }
    },
    name: 'Sample',
    hovertemplate: 'Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>'
  }

  // Add reference line for perfect normality
  const qqLineTrace = {
    type: 'scatter',
    mode: 'lines',
    x: theoreticalQuantiles,
    y: theoreticalQuantiles,
    line: {
      color: '#ef4444',
      width: 2,
      dash: 'dash'
    },
    name: 'Normal',
    showlegend: false,
    hoverinfo: 'skip'
  }

  // 3. Scale-Location Plot (sqrt of absolute standardized residuals)
  const sqrtAbsStdResiduals = standardized_residuals.map(r => Math.sqrt(Math.abs(r)))

  const scaleLocationTrace = {
    type: 'scatter',
    mode: 'markers',
    x: fitted_values,
    y: sqrtAbsStdResiduals,
    marker: {
      size: 8,
      color: '#10b981',
      line: {
        color: '#1e293b',
        width: 1
      }
    },
    text: runOrder.map((i, idx) => `Run ${i}<br>Fitted: ${fitted_values[idx]}<br>√|Std Residual|: ${sqrtAbsStdResiduals[idx].toFixed(2)}`),
    hovertemplate: '%{text}<extra></extra>'
  }

  // 4. Residuals vs Leverage (Cook's distance contours)
  const residualsVsLeverageTrace = {
    type: 'scatter',
    mode: 'markers',
    x: leverage,
    y: studentized_residuals,
    marker: {
      size: cooks_distance.map(d => 8 + d * 100),
      color: cooks_distance,
      colorscale: 'Hot',
      showscale: true,
      colorbar: {
        title: "Cook's D",
        thickness: 15,
        len: 0.5
      },
      line: {
        color: '#1e293b',
        width: 1
      }
    },
    text: runOrder.map((i, idx) => `Run ${i}<br>Leverage: ${leverage[idx].toFixed(4)}<br>Studentized Resid: ${studentized_residuals[idx].toFixed(2)}<br>Cook's D: ${cooks_distance[idx].toFixed(4)}`),
    hovertemplate: '%{text}<extra></extra>'
  }

  // 5. Histogram of Residuals
  const histogramTrace = {
    type: 'histogram',
    x: residuals,
    nbinsx: Math.min(20, Math.ceil(Math.sqrt(n))),
    marker: {
      color: '#8b5cf6',
      line: {
        color: '#1e293b',
        width: 1
      }
    },
    name: 'Residuals'
  }

  // Common layout properties
  const commonLayout = {
    paper_bgcolor: '#334155',
    plot_bgcolor: '#1e293b',
    font: {
      color: '#e2e8f0'
    },
    xaxis: {
      gridcolor: '#475569',
      zerolinecolor: '#64748b'
    },
    yaxis: {
      gridcolor: '#475569',
      zerolinecolor: '#64748b'
    },
    margin: { l: 60, r: 40, b: 60, t: 40 }
  }

  const config = getPlotlyConfig('residual-analysis', {
    modeBarButtonsToRemove: ['lasso2d', 'select2d']
  })

  return (
    <div className="space-y-6">
      {/* Diagnostic Tests Summary */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <div className="flex items-center gap-2 mb-4">
          <Activity className="w-5 h-5 text-blue-400" />
          <h3 className="text-xl font-bold text-gray-100">Model Diagnostic Tests</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Shapiro-Wilk Test */}
          <div className="bg-slate-700/30 rounded-lg p-4">
            <h4 className="text-gray-200 font-semibold mb-2">Shapiro-Wilk Test (Normality)</h4>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-400">Statistic:</span>
                <span className="text-gray-100 font-mono">{tests.shapiro_wilk.statistic}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">p-value:</span>
                <span className={`font-mono ${tests.shapiro_wilk.p_value > 0.05 ? 'text-green-400' : 'text-red-400'}`}>
                  {tests.shapiro_wilk.p_value}
                </span>
              </div>
              <div className="flex items-center gap-2 mt-3">
                {tests.shapiro_wilk.p_value > 0.05 ? (
                  <CheckCircle className="w-5 h-5 text-green-400" />
                ) : (
                  <AlertCircle className="w-5 h-5 text-red-400" />
                )}
                <span className="text-sm text-gray-300">{tests.shapiro_wilk.interpretation}</span>
              </div>
            </div>
          </div>

          {/* Durbin-Watson Test */}
          <div className="bg-slate-700/30 rounded-lg p-4">
            <h4 className="text-gray-200 font-semibold mb-2">Durbin-Watson Test (Autocorrelation)</h4>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-400">Statistic:</span>
                <span className="text-gray-100 font-mono">{tests.durbin_watson.statistic}</span>
              </div>
              <div className="text-xs text-gray-500 mt-2">
                Range: 0-4 (2 = no autocorrelation)
              </div>
              <div className="flex items-center gap-2 mt-3">
                {1.5 < tests.durbin_watson.statistic && tests.durbin_watson.statistic < 2.5 ? (
                  <CheckCircle className="w-5 h-5 text-green-400" />
                ) : (
                  <AlertCircle className="w-5 h-5 text-orange-400" />
                )}
                <span className="text-sm text-gray-300">{tests.durbin_watson.interpretation}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Residual Plots Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 1. Residuals vs Fitted */}
        <div className="bg-slate-700/50 rounded-lg p-4">
          <h4 className="text-gray-100 font-semibold mb-2">Residuals vs Fitted Values</h4>
          <Plot
            data={[residualsVsFittedTrace, zeroLineTrace]}
            layout={{
              ...commonLayout,
              xaxis: { ...commonLayout.xaxis, title: 'Fitted Values' },
              yaxis: { ...commonLayout.yaxis, title: 'Residuals' },
              height: 400
            }}
            config={config}
            style={{ width: '100%' }}
          />
          <p className="text-xs text-gray-400 mt-2">
            Check for patterns. Points should be randomly scattered around zero.
          </p>
        </div>

        {/* 2. Normal Q-Q Plot */}
        <div className="bg-slate-700/50 rounded-lg p-4">
          <h4 className="text-gray-100 font-semibold mb-2">Normal Q-Q Plot</h4>
          <Plot
            data={[qqTrace, qqLineTrace]}
            layout={{
              ...commonLayout,
              xaxis: { ...commonLayout.xaxis, title: 'Theoretical Quantiles' },
              yaxis: { ...commonLayout.yaxis, title: 'Sample Quantiles' },
              height: 400
            }}
            config={config}
            style={{ width: '100%' }}
          />
          <p className="text-xs text-gray-400 mt-2">
            Points should follow the diagonal line if residuals are normally distributed.
          </p>
        </div>

        {/* 3. Scale-Location */}
        <div className="bg-slate-700/50 rounded-lg p-4">
          <h4 className="text-gray-100 font-semibold mb-2">Scale-Location Plot</h4>
          <Plot
            data={[scaleLocationTrace]}
            layout={{
              ...commonLayout,
              xaxis: { ...commonLayout.xaxis, title: 'Fitted Values' },
              yaxis: { ...commonLayout.yaxis, title: '√|Standardized Residuals|' },
              height: 400
            }}
            config={config}
            style={{ width: '100%' }}
          />
          <p className="text-xs text-gray-400 mt-2">
            Check for homoscedasticity (constant variance). Should show horizontal band.
          </p>
        </div>

        {/* 4. Residuals vs Leverage */}
        <div className="bg-slate-700/50 rounded-lg p-4">
          <h4 className="text-gray-100 font-semibold mb-2">Residuals vs Leverage</h4>
          <Plot
            data={[residualsVsLeverageTrace]}
            layout={{
              ...commonLayout,
              xaxis: { ...commonLayout.xaxis, title: 'Leverage' },
              yaxis: { ...commonLayout.yaxis, title: 'Studentized Residuals' },
              height: 400
            }}
            config={config}
            style={{ width: '100%' }}
          />
          <p className="text-xs text-gray-400 mt-2">
            Identifies influential points. Large marker size indicates high Cook's distance.
          </p>
        </div>
      </div>

      {/* Histogram */}
      <div className="bg-slate-700/50 rounded-lg p-4">
        <h4 className="text-gray-100 font-semibold mb-2">Distribution of Residuals</h4>
        <Plot
          data={[histogramTrace]}
          layout={{
            ...commonLayout,
            xaxis: { ...commonLayout.xaxis, title: 'Residuals' },
            yaxis: { ...commonLayout.yaxis, title: 'Frequency' },
            height: 300,
            showlegend: false
          }}
          config={config}
          style={{ width: '100%' }}
        />
        <p className="text-xs text-gray-400 mt-2">
          Should approximate a normal distribution (bell curve).
        </p>
      </div>

      {/* Interpretation Guide */}
      <div className="bg-blue-900/20 rounded-lg p-5 border border-blue-700/30">
        <h4 className="text-blue-200 font-semibold mb-3">Interpretation Guide</h4>
        <ul className="space-y-2 text-sm text-blue-100">
          <li><strong>Good Model:</strong> Residuals randomly scattered, normally distributed, constant variance</li>
          <li><strong>Patterns in Residuals vs Fitted:</strong> May indicate missing terms or non-linearity</li>
          <li><strong>Non-normal Q-Q Plot:</strong> Consider transformations or check for outliers</li>
          <li><strong>Funnel shape in Scale-Location:</strong> Indicates non-constant variance (heteroscedasticity)</li>
          <li><strong>High Leverage + Large Residual:</strong> Influential point that may affect model</li>
        </ul>
      </div>
    </div>
  )
}

export default ResidualAnalysis
