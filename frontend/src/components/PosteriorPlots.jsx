import { useMemo } from 'react'
import Plot from 'react-plotly.js'
import { getPlotlyConfig, getPlotlyLayout } from '../utils/plotlyConfig'
import { Activity, TrendingUp, BarChart2, Zap, AlertCircle, CheckCircle, Info } from 'lucide-react'

const PosteriorPlots = ({
  posteriorSamples,           // {param: [samples]}
  posteriorSummary,           // {param: {mean, std, lower_95, upper_95, hdi_width}}
  priors,                     // {param: {dist_type, params}}
  convergenceDiagnostics,     // {parameters: {param: {ess, autocorrelation}}}
  responseName = 'Response'
}) => {
  // Extract parameter names (exclude log_sigma)
  const parameters = useMemo(() =>
    Object.keys(posteriorSamples).filter(p => p !== 'log_sigma'),
    [posteriorSamples]
  )

  // Generate prior distribution curve for overlay
  const generatePriorCurve = (priorSpec) => {
    if (!priorSpec) return { x: [], y: [] }

    const x = []
    const y = []
    const nPoints = 200

    if (priorSpec.dist_type === 'normal') {
      const { loc, scale } = priorSpec.params
      const xMin = loc - 4 * scale
      const xMax = loc + 4 * scale

      for (let i = 0; i < nPoints; i++) {
        const xi = xMin + (xMax - xMin) * i / (nPoints - 1)
        const yi = (1 / (scale * Math.sqrt(2 * Math.PI))) *
                   Math.exp(-0.5 * Math.pow((xi - loc) / scale, 2))
        x.push(xi)
        y.push(yi)
      }
    } else if (priorSpec.dist_type === 'uniform') {
      const { low, high } = priorSpec.params
      const density = 1 / (high - low)
      x.push(low - 0.1 * (high - low), low, low, high, high, high + 0.1 * (high - low))
      y.push(0, 0, density, density, 0, 0)
    } else if (priorSpec.dist_type === 'cauchy') {
      const { loc, scale } = priorSpec.params
      const xMin = loc - 10 * scale
      const xMax = loc + 10 * scale

      for (let i = 0; i < nPoints; i++) {
        const xi = xMin + (xMax - xMin) * i / (nPoints - 1)
        const yi = 1 / (Math.PI * scale * (1 + Math.pow((xi - loc) / scale, 2)))
        x.push(xi)
        y.push(yi)
      }
    } else if (priorSpec.dist_type === 't') {
      const { df, loc, scale } = priorSpec.params
      const xMin = loc - 4 * scale
      const xMax = loc + 4 * scale

      for (let i = 0; i < nPoints; i++) {
        const xi = xMin + (xMax - xMin) * i / (nPoints - 1)
        const t = (xi - loc) / scale
        // Simplified t-distribution density (approximation)
        const yi = Math.exp(-0.5 * (df + 1) / df * t * t) / (scale * Math.sqrt(df * Math.PI))
        x.push(xi)
        y.push(yi)
      }
    }

    return { x, y }
  }

  // Calculate running mean for trace plots
  const calculateRunningMean = (samples) => {
    const runningMean = []
    let sum = 0
    for (let i = 0; i < samples.length; i++) {
      sum += samples[i]
      runningMean.push(sum / (i + 1))
    }
    return runningMean
  }

  // ESS color coding helper
  const getESSColor = (ess) => {
    if (ess > 400) return 'text-green-400'
    if (ess > 200) return 'text-yellow-400'
    return 'text-red-400'
  }

  const getESSBadge = (ess) => {
    if (ess > 400) return { color: 'bg-green-500/20 text-green-400', icon: CheckCircle, label: 'Good' }
    if (ess > 200) return { color: 'bg-yellow-500/20 text-yellow-400', icon: AlertCircle, label: 'Fair' }
    return { color: 'bg-red-500/20 text-red-400', icon: AlertCircle, label: 'Poor' }
  }

  return (
    <div className="space-y-6 mt-8">
      {/* Section Header */}
      <div className="flex items-center gap-3 mb-4">
        <Activity className="w-6 h-6 text-blue-400" />
        <h3 className="text-2xl font-bold text-gray-100">Posterior Analysis & Diagnostics</h3>
      </div>

      {/* Convergence Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {parameters.slice(0, 6).map((param) => {
          const diagnostics = convergenceDiagnostics?.parameters?.[param]
          if (!diagnostics) return null

          const badge = getESSBadge(diagnostics.ess)
          const BadgeIcon = badge.icon

          return (
            <div key={param} className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
              <div className="flex items-center justify-between mb-2">
                <h4 className="font-semibold text-gray-200">{param}</h4>
                <div className={`flex items-center gap-1 px-2 py-1 rounded ${badge.color} text-xs`}>
                  <BadgeIcon className="w-3 h-3" />
                  <span>{badge.label}</span>
                </div>
              </div>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">ESS:</span>
                  <span className={`font-mono ${getESSColor(diagnostics.ess)}`}>
                    {diagnostics.ess.toFixed(0)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">ESS %:</span>
                  <span className="text-gray-300 font-mono">
                    {(diagnostics.ess_per_sample * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">R-hat:</span>
                  <span className="text-gray-300 font-mono">{diagnostics.rhat.toFixed(3)}</span>
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {/* Overall Convergence Summary */}
      {convergenceDiagnostics?.overall_ess_min !== undefined && (
        <div className="bg-blue-900/20 border border-blue-700/30 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <Info className="w-5 h-5 text-blue-400 mt-0.5" />
            <div>
              <h4 className="font-semibold text-blue-300 mb-1">Overall Convergence</h4>
              <p className="text-sm text-gray-300">
                Minimum ESS across all parameters: <span className={`font-mono font-bold ${getESSColor(convergenceDiagnostics.overall_ess_min)}`}>
                  {convergenceDiagnostics.overall_ess_min.toFixed(0)}
                </span>
                {convergenceDiagnostics.overall_ess_min > 400 ?
                  ' - Excellent convergence! MCMC samples are reliable.' :
                  convergenceDiagnostics.overall_ess_min > 200 ?
                  ' - Acceptable convergence. Consider increasing n_samples for more precision.' :
                  ' - Poor convergence. Increase n_samples or check for model issues.'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Posterior Density Plots with Prior Overlay */}
      <div>
        <div className="flex items-center gap-2 mb-4">
          <BarChart2 className="w-5 h-5 text-purple-400" />
          <h4 className="text-xl font-bold text-gray-100">Posterior Distributions</h4>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {parameters.map((param) => {
            const samples = posteriorSamples[param]
            const summary = posteriorSummary[param]
            const prior = priors?.[param]

            // Create histogram
            const histData = {
              x: samples,
              type: 'histogram',
              name: 'Posterior',
              nbinsx: 40,
              marker: { color: 'rgba(99, 102, 241, 0.7)' },
              histnorm: 'probability density'
            }

            const traces = [histData]

            // Add prior overlay if available
            if (prior) {
              const priorCurve = generatePriorCurve(prior)
              if (priorCurve.x.length > 0) {
                traces.push({
                  x: priorCurve.x,
                  y: priorCurve.y,
                  type: 'scatter',
                  mode: 'lines',
                  name: 'Prior',
                  line: { color: 'rgba(34, 197, 94, 0.8)', width: 2, dash: 'dash' }
                })
              }
            }

            // Add HDI markers
            const maxY = Math.max(...samples.map(() => 1)) * 0.1  // Approximate max density
            traces.push({
              x: [summary.lower_95, summary.upper_95],
              y: [maxY, maxY],
              type: 'scatter',
              mode: 'lines+markers',
              name: '95% HDI',
              line: { color: 'rgba(239, 68, 68, 0.8)', width: 3 },
              marker: { size: 8, symbol: 'line-ns-open' }
            })

            const layout = getPlotlyLayout(param, {
              showlegend: true,
              legend: { x: 0.7, y: 1, bgcolor: 'rgba(15, 23, 42, 0.7)' },
              xaxis: { title: 'Parameter Value' },
              yaxis: { title: 'Density' },
              annotations: [{
                x: summary.mean,
                y: 0,
                xref: 'x',
                yref: 'paper',
                text: `Mean: ${summary.mean.toFixed(3)}`,
                showarrow: true,
                arrowhead: 2,
                ax: 0,
                ay: -40,
                bgcolor: 'rgba(15, 23, 42, 0.9)',
                bordercolor: 'rgba(99, 102, 241, 0.5)',
                borderwidth: 1,
                font: { color: '#e2e8f0', size: 10 }
              }]
            })

            return (
              <div key={param} className="bg-slate-800/30 rounded-lg p-3">
                <Plot
                  data={traces}
                  layout={layout}
                  config={getPlotlyConfig(`posterior-${param}`)}
                  style={{ width: '100%', height: '350px' }}
                />
                <div className="text-xs text-gray-400 mt-2 grid grid-cols-2 gap-2">
                  <div>Mean: <span className="text-gray-300 font-mono">{summary.mean.toFixed(4)}</span></div>
                  <div>Std: <span className="text-gray-300 font-mono">{summary.std.toFixed(4)}</span></div>
                  <div>HDI 95%: <span className="text-gray-300 font-mono">[{summary.lower_95.toFixed(3)}, {summary.upper_95.toFixed(3)}]</span></div>
                  <div>Width: <span className="text-gray-300 font-mono">{summary.hdi_width.toFixed(4)}</span></div>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Trace Plots with Running Mean */}
      <div>
        <div className="flex items-center gap-2 mb-4">
          <TrendingUp className="w-5 h-5 text-cyan-400" />
          <h4 className="text-xl font-bold text-gray-100">MCMC Trace Plots</h4>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {parameters.map((param) => {
            const samples = posteriorSamples[param]
            const runningMean = calculateRunningMean(samples)
            const iterations = samples.map((_, i) => i + 1)

            const traces = [
              {
                x: iterations,
                y: samples,
                type: 'scatter',
                mode: 'lines',
                name: 'Samples',
                line: { color: 'rgba(56, 189, 248, 0.6)', width: 1 }
              },
              {
                x: iterations,
                y: runningMean,
                type: 'scatter',
                mode: 'lines',
                name: 'Running Mean',
                line: { color: 'rgba(239, 68, 68, 0.9)', width: 2, dash: 'dash' }
              }
            ]

            const layout = getPlotlyLayout(param, {
              showlegend: true,
              legend: { x: 0.7, y: 1, bgcolor: 'rgba(15, 23, 42, 0.7)' },
              xaxis: { title: 'Iteration' },
              yaxis: { title: 'Parameter Value' }
            })

            return (
              <div key={param} className="bg-slate-800/30 rounded-lg p-3">
                <Plot
                  data={traces}
                  layout={layout}
                  config={getPlotlyConfig(`trace-${param}`)}
                  style={{ width: '100%', height: '300px' }}
                />
              </div>
            )
          })}
        </div>
      </div>

      {/* Autocorrelation Plots */}
      <div>
        <div className="flex items-center gap-2 mb-4">
          <Zap className="w-5 h-5 text-yellow-400" />
          <h4 className="text-xl font-bold text-gray-100">Autocorrelation Functions</h4>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {parameters.map((param) => {
            const diagnostics = convergenceDiagnostics?.parameters?.[param]
            if (!diagnostics?.autocorrelation) return null

            const acf = diagnostics.autocorrelation
            const lags = acf.map((_, i) => i)

            // Color-code bars by ACF magnitude
            const colors = acf.map(val => {
              if (Math.abs(val) > 0.2) return 'rgba(239, 68, 68, 0.8)'      // Red
              if (Math.abs(val) > 0.1) return 'rgba(251, 191, 36, 0.8)'     // Yellow
              return 'rgba(34, 197, 94, 0.8)'                                // Green
            })

            const traces = [{
              x: lags,
              y: acf,
              type: 'bar',
              name: 'ACF',
              marker: { color: colors }
            }]

            const layout = getPlotlyLayout(param, {
              showlegend: false,
              xaxis: { title: 'Lag' },
              yaxis: { title: 'Autocorrelation', range: [-0.1, 1.1] },
              shapes: [
                {
                  type: 'line',
                  x0: 0,
                  x1: lags.length,
                  y0: 0.1,
                  y1: 0.1,
                  line: { color: 'rgba(251, 191, 36, 0.5)', width: 1, dash: 'dash' }
                },
                {
                  type: 'line',
                  x0: 0,
                  x1: lags.length,
                  y0: -0.1,
                  y1: -0.1,
                  line: { color: 'rgba(251, 191, 36, 0.5)', width: 1, dash: 'dash' }
                }
              ]
            })

            return (
              <div key={param} className="bg-slate-800/30 rounded-lg p-3">
                <Plot
                  data={traces}
                  layout={layout}
                  config={getPlotlyConfig(`acf-${param}`)}
                  style={{ width: '100%', height: '300px' }}
                />
                <p className="text-xs text-gray-400 mt-2">
                  Good mixing shows ACF quickly decaying to near-zero (green bars)
                </p>
              </div>
            )
          })}
        </div>
      </div>

      {/* Interpretation Guide */}
      <div className="bg-slate-800/30 rounded-lg p-6 border border-slate-700/50">
        <h4 className="text-lg font-bold text-gray-100 mb-4 flex items-center gap-2">
          <Info className="w-5 h-5 text-blue-400" />
          Interpretation Guide
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <h5 className="font-semibold text-gray-200 mb-2">Effective Sample Size (ESS)</h5>
            <ul className="space-y-1 text-gray-300">
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                <span><strong>&gt;400:</strong> Excellent - reliable inference</span>
              </li>
              <li className="flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-yellow-400 mt-0.5 flex-shrink-0" />
                <span><strong>200-400:</strong> Acceptable - consider more samples</span>
              </li>
              <li className="flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-red-400 mt-0.5 flex-shrink-0" />
                <span><strong>&lt;200:</strong> Poor - increase n_samples</span>
              </li>
            </ul>
          </div>
          <div>
            <h5 className="font-semibold text-gray-200 mb-2">Trace Plots</h5>
            <ul className="space-y-1 text-gray-300">
              <li>• Good: Random scatter around mean (fuzzy caterpillar)</li>
              <li>• Bad: Trends, stuck values, slow mixing</li>
              <li>• Running mean should stabilize quickly</li>
            </ul>
          </div>
          <div>
            <h5 className="font-semibold text-gray-200 mb-2">Autocorrelation</h5>
            <ul className="space-y-1 text-gray-300">
              <li>• Good: ACF decays to near-zero within 10-20 lags</li>
              <li>• Bad: High ACF persists for many lags</li>
              <li>• Green bars (&lt;0.1) indicate good mixing</li>
            </ul>
          </div>
          <div>
            <h5 className="font-semibold text-gray-200 mb-2">HDI vs Percentiles</h5>
            <ul className="space-y-1 text-gray-300">
              <li>• HDI = narrowest 95% credible interval</li>
              <li>• Better for asymmetric posteriors</li>
              <li>• Red markers show HDI bounds</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

export default PosteriorPlots
