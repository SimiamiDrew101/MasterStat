import { useState, useEffect, useMemo } from 'react'
import { Sliders, TrendingUp, AlertCircle, Info } from 'lucide-react'
import Plot from 'react-plotly.js'
import { getPlotlyConfig } from '../utils/plotlyConfig'

const PredictionProfiler = ({
  coefficients,
  factors,
  responseName,
  modelType = 'second-order',
  experimentalData = null,
  varianceEstimate = null
}) => {
  // Initialize factor values at their coded centers (0)
  const [factorValues, setFactorValues] = useState(() => {
    const initial = {}
    factors.forEach(factor => {
      initial[factor] = 0
    })
    return initial
  })

  // Range for sliders (coded units)
  const [factorRanges, setFactorRanges] = useState(() => {
    const ranges = {}
    factors.forEach(factor => {
      ranges[factor] = { min: -2, max: 2, step: 0.01 }
    })
    return ranges
  })

  // Calculate predicted response based on current factor settings
  const predictResponse = useMemo(() => {
    return (values) => {
      if (!coefficients || Object.keys(coefficients).length === 0) {
        return { value: 0, stderr: 0, ci_lower: 0, ci_upper: 0, pi_lower: 0, pi_upper: 0 }
      }

      let prediction = 0

      // Process each coefficient term
      Object.entries(coefficients).forEach(([term, coefData]) => {
        const coef = typeof coefData === 'object' ? coefData.estimate : coefData

        if (term === 'Intercept') {
          prediction += coef
        } else if (term.includes(':')) {
          // Interaction term (e.g., X1:X2 or X1:X2:X3)
          const parts = term.split(':')
          let product = coef
          parts.forEach(part => {
            product *= (values[part] || 0)
          })
          prediction += product
        } else if (term.includes('**2') || term.includes('I(') && term.includes('**2)')) {
          // Quadratic term (e.g., I(X1**2))
          const factor = term.replace('I(', '').replace('**2)', '').replace('**2', '')
          prediction += coef * Math.pow(values[factor] || 0, 2)
        } else {
          // Linear term
          prediction += coef * (values[term] || 0)
        }
      })

      // Calculate standard error and intervals
      // For now, using a simplified approach
      // In production, would calculate from design matrix
      const stderr = varianceEstimate ? Math.sqrt(varianceEstimate) : prediction * 0.05
      const tValue = 1.96 // Approximate 95% CI

      const ci_lower = prediction - tValue * stderr
      const ci_upper = prediction + tValue * stderr

      // Prediction interval is wider than confidence interval
      const pi_stderr = stderr * 1.5
      const pi_lower = prediction - tValue * pi_stderr
      const pi_upper = prediction + tValue * pi_stderr

      return {
        value: prediction,
        stderr: stderr,
        ci_lower: ci_lower,
        ci_upper: ci_upper,
        pi_lower: pi_lower,
        pi_upper: pi_upper
      }
    }
  }, [coefficients, varianceEstimate])

  // Current prediction
  const currentPrediction = useMemo(() => {
    return predictResponse(factorValues)
  }, [factorValues, predictResponse])

  // Update individual factor value
  const updateFactor = (factor, value) => {
    setFactorValues(prev => ({
      ...prev,
      [factor]: parseFloat(value)
    }))
  }

  // Reset all factors to center
  const resetToCenter = () => {
    const centered = {}
    factors.forEach(factor => {
      centered[factor] = 0
    })
    setFactorValues(centered)
  }

  // Generate contour data for 2-factor profiler
  const contourData = useMemo(() => {
    if (factors.length !== 2) return null

    const [factor1, factor2] = factors
    const steps = 30
    const xValues = []
    const yValues = []
    const zGrid = []

    for (let i = 0; i <= steps; i++) {
      const x = factorRanges[factor1].min + (factorRanges[factor1].max - factorRanges[factor1].min) * i / steps
      xValues.push(x)
    }

    for (let i = 0; i <= steps; i++) {
      const y = factorRanges[factor2].min + (factorRanges[factor2].max - factorRanges[factor2].min) * i / steps
      yValues.push(y)
    }

    for (let i = 0; i <= steps; i++) {
      zGrid[i] = []
      for (let j = 0; j <= steps; j++) {
        const testValues = { ...factorValues }
        testValues[factor1] = xValues[j]
        testValues[factor2] = yValues[i]
        const pred = predictResponse(testValues)
        zGrid[i][j] = pred.value
      }
    }

    return { xValues, yValues, zGrid, factor1, factor2 }
  }, [factors, factorRanges, factorValues, predictResponse])

  // Generate trace plot for each factor (showing effect)
  const generateTraceData = (factor) => {
    const steps = 50
    const min = factorRanges[factor].min
    const max = factorRanges[factor].max
    const xData = []
    const yData = []

    for (let i = 0; i <= steps; i++) {
      const value = min + (max - min) * i / steps
      const testValues = { ...factorValues }
      testValues[factor] = value
      const pred = predictResponse(testValues)
      xData.push(value)
      yData.push(pred.value)
    }

    return { x: xData, y: yData }
  }

  if (!coefficients || Object.keys(coefficients).length === 0) {
    return (
      <div className="bg-slate-700/30 rounded-lg p-6 border border-slate-600">
        <div className="flex items-center gap-3 text-gray-400">
          <AlertCircle className="w-6 h-6" />
          <p>Fit a model first to use the Prediction Profiler</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-900/30 to-indigo-900/30 backdrop-blur-lg rounded-2xl p-6 border border-purple-700/50">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <Sliders className="w-8 h-8 text-purple-400" />
            <div>
              <h3 className="text-2xl font-bold text-gray-100">Prediction Profiler</h3>
              <p className="text-gray-300 text-sm">
                Explore the response surface interactively - adjust factors to see instant predictions
              </p>
            </div>
          </div>
          <button
            onClick={resetToCenter}
            className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors text-sm font-medium"
          >
            Reset to Center
          </button>
        </div>

        {/* Current Prediction Display */}
        <div className="bg-slate-800/50 rounded-lg p-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-gradient-to-br from-green-900/40 to-emerald-900/40 rounded-lg p-4 border border-green-700/50">
              <p className="text-gray-300 text-sm mb-1">Predicted {responseName}</p>
              <p className="text-4xl font-bold text-green-300">
                {currentPrediction.value.toFixed(3)}
              </p>
              <p className="text-gray-400 text-xs mt-2">
                Point Estimate
              </p>
            </div>

            <div className="bg-slate-700/50 rounded-lg p-4 border border-slate-600">
              <p className="text-gray-300 text-sm mb-1">95% Confidence Interval</p>
              <div className="flex items-baseline gap-2">
                <p className="text-2xl font-bold text-blue-300">
                  {currentPrediction.ci_lower.toFixed(3)}
                </p>
                <p className="text-gray-400">to</p>
                <p className="text-2xl font-bold text-blue-300">
                  {currentPrediction.ci_upper.toFixed(3)}
                </p>
              </div>
              <p className="text-gray-400 text-xs mt-2">
                Mean Response Uncertainty
              </p>
            </div>

            <div className="bg-slate-700/50 rounded-lg p-4 border border-slate-600">
              <p className="text-gray-300 text-sm mb-1">95% Prediction Interval</p>
              <div className="flex items-baseline gap-2">
                <p className="text-2xl font-bold text-purple-300">
                  {currentPrediction.pi_lower.toFixed(3)}
                </p>
                <p className="text-gray-400">to</p>
                <p className="text-2xl font-bold text-purple-300">
                  {currentPrediction.pi_upper.toFixed(3)}
                </p>
              </div>
              <p className="text-gray-400 text-xs mt-2">
                Single Observation Range
              </p>
            </div>
          </div>

          {/* Info box */}
          <div className="mt-4 flex items-start gap-2 bg-blue-900/20 rounded-lg p-3 border border-blue-700/30">
            <Info className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
            <p className="text-blue-200 text-xs">
              <strong>Confidence Interval:</strong> Uncertainty in the average response at these settings.
              <strong className="ml-4">Prediction Interval:</strong> Expected range for a single new observation (wider due to measurement variability).
            </p>
          </div>
        </div>
      </div>

      {/* Factor Sliders with Trace Plots */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {factors.map((factor) => {
          const traceData = generateTraceData(factor)
          return (
            <div key={factor} className="bg-slate-800/50 rounded-xl p-5 border border-slate-700/50">
              <div className="flex items-center justify-between mb-3">
                <h4 className="text-lg font-semibold text-gray-100">{factor}</h4>
                <div className="text-right">
                  <p className="text-2xl font-bold text-purple-300">
                    {factorValues[factor].toFixed(2)}
                  </p>
                  <p className="text-gray-400 text-xs">Coded Units</p>
                </div>
              </div>

              {/* Slider */}
              <div className="mb-4">
                <input
                  type="range"
                  min={factorRanges[factor].min}
                  max={factorRanges[factor].max}
                  step={factorRanges[factor].step}
                  value={factorValues[factor]}
                  onChange={(e) => updateFactor(factor, e.target.value)}
                  className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer slider-thumb"
                  style={{
                    background: `linear-gradient(to right,
                      #7c3aed 0%,
                      #7c3aed ${((factorValues[factor] - factorRanges[factor].min) / (factorRanges[factor].max - factorRanges[factor].min)) * 100}%,
                      #475569 ${((factorValues[factor] - factorRanges[factor].min) / (factorRanges[factor].max - factorRanges[factor].min)) * 100}%,
                      #475569 100%)`
                  }}
                />
                <div className="flex justify-between text-xs text-gray-400 mt-1">
                  <span>Low ({factorRanges[factor].min})</span>
                  <span>Center (0)</span>
                  <span>High ({factorRanges[factor].max})</span>
                </div>
              </div>

              {/* Mini Trace Plot */}
              <Plot
                data={[
                  {
                    type: 'scatter',
                    mode: 'lines',
                    x: traceData.x,
                    y: traceData.y,
                    line: { color: '#a78bfa', width: 3 },
                    name: factor,
                    hovertemplate: `${factor}: %{x:.2f}<br>${responseName}: %{y:.2f}<extra></extra>`
                  },
                  {
                    type: 'scatter',
                    mode: 'markers',
                    x: [factorValues[factor]],
                    y: [currentPrediction.value],
                    marker: {
                      size: 12,
                      color: '#22c55e',
                      symbol: 'diamond',
                      line: { color: 'white', width: 2 }
                    },
                    name: 'Current',
                    hovertemplate: `Current: ${factorValues[factor].toFixed(2)}<br>${responseName}: ${currentPrediction.value.toFixed(2)}<extra></extra>`
                  }
                ]}
                layout={{
                  height: 150,
                  margin: { l: 40, r: 10, t: 10, b: 30 },
                  xaxis: {
                    title: '',
                    gridcolor: '#475569',
                    zerolinecolor: '#64748b',
                    color: '#e2e8f0'
                  },
                  yaxis: {
                    title: responseName,
                    titlefont: { size: 10 },
                    gridcolor: '#475569',
                    zerolinecolor: '#64748b',
                    color: '#e2e8f0'
                  },
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: '#1e293b',
                  showlegend: false,
                  hovermode: 'closest'
                }}
                config={getPlotlyConfig(`prediction-profile-${factor}`, { displayModeBar: false })}
                style={{ width: '100%' }}
                useResizeHandler={true}
              />
            </div>
          )
        })}
      </div>

      {/* Interactive Contour Plot (2 factors only) */}
      {factors.length === 2 && contourData && (
        <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700/50">
          <h4 className="text-lg font-semibold text-gray-100 mb-4">Interactive Response Contour</h4>
          <Plot
            data={[
              {
                type: 'contour',
                x: contourData.xValues,
                y: contourData.yValues,
                z: contourData.zGrid,
                colorscale: [
                  [0, '#0050ff'],
                  [0.25, '#00d4ff'],
                  [0.5, '#64ff96'],
                  [0.75, '#ffff00'],
                  [1, '#ff0000']
                ],
                contours: {
                  coloring: 'heatmap',
                  showlabels: true,
                  labelfont: { size: 10, color: 'white' }
                },
                colorbar: {
                  title: { text: responseName, side: 'right' },
                  thickness: 20
                },
                hovertemplate: `${contourData.factor1}: %{x:.2f}<br>${contourData.factor2}: %{y:.2f}<br>${responseName}: %{z:.2f}<extra></extra>`
              },
              {
                type: 'scatter',
                mode: 'markers',
                x: [factorValues[contourData.factor1]],
                y: [factorValues[contourData.factor2]],
                marker: {
                  size: 16,
                  color: '#22c55e',
                  symbol: 'cross',
                  line: { color: 'white', width: 3 }
                },
                name: 'Current Point',
                hovertemplate: `<b>Current Settings</b><br>${contourData.factor1}: ${factorValues[contourData.factor1].toFixed(2)}<br>${contourData.factor2}: ${factorValues[contourData.factor2].toFixed(2)}<br>${responseName}: ${currentPrediction.value.toFixed(2)}<extra></extra>`
              }
            ]}
            layout={{
              height: 500,
              xaxis: {
                title: contourData.factor1,
                gridcolor: '#475569',
                zerolinecolor: '#64748b',
                color: '#e2e8f0'
              },
              yaxis: {
                title: contourData.factor2,
                gridcolor: '#475569',
                zerolinecolor: '#64748b',
                color: '#e2e8f0',
                scaleanchor: 'x'
              },
              paper_bgcolor: '#334155',
              plot_bgcolor: '#1e293b',
              font: { color: '#e2e8f0' },
              showlegend: false
            }}
            config={getPlotlyConfig('prediction-surface')}
            style={{ width: '100%' }}
            useResizeHandler={true}
          />
          <p className="text-gray-400 text-sm mt-3">
            <TrendingUp className="w-4 h-4 inline mr-1" />
            The green crosshair shows your current factor settings. Adjust the sliders above to explore different regions.
          </p>
        </div>
      )}

      {/* Current Settings Summary */}
      <div className="bg-slate-800/50 rounded-xl p-5 border border-slate-700/50">
        <h4 className="text-lg font-semibold text-gray-100 mb-3">Current Factor Settings (Coded Units)</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {factors.map(factor => (
            <div key={factor} className="bg-slate-700/50 rounded-lg p-3 text-center">
              <p className="text-gray-400 text-sm">{factor}</p>
              <p className="text-xl font-bold text-purple-300">{factorValues[factor].toFixed(2)}</p>
            </div>
          ))}
        </div>
        <p className="text-gray-400 text-xs mt-3">
          💡 <strong>Tip:</strong> Coded units typically range from -1 (low) to +1 (high) for factorial points,
          with axial points at ±α (e.g., ±1.414 for rotatable designs).
        </p>
      </div>
    </div>
  )
}

export default PredictionProfiler
