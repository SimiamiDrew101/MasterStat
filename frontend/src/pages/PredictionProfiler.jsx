import { useState, useEffect } from 'react'
import axios from 'axios'
import Plot from 'react-plotly.js'
import { Upload, Sliders, Target, TrendingUp, Download, CheckCircle } from 'lucide-react'
import { loadModelFromSession, clearModelFromSession } from '../utils/modelConverter'

const PredictionProfiler = () => {
  // State management
  const [model, setModel] = useState(null)
  const [factorLevels, setFactorLevels] = useState({})
  const [predictions, setPredictions] = useState(null)
  const [surfaceData, setSurfaceData] = useState(null)
  const [activeTab, setActiveTab] = useState('import')
  const [desirabilityGoals, setDesirabilityGoals] = useState([])
  const [optimizationResult, setOptimizationResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [selectedFactors, setSelectedFactors] = useState({ factor1: null, factor2: null })
  const [importedFrom, setImportedFrom] = useState(null)

  // Check for imported model from RSM on mount
  useEffect(() => {
    const importedModel = loadModelFromSession('profiler_model')
    if (importedModel) {
      handleLoadModel(importedModel)
      setImportedFrom(importedModel.source || 'Unknown')
      clearModelFromSession('profiler_model')  // Clear after loading
    }
  }, [])

  // Load model (from import or mock)
  const handleLoadModel = (loadedModel) => {
    setModel(loadedModel)

    // Initialize factor levels to center points
    const centerLevels = {}
    loadedModel.factors.forEach(factor => {
      const [min, max] = loadedModel.factor_ranges[factor]
      centerLevels[factor] = (min + max) / 2
    })
    setFactorLevels(centerLevels)

    // Set default surface plot factors
    if (loadedModel.factors.length >= 2) {
      setSelectedFactors({
        factor1: loadedModel.factors[0],
        factor2: loadedModel.factors[1]
      })
    }

    setActiveTab('predict')
  }

  // Mock model for testing
  const handleImportMockModel = () => {
    const mockModel = {
      model_type: 'rsm_quadratic',
      coefficients: {
        'Intercept': 50,
        'Temperature': 2.5,
        'Time': 1.8,
        'Temperature^2': -0.3,
        'Time^2': -0.2,
        'Temperature*Time': 0.5
      },
      factors: ['Temperature', 'Time'],
      factor_ranges: {
        'Temperature': [50, 90],
        'Time': [10, 30]
      },
      response_name: 'Yield',
      source: 'Example Model'
    }

    handleLoadModel(mockModel)
    setImportedFrom('Example Model')
  }

  // Real-time prediction when factor levels change
  useEffect(() => {
    if (model && Object.keys(factorLevels).length > 0) {
      handlePredict()
    }
  }, [factorLevels])

  // Generate surface when factors or levels change
  useEffect(() => {
    if (model && selectedFactors.factor1 && selectedFactors.factor2) {
      generateSurface()
    }
  }, [factorLevels, selectedFactors])

  const handlePredict = async () => {
    try {
      const response = await axios.post('/api/prediction-profiler/predict', {
        model,
        factor_levels: factorLevels
      })
      setPredictions(response.data)
      setError(null)
    } catch (err) {
      setError('Prediction failed: ' + err.message)
    }
  }

  const handleFactorChange = (factor, value) => {
    setFactorLevels(prev => ({
      ...prev,
      [factor]: parseFloat(value)
    }))
  }

  const generateSurface = async () => {
    if (!selectedFactors.factor1 || !selectedFactors.factor2) return

    try {
      // Fixed factors (all except the two being plotted)
      const fixedFactors = {}
      model.factors.forEach(f => {
        if (f !== selectedFactors.factor1 && f !== selectedFactors.factor2) {
          fixedFactors[f] = factorLevels[f]
        }
      })

      const response = await axios.post('/api/prediction-profiler/generate-surface', {
        model,
        factor1: selectedFactors.factor1,
        factor2: selectedFactors.factor2,
        factor_ranges: model.factor_ranges,
        fixed_factors: fixedFactors
      })
      setSurfaceData(response.data)
    } catch (err) {
      console.error('Surface generation failed:', err)
    }
  }

  const handleOptimize = async () => {
    if (desirabilityGoals.length === 0) {
      setError('Please add at least one desirability goal')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const response = await axios.post('/api/prediction-profiler/optimize-desirability', {
        models: [model],
        goals: desirabilityGoals,
        factor_ranges: model.factor_ranges
      })

      setOptimizationResult(response.data)
      setFactorLevels(response.data.optimal_settings)
      setActiveTab('predict')
    } catch (err) {
      setError('Optimization failed: ' + err.message)
    } finally {
      setLoading(false)
    }
  }

  const addDesirabilityGoal = () => {
    setDesirabilityGoals([...desirabilityGoals, {
      response: model.response_name,
      target_type: 'maximize',
      low: 0,
      high: 100
    }])
  }

  const updateDesirabilityGoal = (index, field, value) => {
    const updated = [...desirabilityGoals]
    updated[index][field] = value
    setDesirabilityGoals(updated)
  }

  const removeDesirabilityGoal = (index) => {
    setDesirabilityGoals(desirabilityGoals.filter((_, i) => i !== index))
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-pink-600 text-transparent bg-clip-text">
            Prediction Profiler
          </h1>
          <p className="text-gray-300">
            Interactive response prediction and optimization tool for exploring fitted models
          </p>
        </div>

        {/* Tab Navigation */}
        <div className="flex space-x-2 bg-slate-800/50 p-2 rounded-lg mb-6">
          <button
            onClick={() => setActiveTab('import')}
            className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors flex items-center justify-center space-x-2 ${
              activeTab === 'import'
                ? 'bg-purple-600 text-white'
                : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
            }`}
          >
            <Upload size={18} />
            <span>1. Import Model</span>
          </button>
          <button
            onClick={() => setActiveTab('predict')}
            disabled={!model}
            className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors flex items-center justify-center space-x-2 ${
              activeTab === 'predict'
                ? 'bg-purple-600 text-white'
                : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700 disabled:opacity-50'
            }`}
          >
            <TrendingUp size={18} />
            <span>2. Predict</span>
          </button>
          <button
            onClick={() => setActiveTab('optimize')}
            disabled={!model}
            className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors flex items-center justify-center space-x-2 ${
              activeTab === 'optimize'
                ? 'bg-purple-600 text-white'
                : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700 disabled:opacity-50'
            }`}
          >
            <Target size={18} />
            <span>3. Optimize</span>
          </button>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-6 bg-red-500/10 border border-red-500/50 rounded-lg p-4">
            <p className="text-red-400">{error}</p>
          </div>
        )}

        {/* Tab Content */}
        {activeTab === 'import' && (
          <div className="bg-slate-800/50 rounded-2xl p-6">
            <h2 className="text-2xl font-bold mb-4">Import Fitted Model</h2>
            <p className="text-gray-300 mb-6">
              Import a fitted model from Response Surface Methodology or Factorial Design analysis
            </p>

            <div className="space-y-4">
              {/* Show imported model info if available */}
              {model && importedFrom && (
                <div className="bg-green-500/10 border border-green-500/50 rounded-lg p-4">
                  <div className="flex items-start space-x-3">
                    <CheckCircle className="text-green-400 mt-1" size={20} />
                    <div>
                      <p className="text-green-300 font-semibold">Model Loaded Successfully</p>
                      <p className="text-gray-300 text-sm mt-1">
                        Source: <strong>{importedFrom}</strong>
                      </p>
                      <p className="text-gray-400 text-sm">
                        Response: {model.response_name} | Factors: {model.factors.join(', ')}
                      </p>
                      {model.r_squared && (
                        <p className="text-gray-400 text-sm">
                          R² = {model.r_squared.toFixed(4)} | RMSE = {model.rmse?.toFixed(4) || 'N/A'}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              )}

              <div className="bg-slate-700/50 rounded-lg p-6 border border-slate-600">
                <h3 className="text-lg font-semibold mb-3">Model Sources</h3>
                <ul className="space-y-2 text-gray-300 mb-4">
                  <li>• Response Surface (RSM) - Use "Open in Profiler" button from RSM results</li>
                  <li>• Factorial Designs - Full or fractional factorial models</li>
                  <li>• Example model - Test the profiler with sample data</li>
                </ul>

                <button
                  onClick={handleImportMockModel}
                  className="w-full px-6 py-3 bg-purple-600 hover:bg-purple-700 rounded-lg font-semibold transition-colors"
                >
                  Load Example Model (Temperature × Time → Yield)
                </button>
              </div>

              {!model && (
                <div className="bg-blue-500/10 border border-blue-500/50 rounded-lg p-4">
                  <p className="text-blue-300 text-sm">
                    <strong>Tip:</strong> After fitting a model in RSM or Factorial Designs, click "Open in Profiler" to import it here
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'predict' && model && (
          <div className="space-y-6">
            {/* Factor Sliders */}
            <div className="bg-slate-800/50 rounded-2xl p-6">
              <div className="flex items-center space-x-2 mb-4">
                <Sliders className="text-purple-400" size={24} />
                <h2 className="text-2xl font-bold">Adjust Factor Levels</h2>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {model.factors.map(factor => {
                  const [min, max] = model.factor_ranges[factor]
                  const value = factorLevels[factor] || (min + max) / 2

                  return (
                    <div key={factor} className="bg-slate-700/50 rounded-lg p-4">
                      <div className="flex justify-between items-center mb-2">
                        <label className="text-gray-200 font-medium">{factor}</label>
                        <span className="text-purple-400 font-bold text-lg">{value.toFixed(2)}</span>
                      </div>

                      <input
                        type="range"
                        min={min}
                        max={max}
                        step={(max - min) / 100}
                        value={value}
                        onChange={(e) => handleFactorChange(factor, e.target.value)}
                        className="w-full h-2 bg-slate-600 rounded-lg appearance-none cursor-pointer slider"
                      />

                      <div className="flex justify-between text-xs text-gray-400 mt-1">
                        <span>{min}</span>
                        <span>{max}</span>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* Prediction Results */}
            {predictions && (
              <div className="bg-slate-800/50 rounded-2xl p-6">
                <h2 className="text-2xl font-bold mb-4">Predicted Response</h2>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="bg-gradient-to-br from-purple-600/20 to-pink-600/20 rounded-lg p-6 border border-purple-500/50">
                    <p className="text-gray-300 text-sm mb-2">Prediction</p>
                    <p className="text-4xl font-bold text-purple-400">
                      {predictions.prediction.toFixed(3)}
                    </p>
                    <p className="text-gray-400 text-xs mt-2">{model.response_name}</p>
                  </div>

                  <div className="bg-slate-700/50 rounded-lg p-6">
                    <p className="text-gray-300 text-sm mb-2">95% CI Lower</p>
                    <p className="text-2xl font-bold text-gray-200">
                      {predictions.confidence_interval.lower.toFixed(3)}
                    </p>
                  </div>

                  <div className="bg-slate-700/50 rounded-lg p-6">
                    <p className="text-gray-300 text-sm mb-2">95% CI Upper</p>
                    <p className="text-2xl font-bold text-gray-200">
                      {predictions.confidence_interval.upper.toFixed(3)}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Response Surface Contour Plot */}
            {surfaceData && model.factors.length >= 2 && (
              <div className="bg-slate-800/50 rounded-2xl p-6">
                <h2 className="text-2xl font-bold mb-4">Response Surface</h2>

                {/* Factor selection for surface */}
                <div className="flex space-x-4 mb-4">
                  <div>
                    <label className="text-gray-300 text-sm">X-Axis Factor:</label>
                    <select
                      value={selectedFactors.factor1}
                      onChange={(e) => setSelectedFactors(prev => ({ ...prev, factor1: e.target.value }))}
                      className="ml-2 bg-slate-700 text-gray-200 rounded px-3 py-1"
                    >
                      {model.factors.map(f => (
                        <option key={f} value={f}>{f}</option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="text-gray-300 text-sm">Y-Axis Factor:</label>
                    <select
                      value={selectedFactors.factor2}
                      onChange={(e) => setSelectedFactors(prev => ({ ...prev, factor2: e.target.value }))}
                      className="ml-2 bg-slate-700 text-gray-200 rounded px-3 py-1"
                    >
                      {model.factors.map(f => (
                        <option key={f} value={f}>{f}</option>
                      ))}
                    </select>
                  </div>
                </div>

                <Plot
                  data={[
                    {
                      type: 'contour',
                      x: surfaceData.x,
                      y: surfaceData.y,
                      z: surfaceData.z,
                      colorscale: 'Viridis',
                      contours: {
                        showlabels: true,
                        labelfont: { size: 12, color: 'white' }
                      },
                      colorbar: {
                        title: model.response_name,
                        titlefont: { color: 'white' },
                        tickfont: { color: 'white' }
                      }
                    },
                    {
                      type: 'scatter',
                      x: [factorLevels[surfaceData.factor1]],
                      y: [factorLevels[surfaceData.factor2]],
                      mode: 'markers',
                      marker: {
                        size: 15,
                        color: 'red',
                        symbol: 'x',
                        line: { width: 2, color: 'white' }
                      },
                      name: 'Current Point',
                      showlegend: true
                    }
                  ]}
                  layout={{
                    xaxis: {
                      title: surfaceData.factor1,
                      titlefont: { color: 'white' },
                      tickfont: { color: 'white' },
                      gridcolor: 'rgba(255,255,255,0.1)'
                    },
                    yaxis: {
                      title: surfaceData.factor2,
                      titlefont: { color: 'white' },
                      tickfont: { color: 'white' },
                      gridcolor: 'rgba(255,255,255,0.1)'
                    },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(30,41,59,0.5)',
                    font: { color: 'white' },
                    height: 500,
                    margin: { t: 40, b: 60, l: 60, r: 40 }
                  }}
                  config={{ responsive: true }}
                  className="w-full"
                />
              </div>
            )}
          </div>
        )}

        {activeTab === 'optimize' && model && (
          <div className="space-y-6">
            <div className="bg-slate-800/50 rounded-2xl p-6">
              <h2 className="text-2xl font-bold mb-4">Desirability Optimization</h2>
              <p className="text-gray-300 mb-6">
                Define goals for response optimization (maximize, minimize, or target specific values)
              </p>

              {/* Desirability Goals */}
              <div className="space-y-4 mb-6">
                {desirabilityGoals.map((goal, index) => (
                  <div key={index} className="bg-slate-700/50 rounded-lg p-4 border border-slate-600">
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                      <div>
                        <label className="text-gray-300 text-sm">Target Type</label>
                        <select
                          value={goal.target_type}
                          onChange={(e) => updateDesirabilityGoal(index, 'target_type', e.target.value)}
                          className="w-full mt-1 bg-slate-600 text-gray-200 rounded px-3 py-2"
                        >
                          <option value="maximize">Maximize</option>
                          <option value="minimize">Minimize</option>
                          <option value="target">Target Value</option>
                        </select>
                      </div>

                      {goal.target_type !== 'target' && (
                        <>
                          <div>
                            <label className="text-gray-300 text-sm">Low Value</label>
                            <input
                              type="number"
                              value={goal.low}
                              onChange={(e) => updateDesirabilityGoal(index, 'low', parseFloat(e.target.value))}
                              className="w-full mt-1 bg-slate-600 text-gray-200 rounded px-3 py-2"
                            />
                          </div>
                          <div>
                            <label className="text-gray-300 text-sm">High Value</label>
                            <input
                              type="number"
                              value={goal.high}
                              onChange={(e) => updateDesirabilityGoal(index, 'high', parseFloat(e.target.value))}
                              className="w-full mt-1 bg-slate-600 text-gray-200 rounded px-3 py-2"
                            />
                          </div>
                        </>
                      )}

                      {goal.target_type === 'target' && (
                        <>
                          <div>
                            <label className="text-gray-300 text-sm">Target</label>
                            <input
                              type="number"
                              value={goal.target || 0}
                              onChange={(e) => updateDesirabilityGoal(index, 'target', parseFloat(e.target.value))}
                              className="w-full mt-1 bg-slate-600 text-gray-200 rounded px-3 py-2"
                            />
                          </div>
                          <div>
                            <label className="text-gray-300 text-sm">Tolerance</label>
                            <input
                              type="number"
                              value={goal.tolerance || 5}
                              onChange={(e) => updateDesirabilityGoal(index, 'tolerance', parseFloat(e.target.value))}
                              className="w-full mt-1 bg-slate-600 text-gray-200 rounded px-3 py-2"
                            />
                          </div>
                        </>
                      )}

                      <div className="flex items-end">
                        <button
                          onClick={() => removeDesirabilityGoal(index)}
                          className="w-full px-4 py-2 bg-red-600/20 hover:bg-red-600/30 text-red-400 rounded transition-colors"
                        >
                          Remove
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="flex space-x-4">
                <button
                  onClick={addDesirabilityGoal}
                  className="px-6 py-3 bg-slate-700 hover:bg-slate-600 rounded-lg font-semibold transition-colors"
                >
                  Add Goal
                </button>

                <button
                  onClick={handleOptimize}
                  disabled={loading || desirabilityGoals.length === 0}
                  className="flex-1 px-6 py-3 bg-purple-600 hover:bg-purple-700 rounded-lg font-semibold transition-colors disabled:opacity-50"
                >
                  {loading ? 'Optimizing...' : 'Find Optimal Settings'}
                </button>
              </div>
            </div>

            {/* Optimization Results */}
            {optimizationResult && (
              <div className="bg-slate-800/50 rounded-2xl p-6">
                <h2 className="text-2xl font-bold mb-4">Optimal Settings Found</h2>

                <div className="bg-green-500/10 border border-green-500/50 rounded-lg p-4 mb-6">
                  <p className="text-green-300">
                    Overall Desirability: <strong>{(optimizationResult.overall_desirability * 100).toFixed(1)}%</strong>
                  </p>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  {Object.entries(optimizationResult.optimal_settings).map(([factor, value]) => (
                    <div key={factor} className="bg-slate-700/50 rounded-lg p-4">
                      <p className="text-gray-400 text-sm">{factor}</p>
                      <p className="text-2xl font-bold text-purple-400">{value.toFixed(3)}</p>
                    </div>
                  ))}
                </div>

                <h3 className="text-lg font-semibold mb-3">Predicted Responses</h3>
                <div className="space-y-2">
                  {optimizationResult.predictions.map((pred, index) => (
                    <div key={index} className="bg-slate-700/50 rounded-lg p-3 flex justify-between items-center">
                      <span className="text-gray-300">{pred.response}</span>
                      <span className="text-xl font-bold text-gray-200">{pred.predicted_value.toFixed(3)}</span>
                    </div>
                  ))}
                </div>

                <button
                  onClick={() => setActiveTab('predict')}
                  className="w-full mt-6 px-6 py-3 bg-purple-600 hover:bg-purple-700 rounded-lg font-semibold transition-colors"
                >
                  View in Prediction Profiler
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Custom CSS for slider */}
      <style jsx>{`
        .slider::-webkit-slider-thumb {
          appearance: none;
          width: 20px;
          height: 20px;
          border-radius: 50%;
          background: #9333ea;
          cursor: pointer;
        }
        .slider::-moz-range-thumb {
          width: 20px;
          height: 20px;
          border-radius: 50%;
          background: #9333ea;
          cursor: pointer;
          border: none;
        }
      `}</style>
    </div>
  )
}

export default PredictionProfiler
