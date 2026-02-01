import { useState, useCallback, useEffect } from 'react'
import axios from 'axios'
import Plot from 'react-plotly.js'
import { Settings, Plus, Trash2, Play, Download, AlertTriangle, CheckCircle, Layers, Target, Zap, Info } from 'lucide-react'

const API_URL = import.meta.env.VITE_API_URL || ''

const CustomDesign = () => {
  // State
  const [activeTab, setActiveTab] = useState('factors')

  // Factor definitions
  const [factors, setFactors] = useState([
    { name: 'X1', type: 'continuous', low: -1, high: 1, levels: [], hardToChange: false },
    { name: 'X2', type: 'continuous', low: -1, high: 1, levels: [], hardToChange: false }
  ])

  // Design settings
  const [nRuns, setNRuns] = useState(12)
  const [criterion, setCriterion] = useState('d_optimal')
  const [modelType, setModelType] = useState('quadratic')
  const [nRandomStarts, setNRandomStarts] = useState(10)
  const [maxIterations, setMaxIterations] = useState(500)

  // Constraints
  const [constraints, setConstraints] = useState([])
  const [disallowed, setDisallowed] = useState([])

  // Results
  const [design, setDesign] = useState(null)
  const [evaluation, setEvaluation] = useState(null)
  const [powerResult, setPowerResult] = useState(null)

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  // Add factor
  const addFactor = () => {
    const newName = `X${factors.length + 1}`
    setFactors([...factors, {
      name: newName,
      type: 'continuous',
      low: -1,
      high: 1,
      levels: [],
      hardToChange: false
    }])
  }

  // Remove factor
  const removeFactor = (index) => {
    if (factors.length > 1) {
      setFactors(factors.filter((_, i) => i !== index))
    }
  }

  // Update factor
  const updateFactor = (index, field, value) => {
    const updated = [...factors]
    updated[index] = { ...updated[index], [field]: value }

    // If switching to categorical, initialize levels
    if (field === 'type' && value === 'categorical' && updated[index].levels.length === 0) {
      updated[index].levels = ['Low', 'High']
    }

    setFactors(updated)
  }

  // Add level to categorical factor
  const addLevel = (factorIndex) => {
    const updated = [...factors]
    updated[factorIndex].levels.push(`Level${updated[factorIndex].levels.length + 1}`)
    setFactors(updated)
  }

  // Remove level from categorical factor
  const removeLevel = (factorIndex, levelIndex) => {
    const updated = [...factors]
    if (updated[factorIndex].levels.length > 2) {
      updated[factorIndex].levels = updated[factorIndex].levels.filter((_, i) => i !== levelIndex)
      setFactors(updated)
    }
  }

  // Update level
  const updateLevel = (factorIndex, levelIndex, value) => {
    const updated = [...factors]
    updated[factorIndex].levels[levelIndex] = value
    setFactors(updated)
  }

  // Add constraint
  const addConstraint = () => {
    const coefficients = {}
    factors.forEach(f => {
      if (f.type === 'continuous') {
        coefficients[f.name] = 0
      }
    })
    setConstraints([...constraints, {
      coefficients,
      bound: 1,
      type: '<='
    }])
  }

  // Remove constraint
  const removeConstraint = (index) => {
    setConstraints(constraints.filter((_, i) => i !== index))
  }

  // Update constraint
  const updateConstraint = (index, field, value) => {
    const updated = [...constraints]
    if (field.startsWith('coef_')) {
      const factorName = field.replace('coef_', '')
      updated[index].coefficients[factorName] = parseFloat(value) || 0
    } else {
      updated[index][field] = field === 'bound' ? parseFloat(value) || 0 : value
    }
    setConstraints(updated)
  }

  // Calculate minimum runs needed
  const getMinRuns = () => {
    let nTerms = 1 // Intercept
    nTerms += factors.length // Linear terms

    if (modelType === 'interaction' || modelType === 'quadratic') {
      nTerms += (factors.length * (factors.length - 1)) / 2 // Interactions
    }

    if (modelType === 'quadratic') {
      nTerms += factors.filter(f => f.type === 'continuous').length // Quadratic
    }

    return nTerms + 1 // Need at least p+1 runs
  }

  // Generate design
  const handleGenerateDesign = async () => {
    if (nRuns < getMinRuns()) {
      setError(`Need at least ${getMinRuns()} runs for this model. You have ${nRuns}.`)
      return
    }

    setLoading(true)
    setError('')

    try {
      const factorDefs = factors.map(f => ({
        name: f.name,
        type: f.type,
        low: f.type === 'continuous' ? f.low : null,
        high: f.type === 'continuous' ? f.high : null,
        levels: f.type === 'categorical' ? f.levels : null,
        hard_to_change: f.hardToChange
      }))

      const constraintDefs = constraints.length > 0 ? constraints.map(c => ({
        coefficients: c.coefficients,
        bound: c.bound,
        type: c.type
      })) : null

      const response = await axios.post(`${API_URL}/api/custom-design/generate`, {
        n_runs: nRuns,
        factors: factorDefs,
        criterion: criterion,
        model_type: modelType,
        constraints: constraintDefs,
        disallowed: disallowed.length > 0 ? disallowed : null,
        n_random_starts: nRandomStarts,
        max_iterations: maxIterations
      })

      setDesign(response.data.design)
      setEvaluation(response.data.evaluation)
      setActiveTab('results')
    } catch (err) {
      setError('Design generation failed: ' + (err.response?.data?.detail || err.message))
    } finally {
      setLoading(false)
    }
  }

  // Run power analysis
  const handlePowerAnalysis = async () => {
    if (!design) {
      setError('Generate a design first')
      return
    }

    setLoading(true)
    setError('')

    try {
      const factorDefs = factors.map(f => ({
        name: f.name,
        type: f.type,
        low: f.type === 'continuous' ? f.low : null,
        high: f.type === 'continuous' ? f.high : null,
        levels: f.type === 'categorical' ? f.levels : null
      }))

      const response = await axios.post(`${API_URL}/api/custom-design/power-analysis`, {
        design: design,
        factors: factorDefs,
        model_type: modelType,
        effect_size: 1.0,
        alpha: 0.05,
        sigma: 1.0
      })

      setPowerResult(response.data)
    } catch (err) {
      setError('Power analysis failed: ' + (err.response?.data?.detail || err.message))
    } finally {
      setLoading(false)
    }
  }

  // Export design as CSV
  const exportDesign = () => {
    if (!design) return

    const headers = ['Run', ...factors.map(f => f.name)]
    const rows = design.map(row => [
      row.Run,
      ...factors.map(f => row[f.name])
    ])

    const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'custom_design.csv'
    a.click()
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-pink-600 text-transparent bg-clip-text">
            Custom Design Platform
          </h1>
          <p className="text-gray-300 text-lg">
            Create optimal experimental designs with constraints, categorical factors, and custom models
          </p>
        </div>

        {/* Tab Navigation */}
        <div className="flex space-x-2 bg-slate-800/50 p-2 rounded-lg mb-6">
          <button
            onClick={() => setActiveTab('factors')}
            className={`flex-1 px-4 py-3 rounded-lg font-medium transition-all duration-200 flex items-center justify-center space-x-2 ${
              activeTab === 'factors'
                ? 'bg-purple-600 text-white shadow-lg'
                : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
            }`}
          >
            <Layers size={20} />
            <span>1. Factors</span>
          </button>
          <button
            onClick={() => setActiveTab('constraints')}
            className={`flex-1 px-4 py-3 rounded-lg font-medium transition-all duration-200 flex items-center justify-center space-x-2 ${
              activeTab === 'constraints'
                ? 'bg-purple-600 text-white shadow-lg'
                : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
            }`}
          >
            <Target size={20} />
            <span>2. Constraints</span>
          </button>
          <button
            onClick={() => setActiveTab('settings')}
            className={`flex-1 px-4 py-3 rounded-lg font-medium transition-all duration-200 flex items-center justify-center space-x-2 ${
              activeTab === 'settings'
                ? 'bg-purple-600 text-white shadow-lg'
                : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
            }`}
          >
            <Settings size={20} />
            <span>3. Settings</span>
          </button>
          <button
            onClick={() => setActiveTab('results')}
            disabled={!design}
            className={`flex-1 px-4 py-3 rounded-lg font-medium transition-all duration-200 flex items-center justify-center space-x-2 ${
              activeTab === 'results'
                ? 'bg-purple-600 text-white shadow-lg'
                : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700 disabled:opacity-40'
            }`}
          >
            <Zap size={20} />
            <span>4. Results</span>
          </button>
          <button
            onClick={() => setActiveTab('info')}
            className={`flex-1 px-4 py-3 rounded-lg font-medium transition-all duration-200 flex items-center justify-center space-x-2 ${
              activeTab === 'info'
                ? 'bg-purple-600 text-white shadow-lg'
                : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
            }`}
          >
            <Info size={20} />
            <span>Info</span>
          </button>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-6 bg-red-900/50 border border-red-600 text-red-200 px-4 py-3 rounded-lg flex items-center space-x-2">
            <AlertTriangle size={20} />
            <span>{error}</span>
          </div>
        )}

        {/* Factors Tab */}
        {activeTab === 'factors' && (
          <div className="space-y-6">
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-2xl font-bold text-gray-100">Define Factors</h2>
                <button
                  onClick={addFactor}
                  className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
                >
                  <Plus size={18} />
                  Add Factor
                </button>
              </div>

              <div className="space-y-4">
                {factors.map((factor, index) => (
                  <div key={index} className="bg-slate-700/30 rounded-lg p-4 border border-slate-600/50">
                    <div className="grid grid-cols-1 md:grid-cols-6 gap-4 items-end">
                      {/* Name */}
                      <div>
                        <label className="block text-xs text-gray-400 mb-1">Name</label>
                        <input
                          type="text"
                          value={factor.name}
                          onChange={(e) => updateFactor(index, 'name', e.target.value)}
                          className="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded text-white text-sm focus:outline-none focus:ring-1 focus:ring-purple-500"
                        />
                      </div>

                      {/* Type */}
                      <div>
                        <label className="block text-xs text-gray-400 mb-1">Type</label>
                        <select
                          value={factor.type}
                          onChange={(e) => updateFactor(index, 'type', e.target.value)}
                          className="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded text-white text-sm focus:outline-none focus:ring-1 focus:ring-purple-500"
                        >
                          <option value="continuous">Continuous</option>
                          <option value="categorical">Categorical</option>
                        </select>
                      </div>

                      {/* Low/High for continuous */}
                      {factor.type === 'continuous' && (
                        <>
                          <div>
                            <label className="block text-xs text-gray-400 mb-1">Low</label>
                            <input
                              type="number"
                              step="any"
                              value={factor.low}
                              onChange={(e) => updateFactor(index, 'low', parseFloat(e.target.value))}
                              className="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded text-white text-sm focus:outline-none focus:ring-1 focus:ring-purple-500"
                            />
                          </div>
                          <div>
                            <label className="block text-xs text-gray-400 mb-1">High</label>
                            <input
                              type="number"
                              step="any"
                              value={factor.high}
                              onChange={(e) => updateFactor(index, 'high', parseFloat(e.target.value))}
                              className="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded text-white text-sm focus:outline-none focus:ring-1 focus:ring-purple-500"
                            />
                          </div>
                        </>
                      )}

                      {/* Hard to change checkbox */}
                      <div className="flex items-center">
                        <label className="flex items-center gap-2 text-sm text-gray-300">
                          <input
                            type="checkbox"
                            checked={factor.hardToChange}
                            onChange={(e) => updateFactor(index, 'hardToChange', e.target.checked)}
                            className="w-4 h-4 text-purple-600 bg-slate-700 border-slate-600 rounded"
                          />
                          Hard to change
                        </label>
                      </div>

                      {/* Delete button */}
                      <div>
                        <button
                          onClick={() => removeFactor(index)}
                          disabled={factors.length <= 1}
                          className="p-2 text-red-400 hover:text-red-300 hover:bg-red-900/30 rounded transition-colors disabled:opacity-40"
                        >
                          <Trash2 size={18} />
                        </button>
                      </div>
                    </div>

                    {/* Categorical levels */}
                    {factor.type === 'categorical' && (
                      <div className="mt-4 pt-4 border-t border-slate-600/50">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-sm text-gray-400">Levels:</span>
                          <button
                            onClick={() => addLevel(index)}
                            className="text-xs px-2 py-1 bg-slate-600 hover:bg-slate-500 text-white rounded"
                          >
                            + Add Level
                          </button>
                        </div>
                        <div className="flex flex-wrap gap-2">
                          {factor.levels.map((level, levelIndex) => (
                            <div key={levelIndex} className="flex items-center gap-1 bg-slate-800 rounded px-2 py-1">
                              <input
                                type="text"
                                value={level}
                                onChange={(e) => updateLevel(index, levelIndex, e.target.value)}
                                className="w-20 bg-transparent text-white text-sm focus:outline-none"
                              />
                              {factor.levels.length > 2 && (
                                <button
                                  onClick={() => removeLevel(index, levelIndex)}
                                  className="text-red-400 hover:text-red-300"
                                >
                                  <Trash2 size={14} />
                                </button>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>

              <div className="mt-4 text-sm text-gray-400">
                {factors.length} factors defined. Minimum {getMinRuns()} runs needed for {modelType} model.
              </div>
            </div>
          </div>
        )}

        {/* Constraints Tab */}
        {activeTab === 'constraints' && (
          <div className="space-y-6">
            {/* Linear Constraints */}
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-2xl font-bold text-gray-100">Linear Constraints</h2>
                <button
                  onClick={addConstraint}
                  className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
                >
                  <Plus size={18} />
                  Add Constraint
                </button>
              </div>

              {constraints.length === 0 ? (
                <p className="text-gray-400">No constraints defined. Click "Add Constraint" to add one.</p>
              ) : (
                <div className="space-y-3">
                  {constraints.map((constraint, cIndex) => (
                    <div key={cIndex} className="bg-slate-700/30 rounded-lg p-4 flex items-center gap-3 flex-wrap">
                      {factors.filter(f => f.type === 'continuous').map((factor, fIndex) => (
                        <div key={fIndex} className="flex items-center gap-1">
                          <input
                            type="number"
                            step="any"
                            value={constraint.coefficients[factor.name] || 0}
                            onChange={(e) => updateConstraint(cIndex, `coef_${factor.name}`, e.target.value)}
                            className="w-16 px-2 py-1 bg-slate-800 border border-slate-600 rounded text-white text-sm"
                          />
                          <span className="text-gray-300 text-sm">{factor.name}</span>
                          {fIndex < factors.filter(f => f.type === 'continuous').length - 1 && (
                            <span className="text-gray-500 mx-1">+</span>
                          )}
                        </div>
                      ))}

                      <select
                        value={constraint.type}
                        onChange={(e) => updateConstraint(cIndex, 'type', e.target.value)}
                        className="px-2 py-1 bg-slate-800 border border-slate-600 rounded text-white text-sm"
                      >
                        <option value="<=">≤</option>
                        <option value=">=">≥</option>
                        <option value="==">=</option>
                      </select>

                      <input
                        type="number"
                        step="any"
                        value={constraint.bound}
                        onChange={(e) => updateConstraint(cIndex, 'bound', e.target.value)}
                        className="w-20 px-2 py-1 bg-slate-800 border border-slate-600 rounded text-white text-sm"
                      />

                      <button
                        onClick={() => removeConstraint(cIndex)}
                        className="p-1 text-red-400 hover:text-red-300"
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>
                  ))}
                </div>
              )}

              <p className="mt-4 text-sm text-gray-400">
                Constraints restrict the design space. For example: X1 + X2 ≤ 1 ensures the sum of two factors doesn't exceed 1.
              </p>
            </div>
          </div>
        )}

        {/* Settings Tab */}
        {activeTab === 'settings' && (
          <div className="space-y-6">
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <h2 className="text-2xl font-bold mb-6 text-gray-100">Design Settings</h2>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {/* Number of runs */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Number of Runs
                  </label>
                  <input
                    type="number"
                    min={getMinRuns()}
                    value={nRuns}
                    onChange={(e) => setNRuns(parseInt(e.target.value) || getMinRuns())}
                    className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                  <p className="text-xs text-gray-400 mt-1">Minimum: {getMinRuns()}</p>
                </div>

                {/* Optimality criterion */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Optimality Criterion
                  </label>
                  <select
                    value={criterion}
                    onChange={(e) => setCriterion(e.target.value)}
                    className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                  >
                    <option value="d_optimal">D-Optimal (Parameter Estimation)</option>
                    <option value="i_optimal">I-Optimal (Prediction)</option>
                    <option value="a_optimal">A-Optimal (Avg. Variance)</option>
                  </select>
                </div>

                {/* Model type */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Model Type
                  </label>
                  <select
                    value={modelType}
                    onChange={(e) => setModelType(e.target.value)}
                    className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                  >
                    <option value="linear">Linear (Main Effects Only)</option>
                    <option value="interaction">Two-Factor Interaction</option>
                    <option value="quadratic">Quadratic (Full RSM)</option>
                  </select>
                </div>

                {/* Random starts */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Random Starts
                  </label>
                  <input
                    type="number"
                    min={1}
                    max={100}
                    value={nRandomStarts}
                    onChange={(e) => setNRandomStarts(parseInt(e.target.value) || 10)}
                    className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                  <p className="text-xs text-gray-400 mt-1">More starts = better design, slower</p>
                </div>

                {/* Max iterations */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Max Iterations
                  </label>
                  <input
                    type="number"
                    min={10}
                    max={5000}
                    value={maxIterations}
                    onChange={(e) => setMaxIterations(parseInt(e.target.value) || 500)}
                    className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>
              </div>
            </div>

            {/* Generate Button */}
            <div className="flex justify-center gap-4">
              <button
                onClick={handleGenerateDesign}
                disabled={loading}
                className="px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white rounded-xl font-bold text-lg shadow-lg hover:shadow-purple-500/50 transition-all duration-200 disabled:opacity-50 flex items-center gap-2"
              >
                {loading ? (
                  <>Generating...</>
                ) : (
                  <>
                    <Play size={20} />
                    Generate Design
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {/* Results Tab */}
        {activeTab === 'results' && design && (
          <div className="space-y-6">
            {/* Efficiency Metrics */}
            {evaluation && (
              <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
                <h2 className="text-2xl font-bold mb-4 text-gray-100">Design Efficiency</h2>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-slate-700/30 rounded-lg p-4">
                    <p className="text-gray-400 text-sm">D-Efficiency</p>
                    <p className="text-2xl font-bold text-green-400">
                      {(evaluation.d_efficiency * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="bg-slate-700/30 rounded-lg p-4">
                    <p className="text-gray-400 text-sm">G-Efficiency</p>
                    <p className="text-2xl font-bold text-blue-400">
                      {(evaluation.g_efficiency * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="bg-slate-700/30 rounded-lg p-4">
                    <p className="text-gray-400 text-sm">Condition Number</p>
                    <p className={`text-2xl font-bold ${evaluation.condition_number < 10 ? 'text-green-400' : evaluation.condition_number < 100 ? 'text-yellow-400' : 'text-red-400'}`}>
                      {evaluation.condition_number?.toFixed(1)}
                    </p>
                  </div>
                  <div className="bg-slate-700/30 rounded-lg p-4">
                    <p className="text-gray-400 text-sm">Max VIF</p>
                    <p className={`text-2xl font-bold ${evaluation.max_vif < 5 ? 'text-green-400' : evaluation.max_vif < 10 ? 'text-yellow-400' : 'text-red-400'}`}>
                      {evaluation.max_vif?.toFixed(2)}
                    </p>
                  </div>
                </div>

                {evaluation.estimable ? (
                  <div className="mt-4 flex items-center gap-2 text-green-400">
                    <CheckCircle size={18} />
                    <span>Design is estimable ({evaluation.n_parameters} parameters, {evaluation.df_residual} df residual)</span>
                  </div>
                ) : (
                  <div className="mt-4 flex items-center gap-2 text-red-400">
                    <AlertTriangle size={18} />
                    <span>Design is NOT estimable - add more runs</span>
                  </div>
                )}
              </div>
            )}

            {/* Design Table */}
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-2xl font-bold text-gray-100">Design Matrix ({design.length} runs)</h2>
                <div className="flex gap-2">
                  <button
                    onClick={handlePowerAnalysis}
                    disabled={loading}
                    className="flex items-center gap-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition-colors"
                  >
                    <Zap size={18} />
                    Power Analysis
                  </button>
                  <button
                    onClick={exportDesign}
                    className="flex items-center gap-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition-colors"
                  >
                    <Download size={18} />
                    Export CSV
                  </button>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-600">
                      <th className="px-4 py-2 text-left text-gray-400">Run</th>
                      {factors.map(f => (
                        <th key={f.name} className="px-4 py-2 text-right text-gray-400">{f.name}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {design.map((row, i) => (
                      <tr key={i} className="border-b border-slate-700/50 hover:bg-slate-700/30">
                        <td className="px-4 py-2 text-white font-medium">{row.Run}</td>
                        {factors.map(f => (
                          <td key={f.name} className="px-4 py-2 text-right text-gray-300">
                            {typeof row[f.name] === 'number' ? row[f.name].toFixed(4) : row[f.name]}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Power Analysis Results */}
            {powerResult && (
              <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
                <h2 className="text-2xl font-bold mb-4 text-gray-100">Power Analysis</h2>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                  <div className="bg-slate-700/30 rounded-lg p-3">
                    <p className="text-gray-400 text-xs">Average Power</p>
                    <p className={`text-xl font-bold ${powerResult.summary.average_power >= 0.8 ? 'text-green-400' : 'text-yellow-400'}`}>
                      {(powerResult.summary.average_power * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="bg-slate-700/30 rounded-lg p-3">
                    <p className="text-gray-400 text-xs">Minimum Power</p>
                    <p className={`text-xl font-bold ${powerResult.summary.minimum_power >= 0.8 ? 'text-green-400' : 'text-yellow-400'}`}>
                      {(powerResult.summary.minimum_power * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="bg-slate-700/30 rounded-lg p-3">
                    <p className="text-gray-400 text-xs">Effect Size</p>
                    <p className="text-xl font-bold text-white">{powerResult.summary.effect_size}</p>
                  </div>
                  <div className="bg-slate-700/30 rounded-lg p-3">
                    <p className="text-gray-400 text-xs">Alpha</p>
                    <p className="text-xl font-bold text-white">{powerResult.summary.alpha}</p>
                  </div>
                </div>

                <p className="text-gray-300">{powerResult.interpretation}</p>

                <div className="mt-4 overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-slate-600">
                        <th className="px-4 py-2 text-left text-gray-400">Term</th>
                        <th className="px-4 py-2 text-right text-gray-400">Std Error</th>
                        <th className="px-4 py-2 text-right text-gray-400">Power</th>
                        <th className="px-4 py-2 text-right text-gray-400">Min Detectable</th>
                      </tr>
                    </thead>
                    <tbody>
                      {powerResult.power_results.map((r, i) => (
                        <tr key={i} className="border-b border-slate-700/50">
                          <td className="px-4 py-2 text-white">{r.term}</td>
                          <td className="px-4 py-2 text-right text-gray-300">{r.se.toFixed(4)}</td>
                          <td className="px-4 py-2 text-right">
                            <span className={r.power >= 0.8 ? 'text-green-400' : 'text-yellow-400'}>
                              {(r.power * 100).toFixed(1)}%
                            </span>
                          </td>
                          <td className="px-4 py-2 text-right text-gray-300">{r.detectable_effect.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* VIF Plot */}
            {evaluation && evaluation.vif_values && (
              <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
                <h2 className="text-xl font-bold mb-4 text-gray-100">Variance Inflation Factors</h2>
                <Plot
                  data={[{
                    x: evaluation.term_names || evaluation.vif_values.map((_, i) => `Term ${i}`),
                    y: evaluation.vif_values,
                    type: 'bar',
                    marker: {
                      color: evaluation.vif_values.map(v =>
                        v < 5 ? '#22c55e' : v < 10 ? '#eab308' : '#ef4444'
                      )
                    }
                  }]}
                  layout={{
                    xaxis: { title: 'Model Term', gridcolor: '#334155', tickangle: -45 },
                    yaxis: { title: 'VIF', gridcolor: '#334155' },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(15,23,42,0.5)',
                    font: { color: '#e2e8f0' },
                    autosize: true,
                    shapes: [{
                      type: 'line',
                      x0: -0.5,
                      x1: evaluation.vif_values.length - 0.5,
                      y0: 5,
                      y1: 5,
                      line: { color: '#eab308', dash: 'dash', width: 2 }
                    }],
                    margin: { b: 100 }
                  }}
                  style={{ width: '100%', height: '400px' }}
                  useResizeHandler={true}
                />
                <p className="text-sm text-gray-400 mt-2">
                  VIF &lt; 5: Excellent (green) | VIF 5-10: Acceptable (yellow) | VIF &gt; 10: Problematic (red)
                </p>
              </div>
            )}
          </div>
        )}

        {/* Info Tab */}
        {activeTab === 'info' && (
          <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
            <h2 className="text-2xl font-bold mb-4 text-gray-100">About Custom Design</h2>
            <div className="space-y-4 text-gray-300">
              <p>
                The Custom Design Platform uses coordinate exchange algorithm to generate
                optimal experimental designs tailored to your specific needs.
              </p>

              <h3 className="text-xl font-bold text-white mt-6">Optimality Criteria</h3>
              <ul className="list-disc list-inside space-y-2 ml-4">
                <li><strong className="text-purple-400">D-Optimal:</strong> Maximizes |X'X|. Best for precise parameter estimation. Minimizes the volume of the confidence ellipsoid.</li>
                <li><strong className="text-purple-400">I-Optimal:</strong> Minimizes average prediction variance. Best when you want accurate predictions across the design space.</li>
                <li><strong className="text-purple-400">A-Optimal:</strong> Minimizes trace((X'X)⁻¹). Minimizes average variance of parameter estimates.</li>
              </ul>

              <h3 className="text-xl font-bold text-white mt-6">Model Types</h3>
              <ul className="list-disc list-inside space-y-2 ml-4">
                <li><strong>Linear:</strong> Main effects only: y = β₀ + Σβᵢxᵢ</li>
                <li><strong>Interaction:</strong> Main effects + 2-way interactions</li>
                <li><strong>Quadratic:</strong> Full RSM model with squared terms</li>
              </ul>

              <h3 className="text-xl font-bold text-white mt-6">Factor Types</h3>
              <ul className="list-disc list-inside space-y-2 ml-4">
                <li><strong>Continuous:</strong> Numeric factors with a range [low, high]</li>
                <li><strong>Categorical:</strong> Discrete factors with named levels</li>
                <li><strong>Hard-to-Change:</strong> Factors that are difficult to reset between runs (for split-plot designs)</li>
              </ul>

              <h3 className="text-xl font-bold text-white mt-6">Constraints</h3>
              <p>Linear constraints of the form: a₁X₁ + a₂X₂ + ... ≤ b</p>
              <p className="text-sm">Example: If X1 + X2 must not exceed 1, add constraint: 1×X1 + 1×X2 ≤ 1</p>

              <h3 className="text-xl font-bold text-white mt-6">Efficiency Metrics</h3>
              <ul className="list-disc list-inside space-y-2 ml-4">
                <li><strong>D-Efficiency:</strong> 100% = orthogonal design. Higher is better.</li>
                <li><strong>G-Efficiency:</strong> Based on maximum prediction variance. Higher is better.</li>
                <li><strong>Condition Number:</strong> &lt;10 is excellent, &lt;100 is acceptable.</li>
                <li><strong>VIF (Variance Inflation Factor):</strong> &lt;5 is good, &lt;10 is acceptable.</li>
              </ul>

              <h3 className="text-xl font-bold text-white mt-6">Tips</h3>
              <ul className="list-disc list-inside space-y-2 ml-4">
                <li>More runs generally give better designs and more power</li>
                <li>More random starts help find better designs (at cost of time)</li>
                <li>If VIFs are high, consider adding more runs or simplifying the model</li>
                <li>Use I-optimal if prediction accuracy is more important than parameter estimation</li>
              </ul>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default CustomDesign
