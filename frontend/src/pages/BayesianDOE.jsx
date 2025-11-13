import { useState } from 'react'
import axios from 'axios'
import { Network, TrendingUp, Plus, Trash2, Settings, BarChart2, Layers, Lightbulb } from 'lucide-react'
import Plot from 'react-plotly.js'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const BayesianDOE = () => {
  // Bayesian Design of Experiments Component
  const [activeTab, setActiveTab] = useState('analysis')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Data state
  const [numFactors, setNumFactors] = useState(2)
  const [factorNames, setFactorNames] = useState(['X1', 'X2'])
  const [responseName, setResponseName] = useState('Y')
  const [tableData, setTableData] = useState([])

  // Prior specifications
  const [priors, setPriors] = useState({})

  // Results state
  const [analysisResult, setAnalysisResult] = useState(null)
  const [sequentialResult, setSequentialResult] = useState(null)
  const [comparisonResult, setComparisonResult] = useState(null)

  // MCMC settings
  const [nSamples, setNSamples] = useState(2000)
  const [nBurn, setNBurn] = useState(500)

  // Initialize factors when number changes
  const handleFactorCountChange = (count) => {
    const newCount = parseInt(count) || 2
    setNumFactors(newCount)
    const newFactorNames = Array.from({ length: newCount }, (_, i) => `X${i + 1}`)
    setFactorNames(newFactorNames)

    // Initialize priors for all parameters
    const newPriors = { Intercept: { dist_type: 'normal', params: { loc: 0, scale: 10 } } }
    newFactorNames.forEach(factor => {
      newPriors[factor] = { dist_type: 'normal', params: { loc: 0, scale: 5 } }
    })
    setPriors(newPriors)

    // Initialize empty table
    setTableData(Array(8).fill(null).map(() => Array(newCount + 1).fill('')))
  }

  // Generate factorial design
  const generateFactorialDesign = () => {
    const design = []
    const levels = [-1, 1]

    // Full 2^k factorial
    const n = Math.pow(2, numFactors)
    for (let i = 0; i < n; i++) {
      const row = []
      for (let j = 0; j < numFactors; j++) {
        row.push(levels[Math.floor(i / Math.pow(2, j)) % 2])
      }
      row.push('') // Empty response
      design.push(row)
    }

    setTableData(design)
  }

  // Fill test data
  const fillTestData = () => {
    const newData = tableData.map(row => {
      if (!row[0] && row[0] !== 0) return row
      const newRow = [...row]
      // Generate response with some effect and noise
      let response = 10
      row.slice(0, -1).forEach((factor, i) => {
        response += (i + 1) * factor + 0.5 * factor * factor
      })
      response += (Math.random() - 0.5) * 2
      newRow[newRow.length - 1] = response.toFixed(2)
      return newRow
    })
    setTableData(newData)
  }

  // Handle prior changes
  const updatePrior = (param, field, value) => {
    setPriors(prev => ({
      ...prev,
      [param]: {
        ...prev[param],
        [field]: field === 'params' ? value : value
      }
    }))
  }

  // Run Bayesian analysis
  const runBayesianAnalysis = async () => {
    setLoading(true)
    setError(null)

    try {
      const validRows = tableData.filter(row => {
        const responseValue = row[row.length - 1]
        return responseValue !== '' && responseValue !== null && responseValue !== undefined
      })

      if (validRows.length < 5) {
        throw new Error('Need at least 5 complete data points')
      }

      const data = validRows.map(row => {
        const point = {}
        factorNames.forEach((factor, i) => {
          point[factor] = parseFloat(row[i])
        })
        point[responseName] = parseFloat(row[row.length - 1])
        return point
      })

      const response = await axios.post(`${API_URL}/api/bayesian-doe/factorial-analysis`, {
        data,
        factors: factorNames,
        response: responseName,
        priors,
        n_samples: nSamples,
        n_burn: nBurn
      })

      setAnalysisResult(response.data)
      setActiveTab('results')
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Analysis failed')
    } finally {
      setLoading(false)
    }
  }

  // Table editing
  const handleCellChange = (rowIndex, colIndex, value) => {
    const newData = [...tableData]
    newData[rowIndex][colIndex] = value
    setTableData(newData)
  }

  const addRow = () => {
    setTableData([...tableData, Array(numFactors + 1).fill('')])
  }

  const removeRow = (rowIndex) => {
    if (tableData.length > 1) {
      setTableData(tableData.filter((_, idx) => idx !== rowIndex))
    }
  }

  // Common layout for Plotly
  const plotLayout = {
    paper_bgcolor: '#334155',
    plot_bgcolor: '#1e293b',
    font: { color: '#e2e8f0' },
    xaxis: { gridcolor: '#475569', zerolinecolor: '#64748b' },
    yaxis: { gridcolor: '#475569', zerolinecolor: '#64748b' },
    margin: { l: 60, r: 40, b: 60, t: 40 }
  }

  const plotConfig = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-900/30 to-indigo-900/30 backdrop-blur-lg rounded-2xl p-8 border border-purple-700/50">
        <div className="flex items-center space-x-3 mb-4">
          <Network className="w-10 h-10 text-purple-400" />
          <h2 className="text-4xl font-bold text-gray-100">Bayesian Design of Experiments</h2>
        </div>
        <p className="text-gray-300 text-lg">
          Apply Bayesian methods to design and analyze experiments with prior knowledge, uncertainty quantification, and adaptive strategies.
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="flex space-x-2 bg-slate-800/50 p-2 rounded-lg">
        <button
          onClick={() => setActiveTab('analysis')}
          className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
            activeTab === 'analysis'
              ? 'bg-purple-600 text-white'
              : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
          }`}
        >
          1. Factorial Analysis
        </button>
        <button
          onClick={() => setActiveTab('results')}
          className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
            activeTab === 'results'
              ? 'bg-purple-600 text-white'
              : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
          }`}
          disabled={!analysisResult}
        >
          2. Results & Inference
        </button>
        <button
          onClick={() => setActiveTab('sequential')}
          className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
            activeTab === 'sequential'
              ? 'bg-purple-600 text-white'
              : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
          }`}
          disabled={!analysisResult}
        >
          3. Sequential Design
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-900/30 backdrop-blur-lg rounded-xl p-4 border border-red-700/50">
          <p className="text-red-200 font-medium">Error: {error}</p>
        </div>
      )}

      {/* Tab Content */}
      {activeTab === 'analysis' && (
        <div className="space-y-6">
          {/* Design Configuration */}
          <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
            <div className="flex items-center gap-2 mb-6">
              <Settings className="w-6 h-6 text-purple-400" />
              <h3 className="text-2xl font-bold text-gray-100">Design Configuration</h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div>
                <label className="block text-gray-100 font-medium mb-2">Number of Factors</label>
                <input
                  type="number"
                  min={2}
                  max={5}
                  value={numFactors}
                  onChange={(e) => handleFactorCountChange(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-purple-500"
                />
              </div>

              <div>
                <label className="block text-gray-100 font-medium mb-2">Response Variable</label>
                <input
                  type="text"
                  value={responseName}
                  onChange={(e) => setResponseName(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-purple-500"
                />
              </div>

              <div className="flex items-end">
                <button
                  onClick={generateFactorialDesign}
                  className="w-full bg-purple-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-purple-700 transition-colors"
                >
                  Generate 2^k Design
                </button>
              </div>
            </div>
          </div>

          {/* Prior Specifications */}
          <div className="bg-gradient-to-r from-indigo-900/30 to-blue-900/30 backdrop-blur-lg rounded-2xl p-6 border border-indigo-700/50">
            <div className="flex items-center gap-2 mb-6">
              <Lightbulb className="w-6 h-6 text-indigo-400" />
              <h3 className="text-2xl font-bold text-gray-100">Prior Distributions</h3>
            </div>

            <div className="space-y-4">
              {['Intercept', ...factorNames].map((param) => (
                <div key={param} className="bg-slate-700/30 rounded-lg p-4">
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
                    <div>
                      <label className="block text-gray-300 text-sm mb-2 font-semibold">{param}</label>
                      <select
                        value={priors[param]?.dist_type || 'normal'}
                        onChange={(e) => {
                          const newParams = e.target.value === 'normal' ? { loc: 0, scale: 5 } :
                                          e.target.value === 'uniform' ? { low: -10, high: 10 } :
                                          e.target.value === 'cauchy' ? { loc: 0, scale: 2 } :
                                          e.target.value === 't' ? { df: 3, loc: 0, scale: 5 } :
                                          { scale: 1 }
                          updatePrior(param, 'dist_type', e.target.value)
                          updatePrior(param, 'params', newParams)
                        }}
                        className="w-full px-3 py-2 rounded bg-slate-700 text-gray-100 border border-slate-600 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                      >
                        <option value="normal">Normal</option>
                        <option value="uniform">Uniform</option>
                        <option value="cauchy">Cauchy</option>
                        <option value="t">Student-t</option>
                      </select>
                    </div>

                    {priors[param]?.dist_type === 'normal' && (
                      <>
                        <div>
                          <label className="block text-gray-400 text-xs mb-1">Mean (μ)</label>
                          <input
                            type="number"
                            step="0.1"
                            value={priors[param]?.params?.loc || 0}
                            onChange={(e) => updatePrior(param, 'params', { ...priors[param].params, loc: parseFloat(e.target.value) })}
                            className="w-full px-3 py-2 rounded bg-slate-700 text-gray-100 border border-slate-600 text-sm"
                          />
                        </div>
                        <div>
                          <label className="block text-gray-400 text-xs mb-1">Std Dev (σ)</label>
                          <input
                            type="number"
                            step="0.1"
                            min="0.1"
                            value={priors[param]?.params?.scale || 5}
                            onChange={(e) => updatePrior(param, 'params', { ...priors[param].params, scale: parseFloat(e.target.value) })}
                            className="w-full px-3 py-2 rounded bg-slate-700 text-gray-100 border border-slate-600 text-sm"
                          />
                        </div>
                      </>
                    )}

                    {priors[param]?.dist_type === 'uniform' && (
                      <>
                        <div>
                          <label className="block text-gray-400 text-xs mb-1">Lower Bound</label>
                          <input
                            type="number"
                            step="0.1"
                            value={priors[param]?.params?.low || -10}
                            onChange={(e) => updatePrior(param, 'params', { ...priors[param].params, low: parseFloat(e.target.value) })}
                            className="w-full px-3 py-2 rounded bg-slate-700 text-gray-100 border border-slate-600 text-sm"
                          />
                        </div>
                        <div>
                          <label className="block text-gray-400 text-xs mb-1">Upper Bound</label>
                          <input
                            type="number"
                            step="0.1"
                            value={priors[param]?.params?.high || 10}
                            onChange={(e) => updatePrior(param, 'params', { ...priors[param].params, high: parseFloat(e.target.value) })}
                            className="w-full px-3 py-2 rounded bg-slate-700 text-gray-100 border border-slate-600 text-sm"
                          />
                        </div>
                      </>
                    )}

                    <div className="text-xs text-gray-400 flex items-center">
                      {priors[param]?.dist_type === 'normal' && `N(${priors[param]?.params?.loc || 0}, ${priors[param]?.params?.scale || 5}²)`}
                      {priors[param]?.dist_type === 'uniform' && `U(${priors[param]?.params?.low || -10}, ${priors[param]?.params?.high || 10})`}
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-4 bg-blue-900/20 rounded-lg p-4 border border-blue-700/30">
              <p className="text-blue-200 text-sm">
                <strong>Prior Selection:</strong> Choose distributions that reflect your prior beliefs about each parameter.
                Normal priors are common for unbounded parameters. Use uniform priors for minimal prior information.
              </p>
            </div>
          </div>

          {/* MCMC Settings */}
          <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
            <h3 className="text-xl font-bold text-gray-100 mb-4">MCMC Sampler Settings</h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-gray-100 font-medium mb-2">Number of Samples</label>
                <input
                  type="number"
                  min={500}
                  max={10000}
                  step={500}
                  value={nSamples}
                  onChange={(e) => setNSamples(parseInt(e.target.value))}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600"
                />
                <p className="text-xs text-gray-400 mt-1">Posterior samples after burn-in</p>
              </div>

              <div>
                <label className="block text-gray-100 font-medium mb-2">Burn-in Samples</label>
                <input
                  type="number"
                  min={100}
                  max={2000}
                  step={100}
                  value={nBurn}
                  onChange={(e) => setNBurn(parseInt(e.target.value))}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600"
                />
                <p className="text-xs text-gray-400 mt-1">Samples to discard for convergence</p>
              </div>
            </div>
          </div>

          {/* Data Entry */}
          {tableData.length > 0 && (
            <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-bold text-gray-100">Experimental Data</h3>
                <div className="flex gap-2">
                  <button
                    onClick={fillTestData}
                    className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm"
                  >
                    Fill Test Data
                  </button>
                  <button
                    onClick={addRow}
                    className="flex items-center gap-1 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors text-sm"
                  >
                    <Plus className="w-4 h-4" />
                    Add Row
                  </button>
                </div>
              </div>

              <div className="overflow-x-auto bg-slate-700/30 rounded-lg border border-slate-600">
                <table className="w-full border-collapse">
                  <thead>
                    <tr className="bg-slate-700/70">
                      <th className="px-3 py-2 text-center text-gray-100 font-semibold border-b border-slate-600">#</th>
                      {factorNames.map((factor, idx) => (
                        <th key={idx} className="px-3 py-2 text-center text-gray-100 font-semibold border-b border-slate-600">
                          {factor}
                        </th>
                      ))}
                      <th className="px-3 py-2 text-center text-gray-100 font-semibold border-b border-slate-600 bg-purple-900/20">
                        {responseName}
                      </th>
                      <th className="px-3 py-2 text-center text-gray-100 font-semibold border-b border-slate-600"></th>
                    </tr>
                  </thead>
                  <tbody>
                    {tableData.map((row, rowIndex) => (
                      <tr key={rowIndex} className="border-b border-slate-700/30 hover:bg-slate-600/10">
                        <td className="px-3 py-2 text-center text-gray-300 bg-slate-700/30">{rowIndex + 1}</td>
                        {row.map((cell, colIndex) => (
                          <td key={colIndex} className="px-1 py-1">
                            <input
                              type="text"
                              value={cell}
                              onChange={(e) => handleCellChange(rowIndex, colIndex, e.target.value)}
                              className="w-full px-2 py-1.5 bg-slate-800/50 text-gray-100 border border-slate-600/50 focus:border-purple-500 rounded-sm focus:outline-none text-sm"
                            />
                          </td>
                        ))}
                        <td className="px-2 py-2 text-center">
                          <button
                            onClick={() => removeRow(rowIndex)}
                            disabled={tableData.length === 1}
                            className="p-1 text-red-400 hover:text-red-300 disabled:opacity-30 transition-colors"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <button
                onClick={runBayesianAnalysis}
                disabled={loading}
                className="w-full mt-4 bg-gradient-to-r from-purple-600 to-indigo-600 text-white font-bold py-3 px-6 rounded-lg hover:from-purple-700 hover:to-indigo-700 transition-all disabled:opacity-50"
              >
                {loading ? 'Running MCMC Sampler...' : 'Run Bayesian Analysis'}
              </button>
            </div>
          )}
        </div>
      )}

      {activeTab === 'results' && analysisResult && (
        <div className="space-y-6">
          {/* MCMC Diagnostics */}
          <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
            <h3 className="text-2xl font-bold text-gray-100 mb-4">MCMC Convergence</h3>

            <div className="grid grid-cols-3 gap-4">
              <div className="bg-green-900/20 rounded-lg p-4 border border-green-700/30">
                <p className="text-green-200 text-sm">Acceptance Rate</p>
                <p className="text-3xl font-bold text-green-100">{(analysisResult.acceptance_rate * 100).toFixed(1)}%</p>
                <p className="text-xs text-green-300 mt-1">Target: 20-50%</p>
              </div>
              <div className="bg-blue-900/20 rounded-lg p-4 border border-blue-700/30">
                <p className="text-blue-200 text-sm">Samples</p>
                <p className="text-3xl font-bold text-blue-100">{analysisResult.n_samples}</p>
                <p className="text-xs text-blue-300 mt-1">After burn-in</p>
              </div>
              <div className="bg-purple-900/20 rounded-lg p-4 border border-purple-700/30">
                <p className="text-purple-200 text-sm">Burn-in</p>
                <p className="text-3xl font-bold text-purple-100">{analysisResult.n_burn}</p>
                <p className="text-xs text-purple-300 mt-1">Discarded samples</p>
              </div>
            </div>
          </div>

          {/* Posterior Summary */}
          <div className="bg-gradient-to-r from-purple-900/30 to-pink-900/30 backdrop-blur-lg rounded-2xl p-6 border border-purple-700/50">
            <h3 className="text-2xl font-bold text-gray-100 mb-4">Posterior Distributions</h3>

            <div className="overflow-x-auto bg-slate-700/30 rounded-lg">
              <table className="w-full">
                <thead>
                  <tr className="bg-slate-700/70">
                    <th className="px-4 py-3 text-left text-gray-100 font-semibold">Parameter</th>
                    <th className="px-4 py-3 text-right text-gray-100 font-semibold">Mean</th>
                    <th className="px-4 py-3 text-right text-gray-100 font-semibold">Std Dev</th>
                    <th className="px-4 py-3 text-right text-gray-100 font-semibold">Median</th>
                    <th className="px-4 py-3 text-right text-gray-100 font-semibold">95% CI Lower</th>
                    <th className="px-4 py-3 text-right text-gray-100 font-semibold">95% CI Upper</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(analysisResult.posterior_summary).map(([param, stats]) => (
                    <tr key={param} className="border-b border-slate-700/30 hover:bg-slate-600/10">
                      <td className="px-4 py-3 text-gray-100 font-mono text-sm font-semibold">{param}</td>
                      <td className="px-4 py-3 text-right text-gray-100 font-mono">{stats.mean.toFixed(4)}</td>
                      <td className="px-4 py-3 text-right text-gray-300 font-mono">{stats.std.toFixed(4)}</td>
                      <td className="px-4 py-3 text-right text-gray-100 font-mono">{stats.median.toFixed(4)}</td>
                      <td className="px-4 py-3 text-right text-purple-300 font-mono">{stats.lower_95.toFixed(4)}</td>
                      <td className="px-4 py-3 text-right text-purple-300 font-mono">{stats.upper_95.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Bayes Factors */}
          <div className="bg-gradient-to-r from-orange-900/30 to-red-900/30 backdrop-blur-lg rounded-2xl p-6 border border-orange-700/50">
            <div className="flex items-center gap-2 mb-4">
              <BarChart2 className="w-6 h-6 text-orange-400" />
              <h3 className="text-2xl font-bold text-gray-100">Bayes Factors (Effect Significance)</h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.entries(analysisResult.bayes_factors).map(([param, bf]) => (
                <div key={param} className="bg-slate-700/30 rounded-lg p-4">
                  <div className="flex justify-between items-start mb-2">
                    <span className="text-gray-100 font-semibold font-mono">{param}</span>
                    <span className={`px-2 py-1 rounded text-xs font-bold ${
                      bf.bayes_factor > 10 ? 'bg-green-900/50 text-green-200' :
                      bf.bayes_factor > 3 ? 'bg-yellow-900/50 text-yellow-200' :
                      'bg-gray-700 text-gray-400'
                    }`}>
                      {bf.interpretation}
                    </span>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Bayes Factor:</span>
                      <span className="text-gray-100 font-mono">{bf.bayes_factor.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">P(Effect ≠ 0):</span>
                      <span className="text-gray-100 font-mono">{(bf.prob_significant * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-4 bg-orange-900/20 rounded-lg p-4 border border-orange-700/30">
              <p className="text-orange-200 text-sm">
                <strong>Interpretation:</strong> BF &gt; 10: Strong evidence, BF &gt; 3: Moderate evidence, BF &lt; 3: Weak evidence for the effect being non-zero.
              </p>
            </div>
          </div>

          {/* Posterior Predictive Check */}
          {analysisResult.posterior_predictive && (
            <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
              <h3 className="text-2xl font-bold text-gray-100 mb-4">Posterior Predictive Check</h3>

              <div className="bg-slate-700/30 rounded-lg p-4">
                <Plot
                  data={[
                    {
                      type: 'scatter',
                      mode: 'markers',
                      x: analysisResult.posterior_predictive.observed,
                      y: analysisResult.posterior_predictive.predicted_mean,
                      error_y: {
                        type: 'data',
                        symmetric: false,
                        array: analysisResult.posterior_predictive.predicted_upper_95.map((u, i) => u - analysisResult.posterior_predictive.predicted_mean[i]),
                        arrayminus: analysisResult.posterior_predictive.predicted_mean.map((m, i) => m - analysisResult.posterior_predictive.predicted_lower_95[i])
                      },
                      marker: { size: 10, color: '#a855f7', line: { color: '#1e293b', width: 1 } },
                      name: 'Predictions'
                    },
                    {
                      type: 'scatter',
                      mode: 'lines',
                      x: [Math.min(...analysisResult.posterior_predictive.observed), Math.max(...analysisResult.posterior_predictive.observed)],
                      y: [Math.min(...analysisResult.posterior_predictive.observed), Math.max(...analysisResult.posterior_predictive.observed)],
                      line: { color: '#ef4444', dash: 'dash', width: 2 },
                      name: 'Perfect Fit',
                      showlegend: false
                    }
                  ]}
                  layout={{
                    ...plotLayout,
                    xaxis: { ...plotLayout.xaxis, title: 'Observed' },
                    yaxis: { ...plotLayout.yaxis, title: 'Predicted (with 95% CI)' },
                    title: 'Observed vs Predicted',
                    height: 400
                  }}
                  config={plotConfig}
                  style={{ width: '100%' }}
                />
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === 'sequential' && analysisResult && (
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
          <div className="flex items-center gap-2 mb-6">
            <TrendingUp className="w-6 h-6 text-green-400" />
            <h3 className="text-2xl font-bold text-gray-100">Sequential Design & Adaptive Strategies</h3>
          </div>

          <div className="bg-gradient-to-r from-green-900/30 to-teal-900/30 rounded-lg p-6 border border-green-700/30">
            <h4 className="text-xl font-bold text-green-100 mb-4">Coming in Next Update</h4>
            <ul className="space-y-3 text-green-200">
              <li className="flex items-start gap-2">
                <span className="text-green-400 font-bold">•</span>
                <span>Generate candidate points and select the most informative next experiments</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-400 font-bold">•</span>
                <span>Expected information gain calculation for each candidate</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-400 font-bold">•</span>
                <span>Uncertainty reduction visualization</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-400 font-bold">•</span>
                <span>Adaptive stopping criteria based on posterior uncertainty</span>
              </li>
            </ul>

            <div className="mt-6 p-4 bg-blue-900/20 rounded-lg border border-blue-700/30">
              <p className="text-blue-200 text-sm">
                <strong>Current Status:</strong> You've completed the Bayesian factorial analysis.
                The sequential design module will allow you to intelligently select the next experiments
                to run based on maximizing expected information gain or reducing uncertainty.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Help/Guide Section */}
      <div className="bg-gradient-to-r from-blue-900/20 to-cyan-900/20 rounded-lg p-5 border border-blue-700/30">
        <h4 className="text-blue-200 font-semibold mb-3 flex items-center gap-2">
          <Layers className="w-5 h-5" />
          Bayesian DOE Features
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-blue-100">
          <div>
            <p className="font-semibold mb-2">Implemented:</p>
            <ul className="space-y-1">
              <li>✓ Bayesian factorial design analysis</li>
              <li>✓ Prior distribution specification</li>
              <li>✓ MCMC parameter estimation</li>
              <li>✓ Posterior distributions & credible intervals</li>
              <li>✓ Bayes factors for effect significance</li>
            </ul>
          </div>
          <div>
            <p className="font-semibold mb-2">Backend Ready:</p>
            <ul className="space-y-1">
              <li>✓ Sequential experimental design</li>
              <li>✓ Expected information gain</li>
              <li>✓ Model comparison & selection</li>
              <li>✓ Adaptive DOE strategies</li>
              <li>✓ Optimal design under uncertainty</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

export default BayesianDOE
