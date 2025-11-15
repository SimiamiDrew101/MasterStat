import { useState } from 'react'
import axios from 'axios'
import ResultCard from '../components/ResultCard'
import DistributionPlot from '../components/DistributionPlot'
import AssumptionsPanel from '../components/AssumptionsPanel'
import EffectSizeCard from '../components/EffectSizeCard'
import { Calculator, AlertCircle } from 'lucide-react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const HypothesisTesting = () => {
  const [activeTab, setActiveTab] = useState('t-test') // 't-test', 'f-test', 'z-test'
  const [resultsTab, setResultsTab] = useState('results') // 'results', 'diagnostics', 'visualization'
  const [alpha, setAlpha] = useState(0.05)
  const [alternative, setAlternative] = useState('two-sided')
  const [paired, setPaired] = useState(false)
  const [mu, setMu] = useState(0)
  const [sigma, setSigma] = useState(1)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Table data for samples
  const [sample1TableData, setSample1TableData] = useState(Array(15).fill(''))
  const [sample2TableData, setSample2TableData] = useState(Array(15).fill(''))

  // Handle cell changes
  const handleSample1CellChange = (index, value) => {
    const newData = [...sample1TableData]
    newData[index] = value

    // Auto-expand: add more rows if we're near the end
    if (index >= newData.length - 3 && value.trim() !== '') {
      for (let i = 0; i < 5; i++) {
        newData.push('')
      }
    }

    setSample1TableData(newData)
  }

  const handleSample2CellChange = (index, value) => {
    const newData = [...sample2TableData]
    newData[index] = value

    // Auto-expand: add more rows if we're near the end
    if (index >= newData.length - 3 && value.trim() !== '') {
      for (let i = 0; i < 5; i++) {
        newData.push('')
      }
    }

    setSample2TableData(newData)
  }

  // Keyboard navigation
  const handleKeyDown = (e, index, isSample1 = true) => {
    const tableData = isSample1 ? sample1TableData : sample2TableData
    const prefix = isSample1 ? 'sample1' : 'sample2'
    let newIndex = index

    switch (e.key) {
      case 'ArrowUp':
        e.preventDefault()
        newIndex = Math.max(0, index - 1)
        break
      case 'ArrowDown':
      case 'Enter':
        e.preventDefault()
        newIndex = Math.min(tableData.length - 1, index + 1)
        break
      default:
        return
    }

    const cellId = `${prefix}-cell-${newIndex}`
    const cell = document.getElementById(cellId)
    if (cell) {
      cell.focus()
      cell.select()
    }
  }

  // Generate random normal data
  const generateRandomNormal = (mean = 0, stdDev = 1) => {
    const u1 = Math.random()
    const u2 = Math.random()
    const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
    return mean + z0 * stdDev
  }

  const populateSample1Data = () => {
    const n = 20
    const mean = 50
    const stdDev = 5
    const newData = []

    for (let i = 0; i < n; i++) {
      newData.push((mean + generateRandomNormal(0, stdDev)).toFixed(2))
    }

    // Fill remaining with empty strings
    while (newData.length < 15) {
      newData.push('')
    }

    setSample1TableData(newData)
  }

  const populateSample2Data = () => {
    const n = 20
    const mean = 55
    const stdDev = 5
    const newData = []

    for (let i = 0; i < n; i++) {
      newData.push((mean + generateRandomNormal(0, stdDev)).toFixed(2))
    }

    // Fill remaining with empty strings
    while (newData.length < 15) {
      newData.push('')
    }

    setSample2TableData(newData)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      // Extract data from tables
      const sample1Data = sample1TableData
        .filter(val => val && String(val).trim() !== '' && !isNaN(parseFloat(val)))
        .map(val => parseFloat(val))

      const sample2Data = sample2TableData
        .filter(val => val && String(val).trim() !== '' && !isNaN(parseFloat(val)))
        .map(val => parseFloat(val))

      // Validation
      if (sample1Data.length === 0) {
        throw new Error('Sample 1 must contain at least one valid number')
      }

      let endpoint = ''
      let payload = {}

      if (activeTab === 't-test') {
        endpoint = '/api/hypothesis/t-test'
        payload = {
          sample1: sample1Data,
          sample2: sample2Data.length > 0 ? sample2Data : null,
          alternative,
          alpha,
          paired,
          mu
        }
      } else if (activeTab === 'f-test') {
        endpoint = '/api/hypothesis/f-test'
        if (sample2Data.length === 0) {
          throw new Error('F-test requires two samples')
        }
        payload = {
          sample1: sample1Data,
          sample2: sample2Data,
          alpha
        }
      } else if (activeTab === 'z-test') {
        endpoint = '/api/hypothesis/z-test'
        payload = {
          sample: sample1Data,
          mu0: mu,
          sigma,
          alternative,
          alpha
        }
      }

      const response = await axios.post(`${API_URL}${endpoint}`, payload)
      setResult(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <div className="flex items-center space-x-3">
          <Calculator className="w-8 h-8 text-blue-400" />
          <h2 className="text-3xl font-bold text-gray-100">Hypothesis Testing</h2>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl border border-slate-700/50 overflow-hidden">
        <div className="grid grid-cols-1 md:grid-cols-3">
          {/* t-Test Tab */}
          <button
            onClick={() => {
              setActiveTab('t-test')
              setResult(null)
              setError(null)
            }}
            className={`px-6 py-4 font-semibold text-center transition-all border-b-4 ${
              activeTab === 't-test'
                ? 'bg-blue-500/20 text-blue-400 border-blue-500'
                : 'bg-slate-700/30 text-gray-400 border-transparent hover:bg-slate-700/50 hover:text-gray-300'
            }`}
          >
            <div className="text-lg">t-Test</div>
            <div className="text-xs mt-1 opacity-75">Compare means</div>
          </button>

          {/* F-Test Tab */}
          <button
            onClick={() => {
              setActiveTab('f-test')
              setResult(null)
              setError(null)
            }}
            className={`px-6 py-4 font-semibold text-center transition-all border-b-4 ${
              activeTab === 'f-test'
                ? 'bg-orange-500/20 text-orange-400 border-orange-500'
                : 'bg-slate-700/30 text-gray-400 border-transparent hover:bg-slate-700/50 hover:text-gray-300'
            }`}
          >
            <div className="text-lg">F-Test</div>
            <div className="text-xs mt-1 opacity-75">Variance equality</div>
          </button>

          {/* Z-Test Tab */}
          <button
            onClick={() => {
              setActiveTab('z-test')
              setResult(null)
              setError(null)
            }}
            className={`px-6 py-4 font-semibold text-center transition-all border-b-4 ${
              activeTab === 'z-test'
                ? 'bg-green-500/20 text-green-400 border-green-500'
                : 'bg-slate-700/30 text-gray-400 border-transparent hover:bg-slate-700/50 hover:text-gray-300'
            }`}
          >
            <div className="text-lg">Z-Test</div>
            <div className="text-xs mt-1 opacity-75">Known population variance</div>
          </button>
        </div>
      </div>

      {/* Content Form */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <form onSubmit={handleSubmit} className="space-y-6">

          {/* Sample 1 */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="block text-gray-200 font-medium">
                Sample 1
              </label>
              <button
                type="button"
                onClick={populateSample1Data}
                className="px-3 py-1 text-sm bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-md hover:from-purple-700 hover:to-blue-700 transition-all"
              >
                Generate Sample Data
              </button>
            </div>
            <div className="bg-slate-700/30 rounded-lg p-2 border border-slate-600 max-h-64 overflow-y-auto">
              <div className="grid grid-cols-1 gap-0.5">
                {sample1TableData.map((value, index) => (
                  <input
                    key={index}
                    id={`sample1-cell-${index}`}
                    type="text"
                    value={value}
                    onChange={(e) => handleSample1CellChange(index, e.target.value)}
                    onKeyDown={(e) => handleKeyDown(e, index, true)}
                    placeholder={index === 0 ? 'Enter value' : ''}
                    className="px-2 py-1 bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 text-sm"
                  />
                ))}
              </div>
            </div>
            <p className="text-xs text-gray-400 mt-1">
              Use ↑↓ arrow keys or Enter to navigate between cells
            </p>
          </div>

          {/* Sample 2 */}
          {(activeTab === 't-test' || activeTab === 'f-test') && activeTab !== 'z-test' && (
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="block text-gray-200 font-medium">
                  Sample 2 {activeTab === 'f-test' ? '(required)' : '(optional for one-sample t-test)'}
                </label>
                <button
                  type="button"
                  onClick={populateSample2Data}
                  className="px-3 py-1 text-sm bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-md hover:from-purple-700 hover:to-blue-700 transition-all"
                >
                  Generate Sample Data
                </button>
              </div>
              <div className="bg-slate-700/30 rounded-lg p-2 border border-slate-600 max-h-64 overflow-y-auto">
                <div className="grid grid-cols-1 gap-0.5">
                  {sample2TableData.map((value, index) => (
                    <input
                      key={index}
                      id={`sample2-cell-${index}`}
                      type="text"
                      value={value}
                      onChange={(e) => handleSample2CellChange(index, e.target.value)}
                      onKeyDown={(e) => handleKeyDown(e, index, false)}
                      placeholder={index === 0 ? 'Enter value' : ''}
                      className="px-2 py-1 bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 text-sm"
                    />
                  ))}
                </div>
              </div>
              <p className="text-xs text-gray-400 mt-1">
                Use ↑↓ arrow keys or Enter to navigate between cells
              </p>
            </div>
          )}

          {/* z-test specific options */}
          {activeTab === 'z-test' && (
            <>
              <div>
                <label className="block text-gray-200 font-medium mb-2">
                  Hypothesized Mean (μ₀)
                </label>
                <input
                  type="number"
                  step="any"
                  value={mu}
                  onChange={(e) => setMu(parseFloat(e.target.value))}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div>
                <label className="block text-gray-200 font-medium mb-2">
                  Known Population Standard Deviation (σ)
                </label>
                <input
                  type="number"
                  step="any"
                  min="0.0001"
                  value={sigma}
                  onChange={(e) => setSigma(parseFloat(e.target.value))}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div>
                <label className="block text-gray-200 font-medium mb-2">
                  Alternative Hypothesis
                </label>
                <select
                  value={alternative}
                  onChange={(e) => setAlternative(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="two-sided">Two-sided (≠)</option>
                  <option value="greater">Greater (&gt;)</option>
                  <option value="less">Less (&lt;)</option>
                </select>
              </div>
            </>
          )}

          {/* t-test specific options */}
          {activeTab === 't-test' && (
            <>
              <div>
                <label className="block text-gray-200 font-medium mb-2">
                  Hypothesized Mean (μ₀) {sample2TableData.some(v => v && String(v).trim() !== '') && '(only for one-sample t-test)'}
                </label>
                <input
                  type="number"
                  step="any"
                  value={mu}
                  onChange={(e) => setMu(parseFloat(e.target.value))}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              {sample2TableData.some(v => v && String(v).trim() !== '') && (
                <div>
                  <label className="flex items-center space-x-2 text-gray-200">
                    <input
                      type="checkbox"
                      checked={paired}
                      onChange={(e) => setPaired(e.target.checked)}
                      className="w-5 h-5 rounded"
                    />
                    <span>Paired samples</span>
                  </label>
                </div>
              )}

              <div>
                <label className="block text-gray-200 font-medium mb-2">
                  Alternative Hypothesis
                </label>
                <select
                  value={alternative}
                  onChange={(e) => setAlternative(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="two-sided">Two-sided (≠)</option>
                  <option value="greater">Greater (&gt;)</option>
                  <option value="less">Less (&lt;)</option>
                </select>
              </div>
            </>
          )}

          {/* Alpha */}
          <div>
            <label className="block text-gray-200 font-medium mb-2">
              Significance Level (α)
            </label>
            <input
              type="number"
              step="0.01"
              min="0"
              max="1"
              value={alpha}
              onChange={(e) => setAlpha(parseFloat(e.target.value))}
              className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Calculating...' : 'Run Analysis'}
          </button>
        </form>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-900/30 backdrop-blur-lg rounded-xl p-4 border border-red-700/50">
          <p className="text-red-200 font-medium">Error: {error}</p>
        </div>
      )}

      {/* Assumptions Warning (if violated) */}
      {result && result.assumptions && !result.assumptions.check?.all_assumptions_met && (
        <div className="bg-orange-900/20 backdrop-blur-lg rounded-xl p-5 border border-orange-700/50">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-6 h-6 text-orange-400 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-orange-200 mb-2">Assumptions Violated</h3>
              <p className="text-orange-100 text-sm mb-3">
                Some statistical assumptions for this test may be violated. Review the Diagnostics tab for details.
              </p>
              {result.assumptions.check.recommendations && result.assumptions.check.recommendations.length > 0 && (
                <div className="bg-orange-950/50 rounded-lg p-3 mt-3">
                  <p className="text-orange-200 font-semibold text-sm mb-2">Recommendations:</p>
                  <ul className="list-disc list-inside space-y-1">
                    {result.assumptions.check.recommendations.map((rec, idx) => (
                      <li key={idx} className="text-orange-100 text-sm">{rec}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Results Display with Tabs */}
      {result && (
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
          {/* Tab Navigation */}
          <div className="flex border-b border-slate-600 mb-6">
            <button
              onClick={() => setResultsTab('results')}
              className={`px-6 py-3 font-semibold transition-colors ${
                resultsTab === 'results'
                  ? 'text-blue-400 border-b-2 border-blue-400'
                  : 'text-gray-400 hover:text-gray-300'
              }`}
            >
              Results & Effect Size
            </button>
            {result.assumptions && (
              <button
                onClick={() => setResultsTab('diagnostics')}
                className={`px-6 py-3 font-semibold transition-colors ${
                  resultsTab === 'diagnostics'
                    ? 'text-purple-400 border-b-2 border-purple-400'
                    : 'text-gray-400 hover:text-gray-300'
                }`}
              >
                Diagnostics
                {!result.assumptions.check?.all_assumptions_met && (
                  <span className="ml-2 px-2 py-0.5 text-xs bg-orange-600 text-white rounded-full">!</span>
                )}
              </button>
            )}
            {result.distribution_plot_data && (
              <button
                onClick={() => setResultsTab('visualization')}
                className={`px-6 py-3 font-semibold transition-colors ${
                  resultsTab === 'visualization'
                    ? 'text-cyan-400 border-b-2 border-cyan-400'
                    : 'text-gray-400 hover:text-gray-300'
                }`}
              >
                Visualization
              </button>
            )}
          </div>

          {/* Tab Content */}
          <div className="space-y-6">
            {resultsTab === 'results' && (
              <>
                <ResultCard result={result} />
                {result.effect_size && (
                  <EffectSizeCard
                    effectSize={result.effect_size}
                    powerAnalysis={result.power_analysis}
                  />
                )}
              </>
            )}

            {resultsTab === 'diagnostics' && result.assumptions && (
              <AssumptionsPanel
                assumptions={result.assumptions}
                alpha={alpha}
              />
            )}

            {resultsTab === 'visualization' && result.distribution_plot_data && (
              <DistributionPlot
                distributionData={result.distribution_plot_data}
                pValue={result.p_value}
                testType={result.test_type}
                alpha={alpha}
              />
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default HypothesisTesting
