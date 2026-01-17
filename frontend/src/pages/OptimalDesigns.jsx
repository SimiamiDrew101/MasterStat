import { useState } from 'react'
import axios from 'axios'
import { Sparkles, Plus, Trash2, Download, Info, BarChart3, Target } from 'lucide-react'
import * as XLSX from 'xlsx'

const OptimalDesigns = () => {
  // State management
  const [numFactors, setNumFactors] = useState(2)
  const [factorNames, setFactorNames] = useState(['X1', 'X2'])
  const [factorRanges, setFactorRanges] = useState({
    'X1': [-1, 1],
    'X2': [-1, 1]
  })
  const [nRuns, setNRuns] = useState(15)
  const [modelOrder, setModelOrder] = useState(2)
  const [criterion, setCriterion] = useState('d_optimal')
  const [maxIterations, setMaxIterations] = useState(1000)
  const [nCandidates, setNCandidates] = useState(20)

  const [design, setDesign] = useState(null)
  const [efficiency, setEfficiency] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('specify')

  // Update factor names and ranges when number changes
  const handleNumFactorsChange = (n) => {
    const newNum = parseInt(n)
    setNumFactors(newNum)

    const newNames = Array.from({ length: newNum }, (_, i) => `X${i + 1}`)
    setFactorNames(newNames)

    const newRanges = {}
    newNames.forEach(name => {
      newRanges[name] = factorRanges[name] || [-1, 1]
    })
    setFactorRanges(newRanges)
  }

  const handleFactorNameChange = (index, newName) => {
    const oldName = factorNames[index]
    const newNames = [...factorNames]
    newNames[index] = newName

    // Update ranges with new name
    const newRanges = {}
    Object.entries(factorRanges).forEach(([key, value]) => {
      if (key === oldName) {
        newRanges[newName] = value
      } else {
        newRanges[key] = value
      }
    })

    setFactorNames(newNames)
    setFactorRanges(newRanges)
  }

  const handleRangeChange = (factor, index, value) => {
    const newRanges = { ...factorRanges }
    newRanges[factor][index] = parseFloat(value)
    setFactorRanges(newRanges)
  }

  const handleGenerateDesign = async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await axios.post('/api/optimal-designs/generate', {
        n_runs: nRuns,
        factors: factorNames,
        factor_ranges: factorRanges,
        model_order: modelOrder,
        criterion: criterion,
        max_iterations: maxIterations,
        n_candidates: nCandidates
      })

      setDesign(response.data.design)
      setEfficiency(response.data.efficiency)
      setActiveTab('results')
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleExportCSV = () => {
    if (!design) return

    const csv = [
      ['Run', ...factorNames].join(','),
      ...design.map(row => [
        row.Run,
        ...factorNames.map(f => row[f])
      ].join(','))
    ].join('\n')

    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${criterion}_design_${nRuns}runs.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  const handleExportExcel = () => {
    if (!design) return

    const ws = XLSX.utils.json_to_sheet(design)
    const wb = XLSX.utils.book_new()
    XLSX.utils.book_append_sheet(wb, ws, 'Design')

    // Add efficiency sheet
    if (efficiency) {
      const effData = [
        { Metric: 'D-Efficiency', Value: efficiency.d_efficiency?.toFixed(4) || 'N/A' },
        { Metric: 'D-Criterion', Value: efficiency.d_criterion?.toExponential(4) || 'N/A' },
        { Metric: 'A-Criterion', Value: efficiency.a_criterion?.toFixed(4) || 'N/A' },
        { Metric: 'G-Efficiency', Value: efficiency.g_efficiency?.toFixed(4) || 'N/A' },
        { Metric: 'Condition Number', Value: efficiency.condition_number?.toFixed(2) || 'N/A' },
        { Metric: 'Max VIF', Value: efficiency.max_vif?.toFixed(2) || 'N/A' },
        { Metric: 'N Runs', Value: efficiency.n_runs },
        { Metric: 'N Parameters', Value: efficiency.n_parameters }
      ]
      const effWs = XLSX.utils.json_to_sheet(effData)
      XLSX.utils.book_append_sheet(wb, effWs, 'Efficiency')
    }

    XLSX.writeFile(wb, `${criterion}_design_${nRuns}runs.xlsx`)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-pink-600 text-transparent bg-clip-text">
            Optimal Designs
          </h1>
          <p className="text-gray-300">
            Generate D-optimal, I-optimal, and A-optimal experimental designs using the Coordinate Exchange algorithm
          </p>
        </div>

        {/* Tab Navigation */}
        <div className="flex space-x-2 bg-slate-800/50 p-2 rounded-lg mb-6">
          <button
            onClick={() => setActiveTab('specify')}
            className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors flex items-center justify-center space-x-2 ${
              activeTab === 'specify'
                ? 'bg-purple-600 text-white'
                : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
            }`}
          >
            <Target size={18} />
            <span>1. Specify Design</span>
          </button>
          <button
            onClick={() => setActiveTab('results')}
            disabled={!design}
            className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors flex items-center justify-center space-x-2 ${
              activeTab === 'results'
                ? 'bg-purple-600 text-white'
                : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700 disabled:opacity-50'
            }`}
          >
            <BarChart3 size={18} />
            <span>2. Results & Efficiency</span>
          </button>
          <button
            onClick={() => setActiveTab('info')}
            className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors flex items-center justify-center space-x-2 ${
              activeTab === 'info'
                ? 'bg-purple-600 text-white'
                : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
            }`}
          >
            <Info size={18} />
            <span>Information</span>
          </button>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-6 bg-red-500/10 border border-red-500/50 rounded-lg p-4">
            <p className="text-red-400">{error}</p>
          </div>
        )}

        {/* Tab: Specify Design */}
        {activeTab === 'specify' && (
          <div className="space-y-6">
            {/* Design Parameters */}
            <div className="bg-slate-800/50 rounded-2xl p-6">
              <h2 className="text-2xl font-bold mb-4">Design Parameters</h2>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Number of Runs */}
                <div>
                  <label className="block text-gray-300 mb-2">Number of Runs</label>
                  <input
                    type="number"
                    value={nRuns}
                    onChange={(e) => setNRuns(parseInt(e.target.value))}
                    min="3"
                    max="1000"
                    className="w-full bg-slate-700 text-gray-200 rounded px-4 py-2 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                  <p className="text-gray-400 text-xs mt-1">Recommended: 1.5-2× number of parameters</p>
                </div>

                {/* Model Order */}
                <div>
                  <label className="block text-gray-300 mb-2">Model Order</label>
                  <select
                    value={modelOrder}
                    onChange={(e) => setModelOrder(parseInt(e.target.value))}
                    className="w-full bg-slate-700 text-gray-200 rounded px-4 py-2 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  >
                    <option value="1">Linear (1st order)</option>
                    <option value="2">Quadratic (2nd order)</option>
                  </select>
                </div>

                {/* Optimality Criterion */}
                <div>
                  <label className="block text-gray-300 mb-2">Optimality Criterion</label>
                  <select
                    value={criterion}
                    onChange={(e) => setCriterion(e.target.value)}
                    className="w-full bg-slate-700 text-gray-200 rounded px-4 py-2 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  >
                    <option value="d_optimal">D-Optimal (Parameter Precision)</option>
                    <option value="i_optimal">I-Optimal (Prediction Accuracy)</option>
                    <option value="a_optimal">A-Optimal (Average Variance)</option>
                  </select>
                </div>
              </div>

              {/* Advanced Settings */}
              <details className="mt-6">
                <summary className="text-gray-300 cursor-pointer hover:text-purple-400 font-medium">
                  Advanced Settings
                </summary>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                  <div>
                    <label className="block text-gray-300 mb-2">Max Iterations</label>
                    <input
                      type="number"
                      value={maxIterations}
                      onChange={(e) => setMaxIterations(parseInt(e.target.value))}
                      min="10"
                      max="10000"
                      className="w-full bg-slate-700 text-gray-200 rounded px-4 py-2"
                    />
                  </div>
                  <div>
                    <label className="block text-gray-300 mb-2">Candidate Points</label>
                    <input
                      type="number"
                      value={nCandidates}
                      onChange={(e) => setNCandidates(parseInt(e.target.value))}
                      min="5"
                      max="100"
                      className="w-full bg-slate-700 text-gray-200 rounded px-4 py-2"
                    />
                  </div>
                </div>
              </details>
            </div>

            {/* Factor Specification */}
            <div className="bg-slate-800/50 rounded-2xl p-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-2xl font-bold">Factors</h2>
                <div className="flex items-center space-x-2">
                  <label className="text-gray-300">Number of Factors:</label>
                  <input
                    type="number"
                    value={numFactors}
                    onChange={(e) => handleNumFactorsChange(e.target.value)}
                    min="1"
                    max="10"
                    className="w-20 bg-slate-700 text-gray-200 rounded px-3 py-1"
                  />
                </div>
              </div>

              <div className="space-y-4">
                {factorNames.map((factor, index) => (
                  <div key={index} className="bg-slate-700/50 rounded-lg p-4">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-center">
                      {/* Factor Name */}
                      <div>
                        <label className="block text-gray-300 text-sm mb-2">Factor Name</label>
                        <input
                          type="text"
                          value={factor}
                          onChange={(e) => handleFactorNameChange(index, e.target.value)}
                          className="w-full bg-slate-600 text-gray-200 rounded px-3 py-2"
                        />
                      </div>

                      {/* Low Level */}
                      <div>
                        <label className="block text-gray-300 text-sm mb-2">Low Level</label>
                        <input
                          type="number"
                          value={factorRanges[factor]?.[0] || -1}
                          onChange={(e) => handleRangeChange(factor, 0, e.target.value)}
                          step="0.1"
                          className="w-full bg-slate-600 text-gray-200 rounded px-3 py-2"
                        />
                      </div>

                      {/* High Level */}
                      <div>
                        <label className="block text-gray-300 text-sm mb-2">High Level</label>
                        <input
                          type="number"
                          value={factorRanges[factor]?.[1] || 1}
                          onChange={(e) => handleRangeChange(factor, 1, e.target.value)}
                          step="0.1"
                          className="w-full bg-slate-600 text-gray-200 rounded px-3 py-2"
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Generate Button */}
            <button
              onClick={handleGenerateDesign}
              disabled={loading}
              className="w-full px-6 py-4 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 rounded-lg font-bold text-white text-lg transition-all duration-200 flex items-center justify-center space-x-3 disabled:opacity-50"
            >
              <Sparkles size={24} />
              <span>{loading ? 'Generating Design...' : 'Generate Optimal Design'}</span>
            </button>
          </div>
        )}

        {/* Tab: Results */}
        {activeTab === 'results' && design && (
          <div className="space-y-6">
            {/* Efficiency Metrics */}
            {efficiency && (
              <div className="bg-slate-800/50 rounded-2xl p-6">
                <h2 className="text-2xl font-bold mb-4">Design Efficiency</h2>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-gradient-to-br from-purple-600/20 to-pink-600/20 rounded-lg p-4 border border-purple-500/50">
                    <p className="text-gray-300 text-sm">D-Efficiency</p>
                    <p className="text-3xl font-bold text-purple-400">
                      {efficiency.d_efficiency?.toFixed(2) || 'N/A'}
                    </p>
                    <p className="text-gray-400 text-xs mt-1">Parameter precision</p>
                  </div>

                  <div className="bg-slate-700/50 rounded-lg p-4">
                    <p className="text-gray-300 text-sm">A-Criterion</p>
                    <p className="text-2xl font-bold text-gray-200">
                      {efficiency.a_criterion?.toFixed(2) || 'N/A'}
                    </p>
                    <p className="text-gray-400 text-xs mt-1">Avg variance</p>
                  </div>

                  <div className="bg-slate-700/50 rounded-lg p-4">
                    <p className="text-gray-300 text-sm">G-Efficiency</p>
                    <p className="text-2xl font-bold text-gray-200">
                      {efficiency.g_efficiency?.toFixed(2) || 'N/A'}
                    </p>
                    <p className="text-gray-400 text-xs mt-1">Max pred var</p>
                  </div>

                  <div className="bg-slate-700/50 rounded-lg p-4">
                    <p className="text-gray-300 text-sm">Condition Number</p>
                    <p className="text-2xl font-bold text-gray-200">
                      {efficiency.condition_number?.toExponential(1) || 'N/A'}
                    </p>
                    <p className="text-gray-400 text-xs mt-1">
                      {efficiency.condition_number < 10 ? '✓ Excellent' :
                       efficiency.condition_number < 100 ? '⚠ Good' : '✗ Poor'}
                    </p>
                  </div>
                </div>

                {/* VIF Values */}
                {efficiency.max_vif && (
                  <div className="mt-4 bg-slate-700/30 rounded-lg p-4">
                    <p className="text-gray-300 text-sm mb-2">
                      Max VIF: <strong>{efficiency.max_vif.toFixed(2)}</strong>
                      <span className="ml-2 text-xs">
                        ({efficiency.max_vif < 5 ? '✓ Excellent' :
                          efficiency.max_vif < 10 ? '⚠ Acceptable' : '✗ High Collinearity'})
                      </span>
                    </p>
                    <p className="text-gray-400 text-xs">
                      Variance Inflation Factor indicates multicollinearity. Values &lt; 10 are acceptable.
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Design Table */}
            <div className="bg-slate-800/50 rounded-2xl p-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-2xl font-bold">Design Matrix ({design.length} runs)</h2>
                <div className="flex space-x-2">
                  <button
                    onClick={handleExportCSV}
                    className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg font-medium transition-colors flex items-center space-x-2"
                  >
                    <Download size={18} />
                    <span>CSV</span>
                  </button>
                  <button
                    onClick={handleExportExcel}
                    className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-medium transition-colors flex items-center space-x-2"
                  >
                    <Download size={18} />
                    <span>Excel</span>
                  </button>
                </div>
              </div>

              <div className="overflow-x-auto bg-slate-700/30 rounded-lg">
                <table className="w-full">
                  <thead>
                    <tr className="bg-slate-700/70">
                      <th className="px-4 py-2 text-left text-gray-100 font-semibold border-b border-slate-600">
                        Run
                      </th>
                      {factorNames.map((factor, idx) => (
                        <th key={idx} className="px-4 py-2 text-right text-gray-100 font-semibold border-b border-slate-600">
                          {factor}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {design.map((row, idx) => (
                      <tr key={idx} className="border-b border-slate-700/30 hover:bg-slate-600/10">
                        <td className="px-4 py-2 text-gray-100">{row.Run}</td>
                        {factorNames.map((factor, fIdx) => (
                          <td key={fIdx} className="px-4 py-2 text-right text-gray-200 font-mono">
                            {row[factor]?.toFixed(4)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Tab: Information */}
        {activeTab === 'info' && (
          <div className="space-y-6">
            <div className="bg-slate-800/50 rounded-2xl p-6">
              <h2 className="text-2xl font-bold mb-4">Optimality Criteria</h2>

              <div className="space-y-4">
                <div className="bg-purple-900/20 rounded-lg p-4 border border-purple-700/30">
                  <h3 className="text-lg font-semibold text-purple-300 mb-2">D-Optimal Design</h3>
                  <p className="text-gray-300 mb-2">
                    Maximizes the determinant of the information matrix (X'X), which minimizes the volume of the confidence ellipsoid for parameter estimates.
                  </p>
                  <p className="text-gray-400 text-sm">
                    <strong>Best for:</strong> Maximum precision in parameter estimation<br />
                    <strong>Metric:</strong> D-efficiency = |X'X|^(1/p) / n
                  </p>
                </div>

                <div className="bg-blue-900/20 rounded-lg p-4 border border-blue-700/30">
                  <h3 className="text-lg font-semibold text-blue-300 mb-2">I-Optimal Design</h3>
                  <p className="text-gray-300 mb-2">
                    Minimizes the average prediction variance across the entire design space, making it ideal when prediction accuracy is the primary goal.
                  </p>
                  <p className="text-gray-400 text-sm">
                    <strong>Best for:</strong> Accurate predictions across the design region<br />
                    <strong>Metric:</strong> Average of x'(X'X)^-1 x over prediction points
                  </p>
                </div>

                <div className="bg-green-900/20 rounded-lg p-4 border border-green-700/30">
                  <h3 className="text-lg font-semibold text-green-300 mb-2">A-Optimal Design</h3>
                  <p className="text-gray-300 mb-2">
                    Minimizes the trace of the inverse information matrix, which minimizes the average variance of parameter estimates.
                  </p>
                  <p className="text-gray-400 text-sm">
                    <strong>Best for:</strong> Overall parameter estimation accuracy<br />
                    <strong>Metric:</strong> A-criterion = trace((X'X)^-1)
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-slate-800/50 rounded-2xl p-6">
              <h2 className="text-2xl font-bold mb-4">Efficiency Metrics Guide</h2>

              <div className="space-y-3">
                <div>
                  <h4 className="text-gray-200 font-semibold">D-Efficiency (0-100)</h4>
                  <p className="text-gray-400 text-sm">
                    Relative efficiency compared to orthogonal design. Higher is better. Values &gt; 80% are excellent.
                  </p>
                </div>

                <div>
                  <h4 className="text-gray-200 font-semibold">Condition Number</h4>
                  <p className="text-gray-400 text-sm">
                    Ratio of largest to smallest eigenvalue. Lower is better. &lt; 10 is excellent, &lt; 100 is good, &gt; 1000 indicates numerical issues.
                  </p>
                </div>

                <div>
                  <h4 className="text-gray-200 font-semibold">VIF (Variance Inflation Factor)</h4>
                  <p className="text-gray-400 text-sm">
                    Measures multicollinearity for each parameter. &lt; 5 is excellent, &lt; 10 is acceptable, &gt; 10 indicates high collinearity.
                  </p>
                </div>

                <div>
                  <h4 className="text-gray-200 font-semibold">G-Efficiency</h4>
                  <p className="text-gray-400 text-sm">
                    Reciprocal of maximum prediction variance. Higher is better. Indicates worst-case prediction precision.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default OptimalDesigns
