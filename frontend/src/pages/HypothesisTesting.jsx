import { useState } from 'react'
import axios from 'axios'
import ResultCard from '../components/ResultCard'
import { Calculator } from 'lucide-react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const HypothesisTesting = () => {
  const [testType, setTestType] = useState('t-test')
  const [sample1, setSample1] = useState('')
  const [sample2, setSample2] = useState('')
  const [alpha, setAlpha] = useState(0.05)
  const [alternative, setAlternative] = useState('two-sided')
  const [paired, setPaired] = useState(false)
  const [mu, setMu] = useState(0)
  const [sigma, setSigma] = useState(1)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const parseSample = (text) => {
    return text.split(/[,\s]+/).filter(x => x).map(x => parseFloat(x))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const sample1Data = parseSample(sample1)
      const sample2Data = sample2 ? parseSample(sample2) : null

      let endpoint = ''
      let payload = {}

      if (testType === 't-test') {
        endpoint = '/api/hypothesis/t-test'
        payload = {
          sample1: sample1Data,
          sample2: sample2Data,
          alternative,
          alpha,
          paired,
          mu
        }
      } else if (testType === 'f-test') {
        endpoint = '/api/hypothesis/f-test'
        if (!sample2Data) {
          throw new Error('F-test requires two samples')
        }
        payload = {
          sample1: sample1Data,
          sample2: sample2Data,
          alpha
        }
      } else if (testType === 'z-test') {
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
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <div className="flex items-center space-x-3 mb-6">
          <Calculator className="w-8 h-8 text-blue-400" />
          <h2 className="text-3xl font-bold text-gray-100">Hypothesis Testing</h2>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Test Type Selection */}
          <div>
            <label className="block text-gray-200 font-medium mb-2">Test Type</label>
            <select
              value={testType}
              onChange={(e) => setTestType(e.target.value)}
              className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="t-test">t-Test</option>
              <option value="f-test">F-Test (Variance Equality)</option>
              <option value="z-test">Z-Test (Known Population Variance)</option>
            </select>
          </div>

          {/* Sample 1 */}
          <div>
            <label className="block text-gray-200 font-medium mb-2">
              Sample 1 (comma or space separated)
            </label>
            <textarea
              value={sample1}
              onChange={(e) => setSample1(e.target.value)}
              placeholder="e.g., 12.5, 13.1, 11.8, 14.2, 12.9"
              rows="3"
              className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 placeholder-gray-400 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
              required
            />
          </div>

          {/* Sample 2 */}
          {(testType === 't-test' || testType === 'f-test') && testType !== 'z-test' && (
            <div>
              <label className="block text-gray-200 font-medium mb-2">
                Sample 2 {testType === 'f-test' ? '(required)' : '(optional for one-sample t-test)'}
              </label>
              <textarea
                value={sample2}
                onChange={(e) => setSample2(e.target.value)}
                placeholder="e.g., 10.2, 11.5, 10.8, 11.9, 10.1"
                rows="3"
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 placeholder-gray-400 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          )}

          {/* z-test specific options */}
          {testType === 'z-test' && (
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
          {testType === 't-test' && (
            <>
              {!sample2 && (
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
              )}

              {sample2 && (
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

      {/* Results Display */}
      {result && <ResultCard result={result} />}
    </div>
  )
}

export default HypothesisTesting
