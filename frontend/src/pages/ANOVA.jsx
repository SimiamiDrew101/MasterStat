import { useState } from 'react'
import axios from 'axios'
import ResultCard from '../components/ResultCard'
import { TrendingUp } from 'lucide-react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const ANOVA = () => {
  const [analysisType, setAnalysisType] = useState('one-way')
  const [groups, setGroups] = useState([
    { name: 'Group A', values: '' },
    { name: 'Group B', values: '' },
    { name: 'Group C', values: '' }
  ])
  const [alpha, setAlpha] = useState(0.05)

  // Two-way ANOVA state
  const [factorA, setFactorA] = useState('Temperature')
  const [factorB, setFactorB] = useState('Pressure')
  const [responseName, setResponseName] = useState('Yield')
  const [twoWayData, setTwoWayData] = useState('')

  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const parseSample = (text) => {
    return text.split(/[,\s]+/).filter(x => x).map(x => parseFloat(x))
  }

  const addGroup = () => {
    setGroups([...groups, { name: `Group ${String.fromCharCode(65 + groups.length)}`, values: '' }])
  }

  const removeGroup = (index) => {
    if (groups.length > 2) {
      setGroups(groups.filter((_, i) => i !== index))
    }
  }

  const updateGroup = (index, field, value) => {
    const newGroups = [...groups]
    newGroups[index][field] = value
    setGroups(newGroups)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      if (analysisType === 'one-way') {
        const groupsData = {}
        groups.forEach(group => {
          if (group.values.trim()) {
            groupsData[group.name] = parseSample(group.values)
          }
        })

        if (Object.keys(groupsData).length < 2) {
          throw new Error('Please provide data for at least 2 groups')
        }

        const payload = {
          groups: groupsData,
          alpha
        }

        const response = await axios.post(`${API_URL}/api/anova/one-way`, payload)
        setResult(response.data)
      } else if (analysisType === 'two-way') {
        // Parse two-way data (CSV format: factorA, factorB, response)
        const lines = twoWayData.trim().split('\n')
        if (lines.length < 2) {
          throw new Error('Please provide at least 2 rows of data')
        }

        const data = lines.map(line => {
          const parts = line.split(/[,\t]/).map(p => p.trim())
          if (parts.length !== 3) {
            throw new Error('Each row must have 3 values: Factor A, Factor B, Response')
          }
          return {
            [factorA]: parts[0],
            [factorB]: parts[1],
            [responseName]: parseFloat(parts[2])
          }
        })

        const payload = {
          data: data,
          factor_a: factorA,
          factor_b: factorB,
          response: responseName,
          alpha
        }

        const response = await axios.post(`${API_URL}/api/anova/two-way`, payload)
        setResult(response.data)
      }
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
          <TrendingUp className="w-8 h-8 text-gray-100" />
          <h2 className="text-3xl font-bold text-gray-100">ANOVA Analysis</h2>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Analysis Type */}
          <div>
            <label className="block text-gray-100 font-medium mb-2">Analysis Type</label>
            <select
              value={analysisType}
              onChange={(e) => setAnalysisType(e.target.value)}
              className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="one-way">One-Way ANOVA</option>
              <option value="two-way">Two-Way ANOVA</option>
            </select>
          </div>

          {/* One-Way ANOVA: Groups */}
          {analysisType === 'one-way' && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <label className="text-gray-100 font-medium">Groups</label>
                <button
                  type="button"
                  onClick={addGroup}
                  className="bg-slate-700/50 text-gray-100 px-4 py-2 rounded-lg hover:bg-white/30 transition-colors text-sm"
                >
                  + Add Group
                </button>
              </div>

              {groups.map((group, index) => (
                <div key={index} className="bg-white/5 rounded-lg p-4 space-y-2">
                  <div className="flex items-center justify-between">
                    <input
                      type="text"
                      value={group.name}
                      onChange={(e) => updateGroup(index, 'name', e.target.value)}
                      className="bg-slate-700/50 text-gray-100 px-3 py-1 rounded-lg border border-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
                      placeholder="Group name"
                    />
                    {groups.length > 2 && (
                      <button
                        type="button"
                        onClick={() => removeGroup(index)}
                        className="text-red-300 hover:text-red-200 text-sm"
                      >
                        Remove
                      </button>
                    )}
                  </div>
                  <textarea
                    value={group.values}
                    onChange={(e) => updateGroup(index, 'values', e.target.value)}
                    placeholder="Enter values (comma or space separated)"
                    rows="2"
                    className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 placeholder-gray-400 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              ))}
            </div>
          )}

          {/* Two-Way ANOVA: Factor Configuration and Data */}
          {analysisType === 'two-way' && (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="block text-gray-100 font-medium mb-2">Factor A Name</label>
                  <input
                    type="text"
                    value={factorA}
                    onChange={(e) => setFactorA(e.target.value)}
                    className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="e.g., Temperature"
                  />
                </div>
                <div>
                  <label className="block text-gray-100 font-medium mb-2">Factor B Name</label>
                  <input
                    type="text"
                    value={factorB}
                    onChange={(e) => setFactorB(e.target.value)}
                    className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="e.g., Pressure"
                  />
                </div>
                <div>
                  <label className="block text-gray-100 font-medium mb-2">Response Variable</label>
                  <input
                    type="text"
                    value={responseName}
                    onChange={(e) => setResponseName(e.target.value)}
                    className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="e.g., Yield"
                  />
                </div>
              </div>

              <div>
                <label className="block text-gray-100 font-medium mb-2">
                  Data (Factor A, Factor B, Response - one row per observation)
                </label>
                <textarea
                  value={twoWayData}
                  onChange={(e) => setTwoWayData(e.target.value)}
                  placeholder={`Example:\nLow, Low, 23.5\nLow, High, 28.3\nMedium, Low, 31.2\nMedium, High, 35.8\nHigh, Low, 29.1\nHigh, High, 38.4`}
                  rows="8"
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 placeholder-gray-400 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-500 font-mono text-sm"
                  required
                />
                <p className="text-gray-400 text-xs mt-2">
                  Enter one observation per line. Separate values with commas or tabs.
                  Each line should have: Factor A level, Factor B level, Response value
                </p>
              </div>
            </div>
          )}

          {/* Alpha */}
          <div>
            <label className="block text-gray-100 font-medium mb-2">
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
            {loading ? 'Calculating...' : 'Run ANOVA'}
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

export default ANOVA
