import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import ResultCard from '../components/ResultCard'
import { Beaker, Plus, Trash2 } from 'lucide-react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const FactorialDesigns = () => {
  const [designType, setDesignType] = useState('2k')
  const [factorNames, setFactorNames] = useState('A,B,C')
  const [responseName, setResponseName] = useState('Yield')
  const [alpha, setAlpha] = useState(0.05)
  const [numReplicates, setNumReplicates] = useState(1)
  const [tableData, setTableData] = useState([])
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const isFirstRender = useRef(true)

  // Get factors array
  const factors = factorNames.split(',').map(f => f.trim()).filter(f => f.length > 0)
  const numCols = factors.length + 1 // factors + response
  const numFactors = factors.length

  // Generate random number from normal distribution using Box-Muller transform
  const randomNormal = (mean = 65, stdDev = 15) => {
    const u1 = Math.random()
    const u2 = Math.random()
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2)
    return mean + z0 * stdDev
  }

  // Generate sample response data for different design sizes
  const generateSampleResponses = (numFactors, levels, replicates) => {
    const numTreatments = Math.pow(levels, numFactors)
    const numRuns = numTreatments * replicates

    // Generate random responses from normal distribution
    // For 3-level designs, add some curvature effects
    if (levels === 3) {
      return Array.from({ length: numRuns }, (_, i) => {
        // Add quadratic effect to simulate curvature
        const baseValue = randomNormal(65, 12)
        const treatmentIdx = Math.floor(i / replicates)
        const curvatureEffect = (treatmentIdx % 3 - 1) * (treatmentIdx % 3 - 1) * 3 // Quadratic term
        return (Math.round((baseValue + curvatureEffect) * 10) / 10).toFixed(1)
      })
    } else {
      // 2-level design
      return Array.from({ length: numRuns }, () =>
        Math.round(randomNormal(65, 15) * 10) / 10
      ).map(val => val.toFixed(1))
    }
  }

  // Generate all combinations for factorial design with replicates
  const generateFactorialRuns = (numFactors, levels, replicates) => {
    if (numFactors === 0) return []

    const numTreatments = Math.pow(levels, numFactors)
    const runs = []

    // Generate each treatment combination
    for (let i = 0; i < numTreatments; i++) {
      const treatment = []
      let temp = i

      for (let j = 0; j < numFactors; j++) {
        const level = temp % levels
        temp = Math.floor(temp / levels)

        if (levels === 2) {
          treatment.push(level === 0 ? 'Low' : 'High')
        } else if (levels === 3) {
          treatment.push(level === 0 ? 'Low' : level === 1 ? 'Medium' : 'High')
        }
      }

      // Add this treatment combination 'replicates' times
      for (let rep = 0; rep < replicates; rep++) {
        const run = [...treatment, ''] // Copy treatment and add empty response
        runs.push(run)
      }
    }

    return runs
  }

  // Regenerate table when number of factors, design type, or replicates change
  useEffect(() => {
    const levels = designType === '2k' ? 2 : 3
    const maxFactors = designType === '2k' ? 6 : 4 // 3^4 = 81 runs max for 3k
    const reps = designType === '2k' ? numReplicates : 1 // Only 2k supports replicates for now

    if (numFactors > 0 && numFactors <= maxFactors) {
      const newRuns = generateFactorialRuns(numFactors, levels, reps)

      // Add sample response values from normal distribution
      const sampleResponses = generateSampleResponses(numFactors, levels, reps)
      newRuns.forEach((run, i) => {
        run[run.length - 1] = sampleResponses[i] || ''
      })

      setTableData(newRuns)
      setResult(null) // Clear previous results
      isFirstRender.current = false
    } else if (numFactors === 0) {
      setTableData([])
    }
  }, [numFactors, designType, numReplicates])

  const handleCellChange = (rowIndex, colIndex, value) => {
    const newData = [...tableData]
    newData[rowIndex][colIndex] = value
    setTableData(newData)
  }

  const addRow = () => {
    const newRow = Array(numCols).fill('')
    setTableData([...tableData, newRow])
  }

  const removeRow = (rowIndex) => {
    if (tableData.length > 1) {
      setTableData(tableData.filter((_, idx) => idx !== rowIndex))
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      // Convert table data to API format
      if (tableData.length < 2) {
        throw new Error('Please provide at least 2 rows of data')
      }

      // Filter out empty rows
      const validRows = tableData.filter(row =>
        row.some(cell => cell !== null && cell !== undefined && cell.toString().trim() !== '')
      )

      if (validRows.length < 2) {
        throw new Error('Please provide at least 2 valid rows of data')
      }

      const data = validRows.map((row, idx) => {
        const rowData = {}

        // Check if row has all values
        if (row.length !== numCols) {
          throw new Error(`Row ${idx + 1}: Expected ${numCols} values, got ${row.length}`)
        }

        // Assign factor values
        factors.forEach((factor, i) => {
          if (!row[i] || row[i].toString().trim() === '') {
            throw new Error(`Row ${idx + 1}: Missing value for ${factor}`)
          }
          rowData[factor] = row[i].toString().trim()
        })

        // Assign response value
        const responseValue = row[row.length - 1]
        if (!responseValue || responseValue.toString().trim() === '') {
          throw new Error(`Row ${idx + 1}: Missing response value`)
        }

        const parsedResponse = parseFloat(responseValue)
        if (isNaN(parsedResponse)) {
          throw new Error(`Row ${idx + 1}: Response value must be a number, got "${responseValue}"`)
        }

        rowData[responseName] = parsedResponse

        return rowData
      })

      const payload = {
        data: data,
        factors: factors,
        response: responseName,
        alpha
      }

      // Choose endpoint based on design type
      const endpoint = designType === '2k'
        ? `${API_URL}/api/factorial/full-factorial`
        : `${API_URL}/api/factorial/three-level-factorial`

      const response = await axios.post(endpoint, payload)
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
          <Beaker className="w-8 h-8 text-purple-400" />
          <h2 className="text-3xl font-bold text-gray-100">Factorial Designs</h2>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Design Type */}
          <div>
            <label className="block text-gray-100 font-medium mb-2">Design Type</label>
            <select
              value={designType}
              onChange={(e) => setDesignType(e.target.value)}
              className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-purple-500"
            >
              <option value="2k">2^k Full Factorial (2 levels: Low/High)</option>
              <option value="3k">3^k Full Factorial (3 levels: Low/Medium/High)</option>
              <option value="fractional" disabled>Fractional Factorial (Coming Soon)</option>
            </select>
            <p className="text-gray-400 text-xs mt-1">
              2^k for screening/main effects • 3^k for optimization/curvature detection
            </p>
          </div>

          {/* Factor Names */}
          <div>
            <label className="block text-gray-100 font-medium mb-2">
              Factor Names (comma-separated)
            </label>
            <input
              type="text"
              value={factorNames}
              onChange={(e) => setFactorNames(e.target.value)}
              className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-purple-500"
              placeholder="e.g., Temperature, Pressure, Time"
              required
            />
            <p className="text-gray-400 text-xs mt-1">
              Enter factor names separated by commas (e.g., A,B,C or Temperature,Pressure,Catalyst)
            </p>
          </div>

          {/* Response Variable Name */}
          <div>
            <label className="block text-gray-100 font-medium mb-2">
              Response Variable Name
            </label>
            <input
              type="text"
              value={responseName}
              onChange={(e) => setResponseName(e.target.value)}
              className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-purple-500"
              placeholder="e.g., Yield, Strength, Quality"
              required
            />
          </div>

          {/* Number of Replicates (2k only) */}
          {designType === '2k' && (
            <div>
              <label className="block text-gray-100 font-medium mb-2">
                Number of Replicates
              </label>
              <input
                type="number"
                min="1"
                max="5"
                value={numReplicates}
                onChange={(e) => setNumReplicates(parseInt(e.target.value) || 1)}
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
              <p className="text-gray-400 text-xs mt-1">
                Replicates allow estimation of pure error for lack-of-fit testing. Each treatment combination will be run {numReplicates} time{numReplicates > 1 ? 's' : ''}.
              </p>
            </div>
          )}

          {/* Data Input Table */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="block text-gray-100 font-medium">
                Experimental Data
              </label>
              <button
                type="button"
                onClick={addRow}
                className="flex items-center space-x-1 px-3 py-1 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors text-sm"
              >
                <Plus className="w-4 h-4" />
                <span>Add Row</span>
              </button>
            </div>

            <div className="overflow-x-auto bg-slate-700/30 rounded-lg border border-slate-600">
              <table className="w-full">
                <thead>
                  <tr className="bg-slate-700/50">
                    <th className="px-3 py-2 text-left text-gray-100 font-semibold text-sm border-b border-slate-600 w-12">
                      #
                    </th>
                    {factors.map((factor, idx) => (
                      <th
                        key={idx}
                        className="px-3 py-2 text-left text-gray-100 font-semibold text-sm border-b border-slate-600"
                      >
                        {factor}
                      </th>
                    ))}
                    <th className="px-3 py-2 text-left text-gray-100 font-semibold text-sm border-b border-slate-600">
                      {responseName}
                    </th>
                    <th className="px-3 py-2 text-center text-gray-100 font-semibold text-sm border-b border-slate-600 w-16">
                      Action
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {tableData.map((row, rowIndex) => (
                    <tr
                      key={rowIndex}
                      className="border-b border-slate-700/30 hover:bg-slate-700/20"
                    >
                      <td className="px-3 py-2 text-gray-300 text-sm">
                        {rowIndex + 1}
                      </td>
                      {row.map((cell, colIndex) => (
                        <td key={colIndex} className="px-2 py-1">
                          <input
                            type="text"
                            value={cell}
                            onChange={(e) => handleCellChange(rowIndex, colIndex, e.target.value)}
                            className="w-full px-2 py-1 bg-slate-800/50 text-gray-100 border border-slate-600 rounded focus:outline-none focus:ring-1 focus:ring-purple-500 text-sm"
                            placeholder={colIndex === row.length - 1 ? '0.0' : 'Low/High'}
                          />
                        </td>
                      ))}
                      <td className="px-3 py-2 text-center">
                        <button
                          type="button"
                          onClick={() => removeRow(rowIndex)}
                          disabled={tableData.length === 1}
                          className="p-1 text-red-400 hover:text-red-300 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                          title="Remove row"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <p className="text-gray-400 text-xs mt-2">
              {designType === '2k'
                ? 'Table automatically generates 2^k runs (max 6 factors = 64 runs). Use "Low"/"High", "-1"/"1", or "0"/"1" for levels.'
                : 'Table automatically generates 3^k runs (max 4 factors = 81 runs). Use "Low"/"Medium"/"High" or "-1"/"0"/"1" for levels.'}
              {' '}Response values must be numbers. {tableData.length > 0 && `Current: ${tableData.length} runs.`}
            </p>
          </div>

          {/* Significance Level */}
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
              className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-purple-500"
            />
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-purple-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-purple-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Analyzing...' : 'Analyze Factorial Design'}
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

export default FactorialDesigns
