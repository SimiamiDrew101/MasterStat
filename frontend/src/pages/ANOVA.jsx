import { useState, useEffect } from 'react'
import axios from 'axios'
import ResultCard from '../components/ResultCard'
import AssumptionTestsANOVA from '../components/AssumptionTestsANOVA'
import EffectSizePanel from '../components/EffectSizePanel'
import InfluenceDiagnostics from '../components/InfluenceDiagnostics'
import DiagnosticPlots from '../components/DiagnosticPlots'
import ContrastsPanel from '../components/ContrastsPanel'
import { TrendingUp } from 'lucide-react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const ANOVA = () => {
  const [activeTab, setActiveTab] = useState('one-way')
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

  // One-Way ANOVA state
  const [oneWayResult, setOneWayResult] = useState(null)
  const [oneWayLoading, setOneWayLoading] = useState(false)
  const [oneWayError, setOneWayError] = useState(null)

  // Two-Way ANOVA state
  const [twoWayResult, setTwoWayResult] = useState(null)
  const [twoWayLoading, setTwoWayLoading] = useState(false)
  const [twoWayError, setTwoWayError] = useState(null)

  // Post-hoc test state (One-Way)
  const [postHocResult, setPostHocResult] = useState(null)
  const [postHocLoading, setPostHocLoading] = useState(false)
  const [postHocError, setPostHocError] = useState(null)

  // Post-hoc test state (Two-Way)
  const [twoWayPostHocResult, setTwoWayPostHocResult] = useState(null)
  const [twoWayPostHocLoading, setTwoWayPostHocLoading] = useState(false)
  const [twoWayPostHocError, setTwoWayPostHocError] = useState(null)

  // Two-way post-hoc state
  const [twoWayComparisonType, setTwoWayComparisonType] = useState('marginal_a')
  const [twoWayTestMethod, setTwoWayTestMethod] = useState('tukey')

  // Contrasts state
  const [contrastsResult, setContrastsResult] = useState(null)
  const [contrastsLoading, setContrastsLoading] = useState(false)
  const [contrastsError, setContrastsError] = useState(null)
  const [contrastType, setContrastType] = useState('polynomial')
  const [polynomialDegree, setPolynomialDegree] = useState(1)
  const [customCoefficients, setCustomCoefficients] = useState('')

  // Table data for One-Way ANOVA
  const [oneWayTableData, setOneWayTableData] = useState(
    Array(10).fill(null).map(() => Array(groups.length).fill(''))
  )

  // Table data for Two-Way ANOVA
  const [twoWayTableData, setTwoWayTableData] = useState(
    Array(10).fill(null).map(() => Array(3).fill('')) // Factor A, Factor B, Response
  )

  // Sync table data when groups change
  useEffect(() => {
    // Ensure table has the right number of columns for groups
    if (oneWayTableData.length > 0 && oneWayTableData[0].length !== groups.length) {
      const newData = oneWayTableData.map(row => {
        const newRow = Array(groups.length).fill('')
        row.forEach((val, idx) => {
          if (idx < groups.length) {
            newRow[idx] = val
          }
        })
        return newRow
      })
      setOneWayTableData(newData)
    }
  }, [groups.length])

  const parseSample = (text) => {
    return text.split(/[,\s]+/).filter(x => x).map(x => parseFloat(x))
  }

  const addGroup = () => {
    setGroups([...groups, { name: `Group ${String.fromCharCode(65 + groups.length)}`, values: '' }])
    // Add a new column to table data
    setOneWayTableData(oneWayTableData.map(row => [...row, '']))
  }

  const removeGroup = (index) => {
    if (groups.length > 2) {
      setGroups(groups.filter((_, i) => i !== index))
      // Remove column from table data
      setOneWayTableData(oneWayTableData.map(row => row.filter((_, i) => i !== index)))
    }
  }

  const updateGroup = (index, field, value) => {
    const newGroups = [...groups]
    newGroups[index][field] = value
    setGroups(newGroups)
  }

  // Handle cell changes for One-Way ANOVA
  const handleOneWayCellChange = (rowIndex, colIndex, value) => {
    const newData = [...oneWayTableData]
    newData[rowIndex][colIndex] = value
    setOneWayTableData(newData)

    // Auto-add row if typing in last row
    if (rowIndex === oneWayTableData.length - 1 && value.trim() !== '') {
      setOneWayTableData([...newData, Array(groups.length).fill('')])
    }
  }

  // Handle cell changes for Two-Way ANOVA
  const handleTwoWayCellChange = (rowIndex, colIndex, value) => {
    const newData = [...twoWayTableData]
    newData[rowIndex][colIndex] = value
    setTwoWayTableData(newData)

    // Auto-add row if typing in last row
    if (rowIndex === twoWayTableData.length - 1 && value.trim() !== '') {
      setTwoWayTableData([...newData, Array(3).fill('')])
    }
  }

  // Generate random normal data
  const generateRandomNormal = (mean = 0, stdDev = 1) => {
    // Box-Muller transform
    const u1 = Math.random()
    const u2 = Math.random()
    const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
    return mean + z0 * stdDev
  }

  // Populate One-Way ANOVA with sample data
  const populateOneWaySampleData = () => {
    const numRows = 8 // Sample size per group
    const newData = Array(numRows).fill(null).map(() =>
      groups.map((_, groupIndex) => {
        // Different means for different groups to show effect
        const mean = 50 + groupIndex * 5
        const stdDev = 5
        return generateRandomNormal(mean, stdDev).toFixed(2)
      })
    )
    setOneWayTableData(newData)
  }

  // Populate Two-Way ANOVA with sample data
  const populateTwoWaySampleData = () => {
    // Use current factor names or defaults
    const currentFactorA = factorA || 'Temperature'
    const currentFactorB = factorB || 'Pressure'

    // Determine sensible levels based on factor names
    let factorALevels, factorBLevels

    if (currentFactorA.toLowerCase().includes('temp')) {
      factorALevels = ['Cold', 'Warm', 'Hot']
    } else {
      factorALevels = ['Low', 'Medium', 'High']
    }

    if (currentFactorB.toLowerCase().includes('press')) {
      factorBLevels = ['Low', 'High']
    } else if (currentFactorB.toLowerCase().includes('temp')) {
      factorBLevels = ['Cold', 'Hot']
    } else {
      factorBLevels = ['Level1', 'Level2']
    }

    const numReplicates = 3 // 3 replicates per combination

    const newData = []
    factorALevels.forEach((levelA, aIndex) => {
      factorBLevels.forEach((levelB, bIndex) => {
        for (let rep = 0; rep < numReplicates; rep++) {
          // Base mean + main effect A + main effect B + interaction
          const mean = 50 + aIndex * 10 + bIndex * 5 + (aIndex * bIndex * 2)
          const stdDev = 4
          newData.push([
            levelA,
            levelB,
            generateRandomNormal(mean, stdDev).toFixed(2)
          ])
        }
      })
    })
    setTwoWayTableData(newData)
  }

  // Handle keyboard navigation
  const handleKeyDown = (e, rowIndex, colIndex, isOneWay = true) => {
    const tableData = isOneWay ? oneWayTableData : twoWayTableData
    const numCols = isOneWay ? groups.length : 3
    const numRows = tableData.length

    let newRow = rowIndex
    let newCol = colIndex

    switch (e.key) {
      case 'ArrowUp':
        e.preventDefault()
        newRow = Math.max(0, rowIndex - 1)
        break
      case 'ArrowDown':
      case 'Enter':
        e.preventDefault()
        newRow = Math.min(numRows - 1, rowIndex + 1)
        break
      case 'ArrowLeft':
        e.preventDefault()
        newCol = Math.max(0, colIndex - 1)
        break
      case 'ArrowRight':
      case 'Tab':
        if (e.key === 'Tab') e.preventDefault()
        newCol = Math.min(numCols - 1, colIndex + 1)
        break
      default:
        return
    }

    // Focus the new cell
    const cellId = isOneWay
      ? `oneway-cell-${newRow}-${newCol}`
      : `twoway-cell-${newRow}-${newCol}`
    const input = document.getElementById(cellId)
    if (input) {
      input.focus()
      input.select()
    }
  }

  const handleOneWaySubmit = async (e) => {
    e.preventDefault()
    setOneWayLoading(true)
    setOneWayError(null)
    setOneWayResult(null)
    setPostHocResult(null)
    setContrastsResult(null)

    try {
      // Extract data from table
      const groupsData = {}
      groups.forEach((group, colIndex) => {
        const values = oneWayTableData
          .map(row => row[colIndex])
          .filter(val => val !== '' && !isNaN(parseFloat(val)))
          .map(val => parseFloat(val))

        if (values.length > 0) {
          groupsData[group.name] = values
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
      setOneWayResult(response.data)
    } catch (err) {
      console.error('One-Way ANOVA error:', err)
      const errorMessage = err.response?.data?.detail || err.message || 'An error occurred during analysis'
      setOneWayError(errorMessage)
      setOneWayResult(null)
    } finally {
      setOneWayLoading(false)
    }
  }

  const handleTwoWaySubmit = async (e) => {
    e.preventDefault()
    setTwoWayLoading(true)
    setTwoWayError(null)
    setTwoWayResult(null)
    setTwoWayPostHocResult(null)

    try {
      // Extract data from table
      const data = twoWayTableData
        .filter(row => {
          // Check if all three columns have values
          const hasFactorA = row[0] && String(row[0]).trim() !== ''
          const hasFactorB = row[1] && String(row[1]).trim() !== ''
          const hasResponse = row[2] && String(row[2]).trim() !== '' && !isNaN(parseFloat(row[2]))
          return hasFactorA && hasFactorB && hasResponse
        })
        .map(row => ({
          [factorA]: String(row[0]).trim(),
          [factorB]: String(row[1]).trim(),
          [responseName]: parseFloat(row[2])
        }))

      if (data.length < 2) {
        throw new Error('Please provide at least 2 complete rows of data with valid response values')
      }

      // Validate factor names
      if (!factorA || !factorB || !responseName) {
        throw new Error('Please provide names for both factors and the response variable')
      }

      const payload = {
        data: data,
        factor_a: factorA,
        factor_b: factorB,
        response: responseName,
        alpha
      }

      console.log('Sending two-way ANOVA request:', payload)
      const response = await axios.post(`${API_URL}/api/anova/two-way`, payload)
      console.log('Received response:', response.data)
      setTwoWayResult(response.data)
    } catch (err) {
      console.error('Two-Way ANOVA error:', err)
      const errorMessage = err.response?.data?.detail || err.message || 'An error occurred during analysis'
      setTwoWayError(errorMessage)
      setTwoWayResult(null)
    } finally {
      setTwoWayLoading(false)
    }
  }

  const handleOneWayPostHoc = async (method) => {
    setPostHocLoading(true)
    setPostHocError(null)
    setPostHocResult(null)

    try {
      // Extract data from table
      const groupsData = {}
      groups.forEach((group, colIndex) => {
        const values = oneWayTableData
          .map(row => row[colIndex])
          .filter(val => val !== '' && !isNaN(parseFloat(val)))
          .map(val => parseFloat(val))

        if (values.length > 0) {
          groupsData[group.name] = values
        }
      })

      const payload = {
        groups: groupsData,
        alpha
      }

      const response = await axios.post(`${API_URL}/api/anova/post-hoc/${method}`, payload)
      setPostHocResult(response.data)
    } catch (err) {
      setPostHocError(err.response?.data?.detail || err.message || 'An error occurred')
    } finally {
      setPostHocLoading(false)
    }
  }

  const handleTwoWayPostHoc = async (method) => {
    setTwoWayPostHocLoading(true)
    setTwoWayPostHocError(null)
    setTwoWayPostHocResult(null)

    try {
      // Extract data from table
      const data = twoWayTableData
        .filter(row => {
          const hasFactorA = row[0] && String(row[0]).trim() !== ''
          const hasFactorB = row[1] && String(row[1]).trim() !== ''
          const hasResponse = row[2] && String(row[2]).trim() !== '' && !isNaN(parseFloat(row[2]))
          return hasFactorA && hasFactorB && hasResponse
        })
        .map(row => ({
          [factorA]: String(row[0]).trim(),
          [factorB]: String(row[1]).trim(),
          [responseName]: parseFloat(row[2])
        }))

      const payload = {
        data,
        factor_a: factorA,
        factor_b: factorB,
        response: responseName,
        comparison_type: twoWayComparisonType,
        test_method: method,
        alpha
      }

      const response = await axios.post(`${API_URL}/api/anova/post-hoc/two-way`, payload)
      setTwoWayPostHocResult(response.data)
    } catch (err) {
      setTwoWayPostHocError(err.response?.data?.detail || err.message || 'An error occurred')
    } finally {
      setTwoWayPostHocLoading(false)
    }
  }

  const handleContrasts = async () => {
    setContrastsLoading(true)
    setContrastsError(null)
    setContrastsResult(null)

    try {
      // Extract data from table (contrasts only work for one-way ANOVA)
      const groupsData = {}
      groups.forEach((group, colIndex) => {
        const values = oneWayTableData
          .map(row => row[colIndex])
          .filter(val => val !== '' && !isNaN(parseFloat(val)))
          .map(val => parseFloat(val))

        if (values.length > 0) {
          groupsData[group.name] = values
        }
      })

      const payload = {
        groups: groupsData,
        contrast_type: contrastType,
        alpha
      }

      if (contrastType === 'custom') {
        // Parse custom coefficients
        const coeffs = customCoefficients
          .split(',')
          .map(c => parseFloat(c.trim()))
          .filter(c => !isNaN(c))

        if (coeffs.length !== groups.length) {
          throw new Error(`Number of coefficients (${coeffs.length}) must match number of groups (${groups.length})`)
        }

        // Check if coefficients sum to 0
        const sum = coeffs.reduce((a, b) => a + b, 0)
        if (Math.abs(sum) > 0.0001) {
          throw new Error(`Coefficients must sum to 0 (current sum: ${sum.toFixed(4)})`)
        }

        payload.coefficients = coeffs
      } else if (contrastType === 'polynomial') {
        payload.polynomial_degree = polynomialDegree
      }

      const response = await axios.post(`${API_URL}/api/anova/contrasts`, payload)
      setContrastsResult(response.data)
    } catch (err) {
      setContrastsError(err.response?.data?.detail || err.message || 'An error occurred')
    } finally {
      setContrastsLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <div className="flex items-center space-x-3 mb-4">
          <TrendingUp className="w-8 h-8 text-cyan-400" />
          <h2 className="text-3xl font-bold text-gray-100">ANOVA Analysis</h2>
        </div>
        <p className="text-gray-300">
          Analysis of Variance: Compare means across groups and identify significant differences. Includes assumptions testing, effect sizes, and advanced diagnostics.
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl border border-slate-700/50 overflow-hidden">
        <div className="grid grid-cols-2">
          <button
            onClick={() => setActiveTab('one-way')}
            className={`px-6 py-4 font-semibold text-sm transition-all ${
              activeTab === 'one-way'
                ? 'bg-cyan-500/20 text-cyan-400 border-b-2 border-cyan-500'
                : 'text-gray-400 hover:text-gray-300 hover:bg-slate-700/30'
            }`}
          >
            One-Way ANOVA
          </button>
          <button
            onClick={() => setActiveTab('two-way')}
            className={`px-6 py-4 font-semibold text-sm transition-all ${
              activeTab === 'two-way'
                ? 'bg-purple-500/20 text-purple-400 border-b-2 border-purple-500'
                : 'text-gray-400 hover:text-gray-300 hover:bg-slate-700/30'
            }`}
          >
            Two-Way ANOVA
          </button>
        </div>
      </div>

      {/* One-Way ANOVA Tab */}
      {activeTab === 'one-way' && (
        <>
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <form onSubmit={handleOneWaySubmit} className="space-y-6">
          {/* One-Way ANOVA: Groups */}
          <div className="space-y-4">
            <div className="space-y-4">
              <div className="flex items-center justify-between mb-2">
                <label className="text-gray-100 font-medium">Data Entry (Excel-like navigation with arrow keys)</label>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={populateOneWaySampleData}
                    className="bg-gradient-to-r from-purple-600 to-blue-600 text-white px-4 py-2 rounded-lg hover:from-purple-700 hover:to-blue-700 transition-colors text-sm font-medium"
                  >
                    Generate Sample Data
                  </button>
                  <button
                    type="button"
                    onClick={addGroup}
                    className="bg-slate-700/50 text-gray-100 px-4 py-2 rounded-lg hover:bg-white/30 transition-colors text-sm"
                  >
                    + Add Group
                  </button>
                </div>
              </div>

              {/* Group name editors */}
              <div className="flex gap-2 mb-2">
                {groups.map((group, index) => (
                  <div key={index} className="flex-1 flex items-center gap-2">
                    <input
                      type="text"
                      value={group.name}
                      onChange={(e) => updateGroup(index, 'name', e.target.value)}
                      className="flex-1 bg-slate-700/50 text-gray-100 px-3 py-1 rounded-lg border border-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-500 text-center font-semibold"
                      placeholder="Group name"
                    />
                    {groups.length > 2 && (
                      <button
                        type="button"
                        onClick={() => removeGroup(index)}
                        className="text-red-300 hover:text-red-200 text-sm px-2"
                      >
                        ✕
                      </button>
                    )}
                  </div>
                ))}
              </div>

              {/* Excel-like table */}
              <div className="overflow-auto max-h-96 border border-slate-600 rounded-lg">
                <table className="w-full border-collapse">
                  <thead className="sticky top-0 bg-slate-700/50 z-10">
                    <tr>
                      {groups.map((group, colIndex) => (
                        <th key={colIndex} className="border border-slate-600 py-2 px-3 text-gray-100 font-semibold min-w-[120px]">
                          {group.name}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {oneWayTableData.map((row, rowIndex) => (
                      <tr key={rowIndex}>
                        {row.map((cell, colIndex) => (
                          <td key={colIndex} className="border border-slate-600 p-0">
                            <input
                              id={`oneway-cell-${rowIndex}-${colIndex}`}
                              type="text"
                              value={cell}
                              onChange={(e) => handleOneWayCellChange(rowIndex, colIndex, e.target.value)}
                              onKeyDown={(e) => handleKeyDown(e, rowIndex, colIndex, true)}
                              className="w-full px-3 py-2 bg-transparent text-gray-100 focus:bg-slate-700/30 focus:outline-none"
                              placeholder="Value"
                            />
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-gray-400 text-xs mt-2">
                Use arrow keys, Tab, or Enter to navigate between cells. New rows are added automatically.
              </p>
            </div>
          </div>

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
            disabled={oneWayLoading}
            className="w-full bg-cyan-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-cyan-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {oneWayLoading ? 'Calculating...' : 'Run One-Way ANOVA'}
          </button>
        </form>
      </div>

      {/* Error Display */}
      {oneWayError && (
        <div className="bg-red-900/30 backdrop-blur-lg rounded-xl p-4 border border-red-700/50">
          <p className="text-red-200 font-medium">Error: {oneWayError}</p>
        </div>
      )}

      {/* Results Display */}
      {oneWayResult && <ResultCard result={oneWayResult} />}

      {/* Effect Sizes */}
      {oneWayResult && oneWayResult.effect_sizes && (
        <EffectSizePanel effectSizes={oneWayResult.effect_sizes} testType={oneWayResult.test_type} />
      )}

      {/* Assumptions Testing */}
      {oneWayResult && oneWayResult.assumptions && (
        <AssumptionTestsANOVA assumptions={oneWayResult.assumptions} />
      )}

      {/* Influence Diagnostics */}
      {oneWayResult && oneWayResult.influence_diagnostics && (
        <InfluenceDiagnostics influenceData={oneWayResult.influence_diagnostics} />
      )}

      {/* Diagnostic Plots */}
      {oneWayResult && oneWayResult.diagnostic_plots && (
        <DiagnosticPlots diagnosticPlots={oneWayResult.diagnostic_plots} />
      )}

      {/* Post-hoc Tests Section (One-Way) */}
      {oneWayResult && (
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
          <div className="mb-6">
            <h3 className="text-2xl font-bold text-gray-100 mb-2">Post-hoc Multiple Comparisons</h3>
            <p className="text-gray-300 text-sm">
              The ANOVA detected significant differences between groups. Use post-hoc tests to identify which specific groups differ.
            </p>
          </div>

          {/* Post-hoc Test Buttons */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <button
              onClick={() => handleOneWayPostHoc('tukey')}
              disabled={postHocLoading}
              className="bg-gradient-to-r from-blue-600 to-blue-700 text-white font-bold py-3 px-4 rounded-lg hover:from-blue-700 hover:to-blue-800 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Tukey HSD
            </button>
            <button
              onClick={() => handleOneWayPostHoc('bonferroni')}
              disabled={postHocLoading}
              className="bg-gradient-to-r from-purple-600 to-purple-700 text-white font-bold py-3 px-4 rounded-lg hover:from-purple-700 hover:to-purple-800 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Bonferroni
            </button>
            <button
              onClick={() => handleOneWayPostHoc('scheffe')}
              disabled={postHocLoading}
              className="bg-gradient-to-r from-indigo-600 to-indigo-700 text-white font-bold py-3 px-4 rounded-lg hover:from-indigo-700 hover:to-indigo-800 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Scheffe
            </button>
            <button
              onClick={() => handleOneWayPostHoc('fisher-lsd')}
              disabled={postHocLoading}
              className="bg-gradient-to-r from-teal-600 to-teal-700 text-white font-bold py-3 px-4 rounded-lg hover:from-teal-700 hover:to-teal-800 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Fisher's LSD
            </button>
          </div>

          {/* Loading State */}
          {postHocLoading && (
            <div className="text-center py-4">
              <p className="text-gray-300">Running post-hoc analysis...</p>
            </div>
          )}

          {/* Error Display */}
          {postHocError && (
            <div className="bg-red-900/30 backdrop-blur-lg rounded-xl p-4 border border-red-700/50 mb-6">
              <p className="text-red-200 font-medium">Error: {postHocError}</p>
            </div>
          )}

          {/* Post-hoc Results */}
          {postHocResult && (
            <div className="space-y-4">
              <div className="bg-gradient-to-r from-blue-900/30 to-purple-900/30 rounded-xl p-5 border border-blue-500/30">
                <h4 className="text-gray-100 font-bold text-lg mb-2">{postHocResult.test_type}</h4>
                <p className="text-gray-300 text-sm">{postHocResult.description}</p>
                <p className="text-gray-400 text-xs mt-2">Significance level (α): {postHocResult.alpha}</p>
              </div>

              {/* Comparisons Table */}
              <div className="overflow-x-auto">
                <table className="w-full text-gray-100">
                  <thead>
                    <tr className="bg-slate-700/30 border-b border-slate-600">
                      <th className="text-left py-3 px-4">Group 1</th>
                      <th className="text-left py-3 px-4">Group 2</th>
                      <th className="text-right py-3 px-4">Mean Difference</th>
                      <th className="text-right py-3 px-4">Lower CI</th>
                      <th className="text-right py-3 px-4">Upper CI</th>
                      {postHocResult.comparisons[0]?.p_adj !== undefined && (
                        <th className="text-right py-3 px-4">p-value (adj)</th>
                      )}
                      <th className="text-center py-3 px-4">Significant</th>
                    </tr>
                  </thead>
                  <tbody>
                    {postHocResult.comparisons.map((comp, idx) => (
                      <tr key={idx} className={`border-b border-slate-700/30 ${comp.reject ? 'bg-green-900/20' : ''}`}>
                        <td className="py-3 px-4">{comp.group1}</td>
                        <td className="py-3 px-4">{comp.group2}</td>
                        <td className="text-right py-3 px-4 font-mono">{comp.mean_diff}</td>
                        <td className="text-right py-3 px-4 font-mono">{comp.lower_ci}</td>
                        <td className="text-right py-3 px-4 font-mono">{comp.upper_ci}</td>
                        {comp.p_adj !== undefined && (
                          <td className="text-right py-3 px-4 font-mono">{comp.p_adj}</td>
                        )}
                        <td className="text-center py-3 px-4">
                          {comp.reject ? (
                            <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-green-500/30 text-green-300">✓</span>
                          ) : (
                            <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-gray-500/30 text-gray-400">✗</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Summary */}
              <div className="bg-slate-700/30 rounded-lg p-4">
                <p className="text-gray-300 text-sm">
                  <strong>Summary:</strong> {postHocResult.comparisons.filter(c => c.reject).length} out of {postHocResult.comparisons.length} pairwise comparisons are statistically significant at α = {postHocResult.alpha}.
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Contrasts Section (One-Way ANOVA only) */}
      {oneWayResult && (
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
          <div className="mb-6">
            <h3 className="text-2xl font-bold text-gray-100 mb-2">Planned Contrasts</h3>
            <p className="text-gray-300 text-sm">
              Test specific hypotheses about group differences using custom contrast weights or pre-defined contrast types.
            </p>
          </div>

          {/* Contrast Type Selection */}
          <div className="mb-6 bg-slate-700/30 rounded-lg p-4">
            <label className="block text-gray-200 font-medium mb-3">Contrast Type</label>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
              <button
                type="button"
                onClick={() => setContrastType('polynomial')}
                className={`p-3 rounded-lg border-2 transition-all text-left ${
                  contrastType === 'polynomial'
                    ? 'border-purple-500 bg-purple-500/10'
                    : 'border-slate-600 bg-slate-700/50 hover:bg-slate-700'
                }`}
              >
                <h4 className="font-semibold text-gray-100">Polynomial</h4>
                <p className="text-xs text-gray-400">Linear, quadratic, cubic trends</p>
              </button>
              <button
                type="button"
                onClick={() => setContrastType('helmert')}
                className={`p-3 rounded-lg border-2 transition-all text-left ${
                  contrastType === 'helmert'
                    ? 'border-purple-500 bg-purple-500/10'
                    : 'border-slate-600 bg-slate-700/50 hover:bg-slate-700'
                }`}
              >
                <h4 className="font-semibold text-gray-100">Helmert</h4>
                <p className="text-xs text-gray-400">Each vs. mean of subsequent</p>
              </button>
              <button
                type="button"
                onClick={() => setContrastType('custom')}
                className={`p-3 rounded-lg border-2 transition-all text-left ${
                  contrastType === 'custom'
                    ? 'border-purple-500 bg-purple-500/10'
                    : 'border-slate-600 bg-slate-700/50 hover:bg-slate-700'
                }`}
              >
                <h4 className="font-semibold text-gray-100">Custom</h4>
                <p className="text-xs text-gray-400">User-specified coefficients</p>
              </button>
            </div>

            {/* Polynomial Degree Selection */}
            {contrastType === 'polynomial' && (
              <div className="mb-4">
                <label className="block text-gray-200 text-sm font-medium mb-2">Polynomial Degree</label>
                <select
                  value={polynomialDegree}
                  onChange={(e) => setPolynomialDegree(parseInt(e.target.value))}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-purple-500"
                >
                  <option value={1}>Linear (1st degree)</option>
                  <option value={2}>Quadratic (2nd degree)</option>
                  <option value={3}>Cubic (3rd degree)</option>
                </select>
                <p className="text-xs text-gray-400 mt-1">
                  Tests for polynomial trends across ordered groups (e.g., dose levels, time points)
                </p>
              </div>
            )}

            {/* Custom Coefficients Input */}
            {contrastType === 'custom' && (
              <div className="mb-4">
                <label className="block text-gray-200 text-sm font-medium mb-2">
                  Contrast Coefficients (comma-separated, must sum to 0)
                </label>
                <input
                  type="text"
                  value={customCoefficients}
                  onChange={(e) => setCustomCoefficients(e.target.value)}
                  placeholder={`e.g., for ${groups.length} groups: 1, -0.5, -0.5`}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-purple-500"
                />
                <p className="text-xs text-gray-400 mt-1">
                  Example: To compare first group vs average of others with 3 groups, use: 2, -1, -1
                </p>
              </div>
            )}

            <button
              type="button"
              onClick={handleContrasts}
              disabled={contrastsLoading || (contrastType === 'custom' && !customCoefficients)}
              className="w-full bg-purple-600 text-white font-semibold py-2 px-4 rounded-lg hover:bg-purple-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {contrastsLoading ? 'Calculating...' : 'Run Contrasts'}
            </button>
          </div>

          {/* Error Display */}
          {contrastsError && (
            <div className="mb-6 bg-red-900/30 backdrop-blur-lg rounded-xl p-4 border border-red-700/50">
              <p className="text-red-200 font-medium">Error: {contrastsError}</p>
            </div>
          )}

          {/* Contrasts Results */}
          {contrastsResult && <ContrastsPanel contrastsResult={contrastsResult} />}
        </div>
      )}
      </>
      )}

      {/* Two-Way ANOVA Tab */}
      {activeTab === 'two-way' && (
        <>
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <form onSubmit={handleTwoWaySubmit} className="space-y-6">
          {/* Factor Configuration */}
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

          {/* Data Entry */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="block text-gray-100 font-medium">
                Data Entry (Excel-like navigation with arrow keys)
              </label>
              <button
                type="button"
                onClick={populateTwoWaySampleData}
                className="bg-gradient-to-r from-purple-600 to-blue-600 text-white px-4 py-2 rounded-lg hover:from-purple-700 hover:to-blue-700 transition-colors text-sm font-medium"
              >
                Generate Sample Data
              </button>
            </div>

            {/* Excel-like table */}
            <div className="overflow-auto max-h-96 border border-slate-600 rounded-lg">
              <table className="w-full border-collapse">
                <thead className="sticky top-0 bg-slate-700/50 z-10">
                  <tr>
                    <th className="border border-slate-600 py-2 px-3 text-gray-100 font-semibold min-w-[150px]">
                      {factorA || 'Factor A'}
                    </th>
                    <th className="border border-slate-600 py-2 px-3 text-gray-100 font-semibold min-w-[150px]">
                      {factorB || 'Factor B'}
                    </th>
                    <th className="border border-slate-600 py-2 px-3 text-gray-100 font-semibold min-w-[150px]">
                      {responseName || 'Response'}
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {twoWayTableData.map((row, rowIndex) => (
                    <tr key={rowIndex}>
                      {row.map((cell, colIndex) => (
                        <td key={colIndex} className="border border-slate-600 p-0">
                          <input
                            id={`twoway-cell-${rowIndex}-${colIndex}`}
                            type="text"
                            value={cell}
                            onChange={(e) => handleTwoWayCellChange(rowIndex, colIndex, e.target.value)}
                            onKeyDown={(e) => handleKeyDown(e, rowIndex, colIndex, false)}
                            className="w-full px-3 py-2 bg-transparent text-gray-100 focus:bg-slate-700/30 focus:outline-none"
                            placeholder={colIndex === 2 ? 'Number' : 'Level'}
                          />
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-gray-400 text-xs mt-2">
              Use arrow keys, Tab, or Enter to navigate between cells. New rows are added automatically.
            </p>
          </div>

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
            disabled={twoWayLoading}
            className="w-full bg-purple-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-purple-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {twoWayLoading ? 'Calculating...' : 'Run Two-Way ANOVA'}
          </button>
        </form>
      </div>

      {/* Error Display */}
      {twoWayError && (
        <div className="bg-red-900/30 backdrop-blur-lg rounded-xl p-4 border border-red-700/50">
          <p className="text-red-200 font-medium">Error: {twoWayError}</p>
        </div>
      )}

      {/* Results Display */}
      {twoWayResult && <ResultCard result={twoWayResult} />}

      {/* Effect Sizes */}
      {twoWayResult && twoWayResult.effect_sizes && (
        <EffectSizePanel effectSizes={twoWayResult.effect_sizes} testType={twoWayResult.test_type} />
      )}

      {/* Assumptions Testing */}
      {twoWayResult && twoWayResult.assumptions && (
        <AssumptionTestsANOVA assumptions={twoWayResult.assumptions} />
      )}

      {/* Influence Diagnostics */}
      {twoWayResult && twoWayResult.influence_diagnostics && (
        <InfluenceDiagnostics influenceData={twoWayResult.influence_diagnostics} />
      )}

      {/* Diagnostic Plots */}
      {twoWayResult && twoWayResult.diagnostic_plots && (
        <DiagnosticPlots diagnosticPlots={twoWayResult.diagnostic_plots} />
      )}

      {/* Post-hoc Tests Section (Two-Way) */}
      {twoWayResult && (
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
          <div className="mb-6">
            <h3 className="text-2xl font-bold text-gray-100 mb-2">Post-hoc Multiple Comparisons</h3>
            <p className="text-gray-300 text-sm">
              Select the type of comparisons you want to perform, then choose a post-hoc test method.
            </p>
          </div>

          {/* Comparison Type Selection */}
          <div className="mb-6 bg-slate-700/30 rounded-lg p-4">
            <label className="block text-gray-200 font-medium mb-3">Comparison Type</label>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <button
                type="button"
                onClick={() => setTwoWayComparisonType('marginal_a')}
                className={`p-3 rounded-lg border-2 transition-all text-left ${
                  twoWayComparisonType === 'marginal_a'
                    ? 'border-purple-500 bg-purple-500/10'
                    : 'border-slate-600 bg-slate-700/50 hover:bg-slate-700'
                }`}
              >
                <h4 className="font-semibold text-gray-100">{factorA} Marginal Means</h4>
                <p className="text-xs text-gray-400">Compare main effects of {factorA}</p>
              </button>
              <button
                type="button"
                onClick={() => setTwoWayComparisonType('marginal_b')}
                className={`p-3 rounded-lg border-2 transition-all text-left ${
                  twoWayComparisonType === 'marginal_b'
                    ? 'border-purple-500 bg-purple-500/10'
                    : 'border-slate-600 bg-slate-700/50 hover:bg-slate-700'
                }`}
              >
                <h4 className="font-semibold text-gray-100">{factorB} Marginal Means</h4>
                <p className="text-xs text-gray-400">Compare main effects of {factorB}</p>
              </button>
              <button
                type="button"
                onClick={() => setTwoWayComparisonType('cell_means')}
                className={`p-3 rounded-lg border-2 transition-all text-left ${
                  twoWayComparisonType === 'cell_means'
                    ? 'border-purple-500 bg-purple-500/10'
                    : 'border-slate-600 bg-slate-700/50 hover:bg-slate-700'
                }`}
              >
                <h4 className="font-semibold text-gray-100">Cell Means</h4>
                <p className="text-xs text-gray-400">Compare all {factorA} × {factorB} combinations</p>
              </button>
              <button
                type="button"
                onClick={() => setTwoWayComparisonType('simple_a')}
                className={`p-3 rounded-lg border-2 transition-all text-left ${
                  twoWayComparisonType === 'simple_a'
                    ? 'border-purple-500 bg-purple-500/10'
                    : 'border-slate-600 bg-slate-700/50 hover:bg-slate-700'
                }`}
              >
                <h4 className="font-semibold text-gray-100">Simple Effects ({factorA})</h4>
                <p className="text-xs text-gray-400">Compare {factorA} at each level of {factorB}</p>
              </button>
            </div>
          </div>

          {/* Post-hoc Test Buttons */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <button
              onClick={() => handleTwoWayPostHoc('tukey')}
              disabled={twoWayPostHocLoading}
              className="bg-gradient-to-r from-blue-600 to-blue-700 text-white font-bold py-3 px-4 rounded-lg hover:from-blue-700 hover:to-blue-800 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Tukey HSD
            </button>
            <button
              onClick={() => handleTwoWayPostHoc('bonferroni')}
              disabled={twoWayPostHocLoading}
              className="bg-gradient-to-r from-purple-600 to-purple-700 text-white font-bold py-3 px-4 rounded-lg hover:from-purple-700 hover:to-purple-800 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Bonferroni
            </button>
            <button
              onClick={() => handleTwoWayPostHoc('scheffe')}
              disabled={twoWayPostHocLoading}
              className="bg-gradient-to-r from-indigo-600 to-indigo-700 text-white font-bold py-3 px-4 rounded-lg hover:from-indigo-700 hover:to-indigo-800 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Scheffe
            </button>
            <button
              onClick={() => handleTwoWayPostHoc('fisher-lsd')}
              disabled={twoWayPostHocLoading}
              className="bg-gradient-to-r from-teal-600 to-teal-700 text-white font-bold py-3 px-4 rounded-lg hover:from-teal-700 hover:to-teal-800 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Fisher's LSD
            </button>
          </div>

          {/* Loading State */}
          {twoWayPostHocLoading && (
            <div className="text-center py-4">
              <p className="text-gray-300">Running post-hoc analysis...</p>
            </div>
          )}

          {/* Error Display */}
          {twoWayPostHocError && (
            <div className="bg-red-900/30 backdrop-blur-lg rounded-xl p-4 border border-red-700/50 mb-6">
              <p className="text-red-200 font-medium">Error: {twoWayPostHocError}</p>
            </div>
          )}

          {/* Post-hoc Results */}
          {twoWayPostHocResult && (
            <div className="space-y-4">
              <div className="bg-gradient-to-r from-blue-900/30 to-purple-900/30 rounded-xl p-5 border border-blue-500/30">
                <h4 className="text-gray-100 font-bold text-lg mb-2">{twoWayPostHocResult.test_type}</h4>
                <p className="text-gray-300 text-sm">{twoWayPostHocResult.description}</p>
                <p className="text-gray-400 text-xs mt-2">Significance level (α): {twoWayPostHocResult.alpha}</p>
              </div>

              {/* Comparisons Table */}
              <div className="overflow-x-auto">
                <table className="w-full text-gray-100">
                  <thead>
                    <tr className="bg-slate-700/30 border-b border-slate-600">
                      <th className="text-left py-3 px-4">Group 1</th>
                      <th className="text-left py-3 px-4">Group 2</th>
                      <th className="text-right py-3 px-4">Mean Difference</th>
                      <th className="text-right py-3 px-4">Lower CI</th>
                      <th className="text-right py-3 px-4">Upper CI</th>
                      {twoWayPostHocResult.comparisons[0]?.p_adj !== undefined && (
                        <th className="text-right py-3 px-4">p-value (adj)</th>
                      )}
                      <th className="text-center py-3 px-4">Significant</th>
                    </tr>
                  </thead>
                  <tbody>
                    {twoWayPostHocResult.comparisons.map((comp, idx) => (
                      <tr key={idx} className={`border-b border-slate-700/30 ${comp.reject ? 'bg-green-900/20' : ''}`}>
                        <td className="py-3 px-4">{comp.group1}</td>
                        <td className="py-3 px-4">{comp.group2}</td>
                        <td className="text-right py-3 px-4 font-mono">{comp.mean_diff}</td>
                        <td className="text-right py-3 px-4 font-mono">{comp.lower_ci}</td>
                        <td className="text-right py-3 px-4 font-mono">{comp.upper_ci}</td>
                        {comp.p_adj !== undefined && (
                          <td className="text-right py-3 px-4 font-mono">{comp.p_adj}</td>
                        )}
                        <td className="text-center py-3 px-4">
                          {comp.reject ? (
                            <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-green-500/30 text-green-300">✓</span>
                          ) : (
                            <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-gray-500/30 text-gray-400">✗</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Summary */}
              <div className="bg-slate-700/30 rounded-lg p-4">
                <p className="text-gray-300 text-sm">
                  <strong>Summary:</strong> {twoWayPostHocResult.comparisons.filter(c => c.reject).length} out of {twoWayPostHocResult.comparisons.length} pairwise comparisons are statistically significant at α = {twoWayPostHocResult.alpha}.
                </p>
              </div>
            </div>
          )}
        </div>
      )}
      </>
      )}
    </div>
  )
}

export default ANOVA
