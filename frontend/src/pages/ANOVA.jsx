import { useState, useEffect } from 'react'
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

  // Post-hoc test state
  const [postHocResult, setPostHocResult] = useState(null)
  const [postHocLoading, setPostHocLoading] = useState(false)
  const [postHocError, setPostHocError] = useState(null)

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

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)
    setPostHocResult(null) // Clear previous post-hoc results

    try {
      if (analysisType === 'one-way') {
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
        setResult(response.data)
      } else if (analysisType === 'two-way') {
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
        setResult(response.data)
      }
    } catch (err) {
      console.error('ANOVA error:', err)
      const errorMessage = err.response?.data?.detail || err.message || 'An error occurred during analysis'
      setError(errorMessage)
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  const handlePostHoc = async (method) => {
    setPostHocLoading(true)
    setPostHocError(null)
    setPostHocResult(null)

    try {
      let payload

      if (analysisType === 'one-way') {
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

        payload = {
          groups: groupsData,
          alpha
        }
      } else {
        setPostHocError('Post-hoc tests are currently only available for One-Way ANOVA')
        setPostHocLoading(false)
        return
      }

      const response = await axios.post(`${API_URL}/api/anova/post-hoc/${method}`, payload)
      setPostHocResult(response.data)
    } catch (err) {
      setPostHocError(err.response?.data?.detail || err.message || 'An error occurred')
    } finally {
      setPostHocLoading(false)
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

      {/* Post-hoc Tests Section */}
      {result && analysisType === 'one-way' && result.reject_null && (
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
              onClick={() => handlePostHoc('tukey')}
              disabled={postHocLoading}
              className="bg-gradient-to-r from-blue-600 to-blue-700 text-white font-bold py-3 px-4 rounded-lg hover:from-blue-700 hover:to-blue-800 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Tukey HSD
            </button>
            <button
              onClick={() => handlePostHoc('bonferroni')}
              disabled={postHocLoading}
              className="bg-gradient-to-r from-purple-600 to-purple-700 text-white font-bold py-3 px-4 rounded-lg hover:from-purple-700 hover:to-purple-800 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Bonferroni
            </button>
            <button
              onClick={() => handlePostHoc('scheffe')}
              disabled={postHocLoading}
              className="bg-gradient-to-r from-indigo-600 to-indigo-700 text-white font-bold py-3 px-4 rounded-lg hover:from-indigo-700 hover:to-indigo-800 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Scheffe
            </button>
            <button
              onClick={() => handlePostHoc('fisher-lsd')}
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
    </div>
  )
}

export default ANOVA
