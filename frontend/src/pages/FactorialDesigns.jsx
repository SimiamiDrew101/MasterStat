import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import ResultCard from '../components/ResultCard'
import { Beaker, Plus, Trash2 } from 'lucide-react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Fractional factorial designs implementation
const FactorialDesigns = () => {
  const [designType, setDesignType] = useState('2k')
  const [factorNames, setFactorNames] = useState('A,B,C')
  const [responseName, setResponseName] = useState('Yield')
  const [alpha, setAlpha] = useState(0.05)
  const [numReplicates, setNumReplicates] = useState(1)
  const [fraction, setFraction] = useState('1/2')
  const [generators, setGenerators] = useState([])
  const [tableData, setTableData] = useState([])
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const isFirstRender = useRef(true)

  // Get factors array
  const factors = factorNames.split(',').map(f => f.trim()).filter(f => f.length > 0)
  const numCols = factors.length + 1 // factors + response
  const numFactors = factors.length

  // Update factor names when switching to fractional design (needs at least 4 factors)
  useEffect(() => {
    if (designType === 'fractional' && numFactors < 4 && numFactors > 0) {
      setFactorNames('A,B,C,D')
    }
  }, [designType])

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

  // Generate fractional factorial design runs
  const generateFractionalFactorialRuns = (numFactors, fraction, gens) => {
    if (numFactors < 3) return []

    // Determine number of base factors and generators
    const fractionMap = { '1/2': 1, '1/4': 2, '1/8': 3 }
    const p = fractionMap[fraction] || 1
    const baseFactors = numFactors - p

    // Generate base design (full factorial for base factors)
    const baseRuns = []
    const numBaseRuns = Math.pow(2, baseFactors)

    for (let i = 0; i < numBaseRuns; i++) {
      const run = []
      let temp = i

      for (let j = 0; j < baseFactors; j++) {
        run.push(temp % 2 === 0 ? 'Low' : 'High')
        temp = Math.floor(temp / 2)
      }

      baseRuns.push(run)
    }

    // Add generated columns based on generators
    const runs = baseRuns.map(baseRun => {
      const fullRun = [...baseRun]

      // Apply each generator
      gens.forEach(gen => {
        // Parse generator (e.g., "D=ABC" means D = A*B*C)
        if (gen.includes('=')) {
          const [, genFactors] = gen.split('=')

          // Calculate generated value
          let value = 0
          for (let char of genFactors.toUpperCase()) {
            const factorIdx = char.charCodeAt(0) - 65 // A=0, B=1, C=2, etc.
            if (factorIdx < baseRun.length) {
              const factorValue = baseRun[factorIdx] === 'Low' ? 0 : 1
              value ^= factorValue // XOR operation
            }
          }

          fullRun.push(value === 0 ? 'Low' : 'High')
        }
      })

      fullRun.push('') // Add empty response column
      return fullRun
    })

    return runs
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

  // Get default generators for common fractional designs
  const getDefaultGenerators = (k, frac) => {
    const designs = {
      '4-1/2': ['D=ABC'],
      '5-1/2': ['E=ABCD'],
      '5-1/4': ['D=AB', 'E=AC'],
      '6-1/2': ['F=ABCDE'],
      '6-1/4': ['E=ABC', 'F=BCD'],
      '7-1/4': ['F=ABC', 'G=BCD'],
    }
    const key = `${k}-${frac}`
    return designs[key] || []
  }

  // Update generators when factors or fraction change
  useEffect(() => {
    if (designType === 'fractional') {
      const defaultGens = getDefaultGenerators(numFactors, fraction)
      setGenerators(defaultGens)
    }
  }, [numFactors, fraction, designType])

  // Generate response values with realistic factor effects for fractional designs
  const generateFractionalResponses = (runs) => {
    return runs.map((run) => {
      // Base response value
      let response = 100

      // Add main effects based on factor levels (exclude last column which is response)
      run.slice(0, -1).forEach((level, idx) => {
        // Generate a random effect for each factor (between -10 and +10)
        const effectSize = (Math.random() * 10 + 5) * (idx % 2 === 0 ? 1 : -1)
        response += level === 'High' ? effectSize : -effectSize
      })

      // Add random noise
      response += randomNormal(0, 8)

      return (Math.round(response * 10) / 10).toFixed(1)
    })
  }

  // Regenerate table when number of factors, design type, or replicates change
  useEffect(() => {
    if (designType === 'fractional') {
      // Fractional factorial design
      if (numFactors >= 4 && generators.length > 0) {
        const newRuns = generateFractionalFactorialRuns(numFactors, fraction, generators)

        // Add sample response values with factor effects
        const responses = generateFractionalResponses(newRuns)
        newRuns.forEach((run, i) => {
          run[run.length - 1] = responses[i] || ''
        })

        setTableData(newRuns)
        setResult(null)
      } else if (numFactors > 0) {
        // Show message if insufficient factors
        setTableData([])
      } else {
        setTableData([])
      }
    } else {
      // Full factorial designs
      const levels = designType === '2k' ? 2 : 3
      const maxFactors = designType === '2k' ? 6 : 4
      const reps = designType === '2k' ? numReplicates : 1

      if (numFactors > 0 && numFactors <= maxFactors) {
        const newRuns = generateFactorialRuns(numFactors, levels, reps)

        // Add sample response values from normal distribution
        const sampleResponses = generateSampleResponses(numFactors, levels, reps)
        newRuns.forEach((run, i) => {
          run[run.length - 1] = sampleResponses[i] || ''
        })

        setTableData(newRuns)
        setResult(null)
        isFirstRender.current = false
      } else if (numFactors === 0) {
        setTableData([])
      }
    }
  }, [numFactors, designType, numReplicates, fraction, generators])

  const handleCellChange = (rowIndex, colIndex, value) => {
    const newData = [...tableData]
    newData[rowIndex][colIndex] = value
    setTableData(newData)
  }

  // Excel-like keyboard navigation
  const handleKeyDown = (e, rowIndex, colIndex) => {
    const maxRow = tableData.length - 1
    const maxCol = numCols - 1

    switch (e.key) {
      case 'ArrowUp':
        e.preventDefault()
        if (rowIndex > 0) {
          document.getElementById(`cell-${rowIndex - 1}-${colIndex}`)?.focus()
        }
        break
      case 'ArrowDown':
        e.preventDefault()
        if (rowIndex < maxRow) {
          document.getElementById(`cell-${rowIndex + 1}-${colIndex}`)?.focus()
        }
        break
      case 'ArrowLeft':
        if (e.target.selectionStart === 0 && colIndex > 0) {
          e.preventDefault()
          document.getElementById(`cell-${rowIndex}-${colIndex - 1}`)?.focus()
        }
        break
      case 'ArrowRight':
        if (e.target.selectionStart === e.target.value.length && colIndex < maxCol) {
          e.preventDefault()
          document.getElementById(`cell-${rowIndex}-${colIndex + 1}`)?.focus()
        }
        break
      case 'Enter':
        e.preventDefault()
        if (rowIndex < maxRow) {
          document.getElementById(`cell-${rowIndex + 1}-${colIndex}`)?.focus()
        } else {
          // If on last row, add a new row and focus on it
          addRow()
          setTimeout(() => {
            document.getElementById(`cell-${rowIndex + 1}-${colIndex}`)?.focus()
          }, 50)
        }
        break
      case 'Tab':
        // Let default tab behavior work but focus next cell
        if (!e.shiftKey && colIndex < maxCol) {
          e.preventDefault()
          document.getElementById(`cell-${rowIndex}-${colIndex + 1}`)?.focus()
        } else if (e.shiftKey && colIndex > 0) {
          e.preventDefault()
          document.getElementById(`cell-${rowIndex}-${colIndex - 1}`)?.focus()
        } else if (!e.shiftKey && colIndex === maxCol && rowIndex < maxRow) {
          e.preventDefault()
          document.getElementById(`cell-${rowIndex + 1}-0`)?.focus()
        }
        break
      default:
        break
    }
  }

  // Handle cell click to select all text
  const handleCellClick = (e) => {
    e.target.select()
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

      let payload, endpoint

      if (designType === 'fractional') {
        // Fractional factorial endpoint
        payload = {
          data: data,
          factors: factors,
          response: responseName,
          alpha,
          generators: generators,
          fraction: fraction
        }
        endpoint = `${API_URL}/api/factorial/fractional-factorial/analyze`
      } else {
        // Full factorial endpoints
        payload = {
          data: data,
          factors: factors,
          response: responseName,
          alpha
        }
        endpoint = designType === '2k'
          ? `${API_URL}/api/factorial/full-factorial`
          : `${API_URL}/api/factorial/three-level-factorial`
      }

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
              <option value="fractional">2^(k-p) Fractional Factorial</option>
            </select>
            <p className="text-gray-400 text-xs mt-1">
              2^k for screening/main effects • 3^k for optimization/curvature detection • Fractional for efficient screening
            </p>
          </div>

          {/* Fraction Selection (fractional only) */}
          {designType === 'fractional' && (
            <div>
              <label className="block text-gray-100 font-medium mb-2">
                Fraction Size
              </label>
              <select
                value={fraction}
                onChange={(e) => setFraction(e.target.value)}
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-purple-500"
              >
                <option value="1/2">1/2 Fraction (Half the runs)</option>
                <option value="1/4">1/4 Fraction (Quarter the runs)</option>
                <option value="1/8">1/8 Fraction (Eighth the runs)</option>
              </select>
              <p className="text-gray-400 text-xs mt-1">
                Smaller fractions = fewer runs but more confounding. Requires at least 4 factors.
              </p>
            </div>
          )}

          {/* Generator Display (fractional only) */}
          {designType === 'fractional' && (
            <>
              {generators.length > 0 ? (
                <div className="bg-purple-900/20 rounded-lg p-4 border border-purple-700/30">
                  <label className="block text-gray-100 font-medium mb-2">
                    Generators (Defining Relations)
                  </label>
                  <div className="flex flex-wrap gap-2">
                    {generators.map((gen, idx) => (
                      <span
                        key={idx}
                        className="px-3 py-1 bg-purple-700/50 text-purple-100 rounded-full text-sm font-mono"
                      >
                        {gen}
                      </span>
                    ))}
                  </div>
                  <p className="text-gray-400 text-xs mt-2">
                    These generators define which effects are confounded. Default generators for common designs are shown.
                  </p>
                </div>
              ) : numFactors > 0 && numFactors < 4 ? (
                <div className="bg-orange-900/20 rounded-lg p-4 border border-orange-700/30">
                  <p className="text-orange-300 text-sm">
                    ⚠️ Fractional factorial designs require at least 4 factors. Currently you have {numFactors} factor{numFactors !== 1 ? 's' : ''}.
                  </p>
                  <p className="text-orange-200/70 text-xs mt-2">
                    Add more factors (e.g., "A,B,C,D") to generate a fractional design.
                  </p>
                </div>
              ) : null}
            </>
          )}

          {/* Design Resolution Table (fractional only) */}
          {designType === 'fractional' && (
            <div className="bg-gradient-to-r from-purple-900/20 to-indigo-900/20 rounded-lg p-5 border border-purple-700/30">
              <h4 className="text-gray-100 font-bold text-xl mb-4 text-center">
                Available Factorial Designs (with Resolution)
              </h4>

              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="bg-slate-700/30">
                      <th className="border border-slate-600 py-2 px-3 text-gray-100 font-bold"></th>
                      <th colSpan="14" className="border border-slate-600 py-2 px-4 text-gray-100 font-bold text-lg">
                        Factors
                      </th>
                    </tr>
                    <tr className="bg-slate-700/30">
                      <th className="border border-slate-600 py-2 px-3 text-gray-100 font-bold">Run</th>
                      {[2,3,4,5,6,7,8,9,10,11,12,13,14,15].map(n => (
                        <th key={n} className="border border-slate-600 py-2 px-3 text-gray-100 font-bold">{n}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {/* 4 runs */}
                    <tr>
                      <td className="border border-slate-600 py-2 px-3 text-gray-100 font-bold bg-slate-700/20">4</td>
                      <td className="border border-slate-600 py-2 px-3 bg-green-600/70 text-gray-900 font-bold text-center">Full</td>
                      <td className="border border-slate-600 py-2 px-3 bg-red-600/70 text-gray-100 font-bold text-center">III</td>
                      {[...Array(12)].map((_, i) => <td key={i} className="border border-slate-600 py-2 px-3 bg-slate-800/50"></td>)}
                    </tr>
                    {/* 8 runs */}
                    <tr>
                      <td className="border border-slate-600 py-2 px-3 text-gray-100 font-bold bg-slate-700/20">8</td>
                      <td className="border border-slate-600 py-2 px-3 bg-slate-800/50"></td>
                      <td className="border border-slate-600 py-2 px-3 bg-green-600/70 text-gray-900 font-bold text-center">Full</td>
                      <td className="border border-slate-600 py-2 px-3 bg-yellow-500/70 text-gray-900 font-bold text-center">IV</td>
                      <td className="border border-slate-600 py-2 px-3 bg-red-600/70 text-gray-100 font-bold text-center">III</td>
                      <td className="border border-slate-600 py-2 px-3 bg-red-600/70 text-gray-100 font-bold text-center">III</td>
                      <td className="border border-slate-600 py-2 px-3 bg-red-600/70 text-gray-100 font-bold text-center">III</td>
                      {[...Array(8)].map((_, i) => <td key={i} className="border border-slate-600 py-2 px-3 bg-slate-800/50"></td>)}
                    </tr>
                    {/* 16 runs */}
                    <tr>
                      <td className="border border-slate-600 py-2 px-3 text-gray-100 font-bold bg-slate-700/20">16</td>
                      <td colSpan="2" className="border border-slate-600 py-2 px-3 bg-slate-800/50"></td>
                      <td className="border border-slate-600 py-2 px-3 bg-green-600/70 text-gray-900 font-bold text-center">Full</td>
                      <td className="border border-slate-600 py-2 px-3 bg-green-500/70 text-gray-900 font-bold text-center">V</td>
                      <td className="border border-slate-600 py-2 px-3 bg-yellow-500/70 text-gray-900 font-bold text-center">IV</td>
                      <td className="border border-slate-600 py-2 px-3 bg-yellow-500/70 text-gray-900 font-bold text-center">IV</td>
                      <td className="border border-slate-600 py-2 px-3 bg-yellow-500/70 text-gray-900 font-bold text-center">IV</td>
                      <td className="border border-slate-600 py-2 px-3 bg-red-600/70 text-gray-100 font-bold text-center">III</td>
                      <td className="border border-slate-600 py-2 px-3 bg-red-600/70 text-gray-100 font-bold text-center">III</td>
                      <td className="border border-slate-600 py-2 px-3 bg-red-600/70 text-gray-100 font-bold text-center">III</td>
                      <td className="border border-slate-600 py-2 px-3 bg-red-600/70 text-gray-100 font-bold text-center">III</td>
                      <td className="border border-slate-600 py-2 px-3 bg-red-600/70 text-gray-100 font-bold text-center">III</td>
                      <td className="border border-slate-600 py-2 px-3 bg-red-600/70 text-gray-100 font-bold text-center">III</td>
                      <td className="border border-slate-600 py-2 px-3 bg-red-600/70 text-gray-100 font-bold text-center">III</td>
                    </tr>
                    {/* 32 runs */}
                    <tr>
                      <td className="border border-slate-600 py-2 px-3 text-gray-100 font-bold bg-slate-700/20">32</td>
                      <td colSpan="3" className="border border-slate-600 py-2 px-3 bg-slate-800/50"></td>
                      <td className="border border-slate-600 py-2 px-3 bg-green-600/70 text-gray-900 font-bold text-center">Full</td>
                      <td className="border border-slate-600 py-2 px-3 bg-green-500/70 text-gray-900 font-bold text-center">VI</td>
                      <td className="border border-slate-600 py-2 px-3 bg-yellow-500/70 text-gray-900 font-bold text-center">IV</td>
                      <td className="border border-slate-600 py-2 px-3 bg-yellow-500/70 text-gray-900 font-bold text-center">IV</td>
                      <td className="border border-slate-600 py-2 px-3 bg-yellow-500/70 text-gray-900 font-bold text-center">IV</td>
                      <td className="border border-slate-600 py-2 px-3 bg-yellow-500/70 text-gray-900 font-bold text-center">IV</td>
                      <td className="border border-slate-600 py-2 px-3 bg-yellow-500/70 text-gray-900 font-bold text-center">IV</td>
                      <td className="border border-slate-600 py-2 px-3 bg-yellow-500/70 text-gray-900 font-bold text-center">IV</td>
                      <td className="border border-slate-600 py-2 px-3 bg-yellow-500/70 text-gray-900 font-bold text-center">IV</td>
                      <td className="border border-slate-600 py-2 px-3 bg-yellow-500/70 text-gray-900 font-bold text-center">IV</td>
                      <td className="border border-slate-600 py-2 px-3 bg-yellow-500/70 text-gray-900 font-bold text-center">IV</td>
                    </tr>
                    {/* 64 runs */}
                    <tr>
                      <td className="border border-slate-600 py-2 px-3 text-gray-100 font-bold bg-slate-700/20">64</td>
                      <td colSpan="4" className="border border-slate-600 py-2 px-3 bg-slate-800/50"></td>
                      <td className="border border-slate-600 py-2 px-3 bg-green-600/70 text-gray-900 font-bold text-center">Full</td>
                      <td className="border border-slate-600 py-2 px-3 bg-green-500/70 text-gray-900 font-bold text-center">VII</td>
                      <td className="border border-slate-600 py-2 px-3 bg-green-500/70 text-gray-900 font-bold text-center">V</td>
                      <td className="border border-slate-600 py-2 px-3 bg-yellow-500/70 text-gray-900 font-bold text-center">IV</td>
                      <td className="border border-slate-600 py-2 px-3 bg-yellow-500/70 text-gray-900 font-bold text-center">IV</td>
                      <td className="border border-slate-600 py-2 px-3 bg-yellow-500/70 text-gray-900 font-bold text-center">IV</td>
                      <td className="border border-slate-600 py-2 px-3 bg-yellow-500/70 text-gray-900 font-bold text-center">IV</td>
                      <td className="border border-slate-600 py-2 px-3 bg-yellow-500/70 text-gray-900 font-bold text-center">IV</td>
                      <td className="border border-slate-600 py-2 px-3 bg-yellow-500/70 text-gray-900 font-bold text-center">IV</td>
                      <td className="border border-slate-600 py-2 px-3 bg-yellow-500/70 text-gray-900 font-bold text-center">IV</td>
                    </tr>
                    {/* 128 runs */}
                    <tr>
                      <td className="border border-slate-600 py-2 px-3 text-gray-100 font-bold bg-slate-700/20">128</td>
                      <td colSpan="5" className="border border-slate-600 py-2 px-3 bg-slate-800/50"></td>
                      <td className="border border-slate-600 py-2 px-3 bg-green-600/70 text-gray-900 font-bold text-center">Full</td>
                      <td className="border border-slate-600 py-2 px-3 bg-green-500/70 text-gray-900 font-bold text-center">VIII</td>
                      <td className="border border-slate-600 py-2 px-3 bg-green-500/70 text-gray-900 font-bold text-center">VI</td>
                      <td className="border border-slate-600 py-2 px-3 bg-green-500/70 text-gray-900 font-bold text-center">V</td>
                      <td className="border border-slate-600 py-2 px-3 bg-green-500/70 text-gray-900 font-bold text-center">V</td>
                      <td className="border border-slate-600 py-2 px-3 bg-yellow-500/70 text-gray-900 font-bold text-center">IV</td>
                      <td className="border border-slate-600 py-2 px-3 bg-yellow-500/70 text-gray-900 font-bold text-center">IV</td>
                      <td className="border border-slate-600 py-2 px-3 bg-yellow-500/70 text-gray-900 font-bold text-center">IV</td>
                      <td className="border border-slate-600 py-2 px-3 bg-yellow-500/70 text-gray-900 font-bold text-center">IV</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div className="mt-6">
                <h4 className="text-gray-100 font-bold text-lg mb-3 text-center">
                  Available Resolution III Plackett-Burman Designs
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-gray-100">
                  <div>
                    <p className="mb-1"><span className="font-bold">Factors 2-7:</span> 12, 20, 24, 28,..., 48</p>
                    <p className="mb-1"><span className="font-bold">Factors 8-11:</span> 12, 20, 24, 28,..., 48</p>
                    <p className="mb-1"><span className="font-bold">Factors 12-15:</span> 20, 24, 28, 36,..., 48</p>
                    <p className="mb-1"><span className="font-bold">Factors 16-19:</span> 20, 24, 28, 32,..., 48</p>
                  </div>
                  <div>
                    <p className="mb-1"><span className="font-bold">Factors 20-23:</span> 24, 28, 32, 36,..., 48</p>
                    <p className="mb-1"><span className="font-bold">Factors 24-27:</span> 28, 32, 36, 40, 44, 48</p>
                    <p className="mb-1"><span className="font-bold">Factors 28-31:</span> 32, 36, 40, 44, 48</p>
                    <p className="mb-1"><span className="font-bold">Factors 32-35:</span> 36, 40, 44, 48</p>
                  </div>
                  <div>
                    <p className="mb-1"><span className="font-bold">Factors 36-39:</span> 40, 44, 48</p>
                    <p className="mb-1"><span className="font-bold">Factors 40-43:</span> 44, 48</p>
                    <p className="mb-1"><span className="font-bold">Factors 44-47:</span> 48</p>
                  </div>
                </div>
              </div>

              <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
                <div className="bg-green-900/20 rounded-lg p-3 border border-green-700/30">
                  <p className="font-semibold text-green-200 mb-1">Full / Resolution V+</p>
                  <p className="text-gray-300">Main effects and 2-way interactions are clear. Ideal for detailed analysis.</p>
                </div>
                <div className="bg-yellow-900/20 rounded-lg p-3 border border-yellow-700/30">
                  <p className="font-semibold text-yellow-200 mb-1">Resolution IV</p>
                  <p className="text-gray-300">Main effects clear, but 2-way interactions confounded with each other. Good for screening.</p>
                </div>
                <div className="bg-red-900/20 rounded-lg p-3 border border-red-700/30">
                  <p className="font-semibold text-red-200 mb-1">Resolution III</p>
                  <p className="text-gray-300">Main effects confounded with 2-way interactions. Use only when interactions are negligible.</p>
                </div>
              </div>
            </div>
          )}

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

            <div className="overflow-x-auto bg-slate-700/30 rounded-lg border-2 border-slate-600">
              <table className="w-full border-collapse">
                <thead>
                  <tr className="bg-slate-700/70">
                    <th className="px-3 py-2 text-center text-gray-100 font-semibold text-sm border-b-2 border-r border-slate-600 w-14 sticky left-0 bg-slate-700/70">
                      #
                    </th>
                    {factors.map((factor, idx) => (
                      <th
                        key={idx}
                        className="px-3 py-2 text-center text-gray-100 font-semibold text-sm border-b-2 border-r border-slate-600 min-w-[100px]"
                      >
                        {factor}
                      </th>
                    ))}
                    <th className="px-3 py-2 text-center text-gray-100 font-semibold text-sm border-b-2 border-r border-slate-600 min-w-[100px] bg-purple-900/20">
                      {responseName}
                    </th>
                    <th className="px-3 py-2 text-center text-gray-100 font-semibold text-sm border-b-2 border-slate-600 w-16">

                    </th>
                  </tr>
                </thead>
                <tbody>
                  {tableData.map((row, rowIndex) => (
                    <tr
                      key={rowIndex}
                      className="border-b border-slate-700/30 hover:bg-slate-600/10"
                    >
                      <td className="px-3 py-2 text-center text-gray-300 text-sm font-medium bg-slate-700/30 border-r border-slate-600 sticky left-0">
                        {rowIndex + 1}
                      </td>
                      {row.map((cell, colIndex) => (
                        <td key={colIndex} className="px-1 py-1 border-r border-slate-700/20">
                          <input
                            id={`cell-${rowIndex}-${colIndex}`}
                            type="text"
                            value={cell}
                            onChange={(e) => handleCellChange(rowIndex, colIndex, e.target.value)}
                            onKeyDown={(e) => handleKeyDown(e, rowIndex, colIndex)}
                            onClick={handleCellClick}
                            className="w-full px-2 py-1.5 bg-slate-800/50 text-gray-100 border border-slate-600/50 focus:border-purple-500 focus:bg-slate-700/50 hover:border-slate-500 rounded-sm focus:outline-none focus:ring-2 focus:ring-purple-500/50 text-sm transition-all"
                            placeholder={colIndex === row.length - 1 ? '0.0' : 'Low/High'}
                            autoComplete="off"
                          />
                        </td>
                      ))}
                      <td className="px-2 py-2 text-center">
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

            <div className="space-y-1 mt-2">
              <p className="text-gray-400 text-xs">
                {designType === '2k'
                  ? 'Table automatically generates 2^k runs (max 6 factors = 64 runs). Use "Low"/"High", "-1"/"1", or "0"/"1" for levels.'
                  : designType === '3k'
                  ? 'Table automatically generates 3^k runs (max 4 factors = 81 runs). Use "Low"/"Medium"/"High" or "-1"/"0"/"1" for levels.'
                  : `Table automatically generates 2^(k-p) fractional design runs using the specified generators. Use "Low"/"High" for levels.`}
                {' '}Response values must be numbers. {tableData.length > 0 && `Current: ${tableData.length} runs.`}
              </p>
              <p className="text-purple-400 text-xs">
                <strong>Excel-like navigation:</strong> Use Arrow keys to move, Enter to go down, Tab to move right, Click to select all.
              </p>
            </div>
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
