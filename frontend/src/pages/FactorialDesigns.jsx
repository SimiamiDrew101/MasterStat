import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import ResultCard from '../components/ResultCard'
import CubePlot from '../components/CubePlot'
import HalfNormalPlot from '../components/HalfNormalPlot'
import FactorialInteractionPlots from '../components/FactorialInteractionPlots'
import AliasStructureGraph from '../components/AliasStructureGraph'
import DiagnosticPlots from '../components/DiagnosticPlots'
import Histogram from '../components/Histogram'
import CorrelationHeatmap from '../components/CorrelationHeatmap'
import ScatterMatrix from '../components/ScatterMatrix'
import { Beaker, Plus, Trash2, Download, Copy, FileJson, FileDown } from 'lucide-react'
import { exportToCSVWithMetadata, copyToClipboard, exportResultsToJSON } from '../utils/exportDesign'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Fractional factorial designs implementation
const FactorialDesigns = () => {
  // Tab state - unified design language
  const [activeTab, setActiveTab] = useState('2k')

  // Common state across all tabs
  const [factorNames, setFactorNames] = useState('A,B,C')
  const [responseName, setResponseName] = useState('Yield')
  const [alpha, setAlpha] = useState(0.05)
  const [tableData, setTableData] = useState([])
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const isFirstRender = useRef(true)

  // 2^k specific state
  const [numReplicates, setNumReplicates] = useState(1)

  // Fractional factorial specific state
  const [fraction, setFraction] = useState('1/2')
  const [generators, setGenerators] = useState([])

  // Plackett-Burman specific state
  const [pbNumRuns, setPbNumRuns] = useState(12)

  // Foldover design state
  const [showFoldover, setShowFoldover] = useState(false)
  const [foldoverType, setFoldoverType] = useState('full')
  const [foldoverFactor, setFoldoverFactor] = useState('')
  const [foldoverData, setFoldoverData] = useState([])
  const [foldoverTableData, setFoldoverTableData] = useState([])
  const [combinedResult, setCombinedResult] = useState(null)
  const [foldoverLoading, setFoldoverLoading] = useState(false)

  // Get factors array
  const factors = factorNames.split(',').map(f => f.trim()).filter(f => f.length > 0)
  const numCols = factors.length + 1 // factors + response
  const numFactors = factors.length

  // Update factor names when switching to fractional design or changing fraction
  useEffect(() => {
    if (activeTab === 'fractional') {
      // Determine minimum factors needed for the selected fraction
      const minFactorsNeeded = {
        '1/2': 4,
        '1/4': 4,
        '1/8': 5,
      }[fraction] || 4

      // Auto-expand factor names if we don't have enough
      if (numFactors < minFactorsNeeded) {
        const factorLetters = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        setFactorNames(factorLetters.slice(0, minFactorsNeeded).join(','))
      }
    }
  }, [activeTab, fraction])

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
  const generateFractionalFactorialRuns = (numFactors, fraction, gens, replicates = 1) => {
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
    const runsWithoutReplicates = baseRuns.map(baseRun => {
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

      return fullRun
    })

    // Add replication
    const runs = []
    for (const run of runsWithoutReplicates) {
      for (let rep = 0; rep < replicates; rep++) {
        runs.push([...run, '']) // Add copy of run with empty response column
      }
    }

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
      // Half-fraction designs (1/2)
      '4-1/2': ['D=ABC'],
      '5-1/2': ['E=ABCD'],
      '6-1/2': ['F=ABCDE'],
      '7-1/2': ['G=ABCDEF'],

      // Quarter-fraction designs (1/4)
      '4-1/4': ['C=AB', 'D=AC'],  // 2^(4-2)
      '5-1/4': ['D=AB', 'E=AC'],   // 2^(5-2)
      '6-1/4': ['E=ABC', 'F=BCD'], // 2^(6-2)
      '7-1/4': ['F=ABCD', 'G=ABCE'], // 2^(7-2)

      // One-eighth fraction designs (1/8)
      '5-1/8': ['C=AB', 'D=AC', 'E=BC'],     // 2^(5-3)
      '6-1/8': ['D=AB', 'E=AC', 'F=BC'],     // 2^(6-3)
      '7-1/8': ['E=ABC', 'F=ABD', 'G=ACD'],  // 2^(7-3)
    }
    const key = `${k}-${frac}`
    return designs[key] || []
  }

  // Update generators when factors or fraction change
  useEffect(() => {
    if (activeTab === 'fractional') {
      const defaultGens = getDefaultGenerators(numFactors, fraction)
      setGenerators(defaultGens)
    }
  }, [numFactors, fraction, activeTab])

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
    if (activeTab === 'pb') {
      // Plackett-Burman design
      if (numFactors > 0 && numFactors < pbNumRuns) {
        const generatePBDesign = async () => {
          try {
            const response = await axios.post(`${API_URL}/api/factorial/plackett-burman/generate`, {
              n_factors: numFactors,
              n_runs: pbNumRuns
            })

            // Convert design matrix to table format
            const designMatrix = response.data.design_matrix
            const newRuns = designMatrix.map(row => {
              const runArray = factors.map(factor => row[factor])
              runArray.push('') // Add empty response column
              return runArray
            })

            // Add sample responses
            const sampleResponses = generateSampleResponses(numFactors, 2, 1)
            newRuns.forEach((run, i) => {
              run[run.length - 1] = sampleResponses[i] || ''
            })

            setTableData(newRuns)
            setResult(null)
          } catch (err) {
            console.error('Error generating Plackett-Burman design:', err)
            setError(err.response?.data?.detail || err.message)
          }
        }
        generatePBDesign()
      } else if (numFactors >= pbNumRuns) {
        setTableData([])
      }
    } else if (activeTab === 'fractional') {
      // Fractional factorial design
      if (numFactors >= 4 && generators.length > 0) {
        const newRuns = generateFractionalFactorialRuns(numFactors, fraction, generators, numReplicates)

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
      // Full factorial designs (2k or 3k)
      const levels = activeTab === '2k' ? 2 : 3
      const maxFactors = activeTab === '2k' ? 6 : 4
      const reps = activeTab === '2k' ? numReplicates : 1

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
  }, [numFactors, activeTab, numReplicates, fraction, generators, pbNumRuns])

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

  const handleGenerateFoldover = async () => {
    setFoldoverLoading(true)
    setError(null)

    try {
      // Convert original table data to API format
      const validRows = tableData.filter(row =>
        row.some(cell => cell !== null && cell !== undefined && cell.toString().trim() !== '')
      )

      const originalData = validRows.map((row) => {
        const rowData = {}
        factors.forEach((factor, i) => {
          rowData[factor] = row[i].toString().trim()
        })
        const responseValue = parseFloat(row[row.length - 1])
        rowData[responseName] = responseValue
        return rowData
      })

      // Call foldover generation endpoint
      const response = await axios.post(`${API_URL}/api/factorial/foldover/generate`, {
        data: originalData,
        factors: factors,
        foldover_type: foldoverType,
        foldover_factor: foldoverType === 'partial' ? foldoverFactor : null,
        generators: generators.length > 0 ? generators : null
      })

      setFoldoverData(response.data)

      // Convert foldover data to table format with empty responses
      const foldoverRuns = response.data.foldover_data.map(run => {
        const row = factors.map(factor => run[factor] || '')
        // Generate sample response value for foldover runs
        const sampleResponse = (Math.round(randomNormal(65, 15) * 10) / 10).toFixed(1)
        row.push(sampleResponse) // Add sample response
        return row
      })

      setFoldoverTableData(foldoverRuns)
      setShowFoldover(true)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to generate foldover')
    } finally {
      setFoldoverLoading(false)
    }
  }

  const handleAnalyzeCombined = async () => {
    setFoldoverLoading(true)
    setError(null)
    setCombinedResult(null)

    try {
      // Convert original table data to API format
      const validOriginalRows = tableData.filter(row =>
        row.some(cell => cell !== null && cell !== undefined && cell.toString().trim() !== '')
      )

      const originalData = validOriginalRows.map((row) => {
        const rowData = {}
        factors.forEach((factor, i) => {
          rowData[factor] = row[i].toString().trim()
        })
        const responseValue = parseFloat(row[row.length - 1])
        rowData[responseName] = responseValue
        return rowData
      })

      // Convert foldover table data to API format
      const validFoldoverRows = foldoverTableData.filter(row =>
        row.some(cell => cell !== null && cell !== undefined && cell.toString().trim() !== '')
      )

      const foldoverResponseData = validFoldoverRows.map((row) => {
        const rowData = {}
        factors.forEach((factor, i) => {
          rowData[factor] = row[i].toString().trim()
        })
        const responseValue = parseFloat(row[row.length - 1])
        if (isNaN(responseValue)) {
          throw new Error('All foldover runs must have response values')
        }
        rowData[responseName] = responseValue
        return rowData
      })

      // Call combined analysis endpoint
      const response = await axios.post(`${API_URL}/api/factorial/foldover/analyze`, {
        original_data: originalData,
        foldover_data: foldoverResponseData,
        factors: factors,
        response: responseName,
        alpha: alpha,
        generators: generators.length > 0 ? generators : null,
        foldover_type: foldoverType,
        foldover_factor: foldoverType === 'partial' ? foldoverFactor : null
      })

      setCombinedResult(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to analyze combined design')
    } finally {
      setFoldoverLoading(false)
    }
  }

  const handleFoldoverCellChange = (rowIndex, colIndex, value) => {
    const newData = [...foldoverTableData]
    newData[rowIndex][colIndex] = value
    setFoldoverTableData(newData)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)
    setShowFoldover(false)
    setCombinedResult(null)

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

      if (activeTab === 'fractional') {
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
      } else if (activeTab === 'pb') {
        // Plackett-Burman endpoint
        payload = {
          data: data,
          factors: factors,
          response: responseName,
          alpha
        }
        endpoint = `${API_URL}/api/factorial/plackett-burman/analyze`
      } else {
        // Full factorial endpoints
        payload = {
          data: data,
          factors: factors,
          response: responseName,
          alpha
        }
        endpoint = activeTab === '2k'
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

  const handleExportPDF = async () => {
    try {
      const payload = {
        results: result,
        design_type: activeTab === '2k' ? 'full_factorial' : activeTab === 'fractional' ? 'fractional_factorial' : activeTab === 'pb' ? 'plackett_burman' : 'three_level_factorial',
        title: `${activeTab === '2k' ? 'Full Factorial' : activeTab === 'fractional' ? 'Fractional Factorial' : activeTab === 'pb' ? 'Plackett-Burman' : 'Three-Level Factorial'} Design Analysis Report`
      }

      const response = await axios.post(`${API_URL}/api/factorial/export-pdf`, payload, {
        responseType: 'blob'
      })

      // Create download link
      const url = window.URL.createObjectURL(new Blob([response.data]))
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', `factorial_${activeTab}_report_${new Date().toISOString().slice(0,10)}.pdf`)
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.URL.revokeObjectURL(url)
    } catch (err) {
      console.error('Error exporting PDF:', err)
      alert('Failed to export PDF: ' + (err.response?.data?.detail || err.message))
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <div className="flex items-center space-x-3 mb-4">
          <Beaker className="w-8 h-8 text-purple-400" />
          <h2 className="text-3xl font-bold text-gray-100">Factorial Designs</h2>
        </div>
        <p className="text-gray-300">
          Powerful experimental designs for identifying important factors and interactions. Choose from full factorial, fractional factorial, or screening designs based on your objectives.
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl border border-slate-700/50 overflow-hidden">
        <div className="grid grid-cols-2 md:grid-cols-4">
          <button
            onClick={() => {
              setActiveTab('2k')
              setResult(null)
              setError(null)
            }}
            className={`px-6 py-4 font-semibold text-sm transition-all ${
              activeTab === '2k'
                ? 'bg-cyan-500/20 text-cyan-400 border-b-2 border-cyan-500'
                : 'text-gray-400 hover:text-gray-300 hover:bg-slate-700/30'
            }`}
          >
            2^k Full Factorial
          </button>
          <button
            onClick={() => {
              setActiveTab('3k')
              setResult(null)
              setError(null)
            }}
            className={`px-6 py-4 font-semibold text-sm transition-all ${
              activeTab === '3k'
                ? 'bg-purple-500/20 text-purple-400 border-b-2 border-purple-500'
                : 'text-gray-400 hover:text-gray-300 hover:bg-slate-700/30'
            }`}
          >
            3^k Full Factorial
          </button>
          <button
            onClick={() => {
              setActiveTab('fractional')
              setResult(null)
              setError(null)
            }}
            className={`px-6 py-4 font-semibold text-sm transition-all ${
              activeTab === 'fractional'
                ? 'bg-green-500/20 text-green-400 border-b-2 border-green-500'
                : 'text-gray-400 hover:text-gray-300 hover:bg-slate-700/30'
            }`}
          >
            Fractional Factorial
          </button>
          <button
            onClick={() => {
              setActiveTab('pb')
              setResult(null)
              setError(null)
            }}
            className={`px-6 py-4 font-semibold text-sm transition-all ${
              activeTab === 'pb'
                ? 'bg-amber-500/20 text-amber-400 border-b-2 border-amber-500'
                : 'text-gray-400 hover:text-gray-300 hover:bg-slate-700/30'
            }`}
          >
            Plackett-Burman
          </button>
        </div>
      </div>

      {/* 2^k Full Factorial Tab */}
      {activeTab === '2k' && (
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* 2k-specific: Number of Replicates */}
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
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
              />
              <p className="text-gray-400 text-xs mt-1">
                Replicates allow estimation of pure error for lack-of-fit testing. Each treatment combination will be repeated {numReplicates} time{numReplicates > 1 ? 's' : ''}.
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
                Table automatically generates 2^k runs (max 6 factors = 64 runs). Use "Low"/"High", "-1"/"1", or "0"/"1" for levels.
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

          {/* Export Design Section */}
          {tableData.length > 0 && (
            <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-600">
              <h4 className="text-gray-100 font-semibold mb-3 text-sm">Export Design</h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <button
                  type="button"
                  onClick={() => {
                    const metadata = {
                      designType: activeTab === '2k' ? '2^k Full Factorial' :
                                  activeTab === '3k' ? '3^k Full Factorial' :
                                  activeTab === 'pb' ? `Plackett-Burman Screening (${pbNumRuns} runs)` :
                                  `2^(${numFactors}-${generators.length}) Fractional Factorial`,
                      numFactors: numFactors,
                      numRuns: tableData.length,
                      fraction: activeTab === 'fractional' ? fraction : null,
                      generators: activeTab === 'fractional' ? generators : null,
                      resolution: result?.alias_structure?.resolution || result?.resolution || null
                    }
                    exportToCSVWithMetadata(tableData, factors, responseName, metadata)
                  }}
                  className="flex items-center justify-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors text-sm font-medium"
                >
                  <Download className="w-4 h-4" />
                  <span>Export CSV</span>
                </button>

                <button
                  type="button"
                  onClick={async () => {
                    const success = await copyToClipboard(tableData, factors, responseName)
                    if (success) {
                      alert('Design copied to clipboard! You can now paste it into Excel.')
                    } else {
                      alert('Failed to copy to clipboard. Please try again.')
                    }
                  }}
                  className="flex items-center justify-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium"
                >
                  <Copy className="w-4 h-4" />
                  <span>Copy to Excel</span>
                </button>

                {result && (
                  <button
                    type="button"
                    onClick={() => {
                      const filename = `factorial-results-${activeTab}`
                      exportResultsToJSON(result, filename)
                    }}
                    className="flex items-center justify-center space-x-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors text-sm font-medium"
                  >
                    <FileJson className="w-4 h-4" />
                    <span>Export Results JSON</span>
                  </button>
                )}
              </div>
              <p className="text-gray-400 text-xs mt-2">
                Export design matrix to CSV or copy to clipboard for use in Excel. Export results as JSON for record keeping.
              </p>
            </div>
          )}

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
      )}

      {/* 3^k Full Factorial Tab */}
      {activeTab === '3k' && (
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
          <form onSubmit={handleSubmit} className="space-y-6">
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
                Enter factor names separated by commas (max 4 factors for 3^k designs)
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
                              placeholder={colIndex === row.length - 1 ? '0.0' : 'Low/Medium/High'}
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
                  Table automatically generates 3^k runs (max 4 factors = 81 runs). Use "Low"/"Medium"/"High" or "-1"/"0"/"1" for levels.
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

            {/* Export Design Section */}
            {tableData.length > 0 && (
              <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-600">
                <h4 className="text-gray-100 font-semibold mb-3 text-sm">Export Design</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <button
                    type="button"
                    onClick={() => {
                      const metadata = {
                        designType: '3^k Full Factorial',
                        numFactors: numFactors,
                        numRuns: tableData.length
                      }
                      exportToCSVWithMetadata(tableData, factors, responseName, metadata)
                    }}
                    className="flex items-center justify-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors text-sm font-medium"
                  >
                    <Download className="w-4 h-4" />
                    <span>Export CSV</span>
                  </button>

                  <button
                    type="button"
                    onClick={async () => {
                      const success = await copyToClipboard(tableData, factors, responseName)
                      if (success) {
                        alert('Design copied to clipboard! You can now paste it into Excel.')
                      } else {
                        alert('Failed to copy to clipboard. Please try again.')
                      }
                    }}
                    className="flex items-center justify-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium"
                  >
                    <Copy className="w-4 h-4" />
                    <span>Copy to Excel</span>
                  </button>

                  {result && (
                    <button
                      type="button"
                      onClick={() => {
                        const filename = `factorial-results-${activeTab}`
                        exportResultsToJSON(result, filename)
                      }}
                      className="flex items-center justify-center space-x-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors text-sm font-medium"
                    >
                      <FileJson className="w-4 h-4" />
                      <span>Export Results JSON</span>
                    </button>
                  )}
                </div>
                <p className="text-gray-400 text-xs mt-2">
                  Export design matrix to CSV or copy to clipboard for use in Excel. Export results as JSON for record keeping.
                </p>
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-purple-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-purple-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Analyzing...' : 'Analyze 3^k Factorial Design'}
            </button>
          </form>
        </div>
      )}

      {/* Fractional Factorial Tab */}
      {activeTab === 'fractional' && (
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
          {/* Design Resolution Table */}
          <div className="bg-gradient-to-r from-purple-900/20 to-indigo-900/20 rounded-lg p-5 border border-purple-700/30 mb-6">
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

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Fraction Selection */}
            <div>
              <label className="block text-gray-100 font-medium mb-2">
                Fraction Size
              </label>
              <select
                value={fraction}
                onChange={(e) => setFraction(e.target.value)}
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-green-500"
              >
                <option value="1/2">1/2 Fraction (Half the runs)</option>
                <option value="1/4">1/4 Fraction (Quarter the runs)</option>
                <option value="1/8">1/8 Fraction (Eighth the runs)</option>
              </select>
              <p className="text-gray-400 text-xs mt-1">
                Smaller fractions = fewer runs but more confounding. Requires 4+ factors (1/2, 1/4) or 5+ factors (1/8).
              </p>
            </div>

            {/* Generator Display */}
            {generators.length > 0 ? (
              <div className="bg-green-900/20 rounded-lg p-4 border border-green-700/30">
                <label className="block text-gray-100 font-medium mb-2">
                  Generators (Defining Relations)
                </label>
                <div className="flex flex-wrap gap-2">
                  {generators.map((gen, idx) => (
                    <span
                      key={idx}
                      className="px-3 py-1 bg-green-700/50 text-green-100 rounded-full text-sm font-mono"
                    >
                      {gen}
                    </span>
                  ))}
                </div>
                <p className="text-gray-400 text-xs mt-2">
                  These generators define which effects are confounded. Default generators for common designs are shown.
                </p>
              </div>
            ) : numFactors > 0 && numFactors < (fraction === '1/8' ? 5 : 4) ? (
              <div className="bg-orange-900/20 rounded-lg p-4 border border-orange-700/30">
                <p className="text-orange-300 text-sm">
                  ⚠️ {fraction === '1/8' ? 'One-eighth fraction' : 'Fractional factorial'} designs require at least {fraction === '1/8' ? '5' : '4'} factors. Currently you have {numFactors} factor{numFactors !== 1 ? 's' : ''}.
                </p>
                <p className="text-orange-200/70 text-xs mt-2">
                  Add more factors to generate a {fraction} fractional design. (Auto-updating...)
                </p>
              </div>
            ) : null}

            {/* Factor Names */}
            <div>
              <label className="block text-gray-100 font-medium mb-2">
                Factor Names (comma-separated)
              </label>
              <input
                type="text"
                value={factorNames}
                onChange={(e) => setFactorNames(e.target.value)}
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-green-500"
                placeholder="e.g., A, B, C, D, E"
                required
              />
              <p className="text-gray-400 text-xs mt-1">
                Enter factor names separated by commas. Fractional designs require at least 4 factors.
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
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-green-500"
                placeholder="e.g., Yield, Strength, Quality"
                required
              />
            </div>

            {/* Number of Replicates */}
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
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-green-500"
              />
              <p className="text-gray-400 text-xs mt-1">
                Replicates allow estimation of pure error. Each run will be repeated {numReplicates} time{numReplicates > 1 ? 's' : ''}.
              </p>
            </div>

            {/* Data Input Table */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="block text-gray-100 font-medium">
                  Experimental Data
                </label>
                <button
                  type="button"
                  onClick={addRow}
                  className="flex items-center space-x-1 px-3 py-1 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors text-sm"
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
                      <th className="px-3 py-2 text-center text-gray-100 font-semibold text-sm border-b-2 border-r border-slate-600 min-w-[100px] bg-green-900/20">
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
                              className="w-full px-2 py-1.5 bg-slate-800/50 text-gray-100 border border-slate-600/50 focus:border-green-500 focus:bg-slate-700/50 hover:border-slate-500 rounded-sm focus:outline-none focus:ring-2 focus:ring-green-500/50 text-sm transition-all"
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
                  Table automatically generates 2^(k-p) fractional design runs using the specified generators. Use "Low"/"High" for levels.
                  {' '}Response values must be numbers. {tableData.length > 0 && `Current: ${tableData.length} runs.`}
                </p>
                <p className="text-green-400 text-xs">
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
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-green-500"
              />
            </div>

            {/* Export Design Section */}
            {tableData.length > 0 && (
              <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-600">
                <h4 className="text-gray-100 font-semibold mb-3 text-sm">Export Design</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <button
                    type="button"
                    onClick={() => {
                      const metadata = {
                        designType: `2^(${numFactors}-${generators.length}) Fractional Factorial`,
                        numFactors: numFactors,
                        numRuns: tableData.length,
                        fraction: fraction,
                        generators: generators,
                        resolution: result?.alias_structure?.resolution || null
                      }
                      exportToCSVWithMetadata(tableData, factors, responseName, metadata)
                    }}
                    className="flex items-center justify-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors text-sm font-medium"
                  >
                    <Download className="w-4 h-4" />
                    <span>Export CSV</span>
                  </button>

                  <button
                    type="button"
                    onClick={async () => {
                      const success = await copyToClipboard(tableData, factors, responseName)
                      if (success) {
                        alert('Design copied to clipboard! You can now paste it into Excel.')
                      } else {
                        alert('Failed to copy to clipboard. Please try again.')
                      }
                    }}
                    className="flex items-center justify-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium"
                  >
                    <Copy className="w-4 h-4" />
                    <span>Copy to Excel</span>
                  </button>

                  {result && (
                    <button
                      type="button"
                      onClick={() => {
                        const filename = `factorial-results-${activeTab}`
                        exportResultsToJSON(result, filename)
                      }}
                      className="flex items-center justify-center space-x-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors text-sm font-medium"
                    >
                      <FileJson className="w-4 h-4" />
                      <span>Export Results JSON</span>
                    </button>
                  )}
                </div>
                <p className="text-gray-400 text-xs mt-2">
                  Export design matrix to CSV or copy to clipboard for use in Excel. Export results as JSON for record keeping.
                </p>
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-green-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Analyzing...' : 'Analyze Fractional Factorial Design'}
            </button>
          </form>
        </div>
      )}

      {/* Plackett-Burman Tab */}
      {activeTab === 'pb' && (
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* PB Configuration */}
            <div className="bg-gradient-to-r from-amber-900/20 to-orange-900/20 rounded-lg p-5 border border-amber-700/30">
              <h4 className="text-amber-200 font-bold text-lg mb-3">
                Plackett-Burman Configuration
              </h4>
              <p className="text-gray-300 text-sm mb-4">
                Plackett-Burman designs are Resolution III orthogonal arrays for efficient screening of many factors.
                Main effects are confounded with 2-way interactions. Use when interactions are expected to be negligible.
              </p>

              <div>
                <label className="block text-gray-100 font-medium mb-2">
                  Number of Runs
                </label>
                <select
                  value={pbNumRuns}
                  onChange={(e) => setPbNumRuns(parseInt(e.target.value))}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-amber-500"
                >
                  <option value={12}>12 runs (screen up to 11 factors)</option>
                  <option value={20}>20 runs (screen up to 19 factors)</option>
                  <option value={24}>24 runs (screen up to 23 factors)</option>
                  <option value={28}>28 runs (screen up to 27 factors)</option>
                  <option value={36}>36 runs (screen up to 35 factors)</option>
                  <option value={44}>44 runs (screen up to 43 factors)</option>
                  <option value={48}>48 runs (screen up to 47 factors)</option>
                </select>
                <p className="text-gray-400 text-xs mt-2">
                  {numFactors > 0 && numFactors < pbNumRuns ? (
                    <span className="text-green-400">
                      ✓ {numFactors} factor{numFactors !== 1 ? 's' : ''} can be screened with {pbNumRuns} runs
                    </span>
                  ) : (
                    <span className="text-orange-400">
                      ⚠️ Number of factors ({numFactors}) must be less than number of runs ({pbNumRuns})
                    </span>
                  )}
                </p>
              </div>

              <div className="mt-4 bg-amber-900/20 rounded-lg p-4 border border-amber-700/20">
                <h5 className="text-amber-200 font-semibold text-sm mb-2">Resolution III Design</h5>
                <ul className="text-gray-300 text-xs space-y-1 list-disc list-inside">
                  <li>Main effects are confounded with 2-way interactions</li>
                  <li>Assumes all interactions are negligible</li>
                  <li>Very efficient for screening: n factors in n+1 runs (approximately)</li>
                  <li>Use this when you have many factors and want to identify the vital few</li>
                </ul>
              </div>
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
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-amber-500"
                placeholder="e.g., A, B, C, D, E, F"
                required
              />
              <p className="text-gray-400 text-xs mt-1">
                Enter factor names separated by commas. Number of factors must be less than number of runs.
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
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-amber-500"
                placeholder="e.g., Yield, Strength, Quality"
                required
              />
            </div>

            {/* Data Input Table */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="block text-gray-100 font-medium">
                  Experimental Data
                </label>
                <button
                  type="button"
                  onClick={addRow}
                  className="flex items-center space-x-1 px-3 py-1 bg-amber-600 text-white rounded-lg hover:bg-amber-700 transition-colors text-sm"
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
                      <th className="px-3 py-2 text-center text-gray-100 font-semibold text-sm border-b-2 border-r border-slate-600 min-w-[100px] bg-amber-900/20">
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
                              className="w-full px-2 py-1.5 bg-slate-800/50 text-gray-100 border border-slate-600/50 focus:border-amber-500 focus:bg-slate-700/50 hover:border-slate-500 rounded-sm focus:outline-none focus:ring-2 focus:ring-amber-500/50 text-sm transition-all"
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
                  Table automatically generates Plackett-Burman design with {pbNumRuns} runs. Use "Low"/"High" or "-1"/"1" for levels.
                  {' '}Response values must be numbers. {tableData.length > 0 && `Current: ${tableData.length} runs.`}
                </p>
                <p className="text-amber-400 text-xs">
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
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-amber-500"
              />
            </div>

            {/* Export Design Section */}
            {tableData.length > 0 && (
              <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-600">
                <h4 className="text-gray-100 font-semibold mb-3 text-sm">Export Design</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <button
                    type="button"
                    onClick={() => {
                      const metadata = {
                        designType: `Plackett-Burman Screening (${pbNumRuns} runs)`,
                        numFactors: numFactors,
                        numRuns: tableData.length
                      }
                      exportToCSVWithMetadata(tableData, factors, responseName, metadata)
                    }}
                    className="flex items-center justify-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors text-sm font-medium"
                  >
                    <Download className="w-4 h-4" />
                    <span>Export CSV</span>
                  </button>

                  <button
                    type="button"
                    onClick={async () => {
                      const success = await copyToClipboard(tableData, factors, responseName)
                      if (success) {
                        alert('Design copied to clipboard! You can now paste it into Excel.')
                      } else {
                        alert('Failed to copy to clipboard. Please try again.')
                      }
                    }}
                    className="flex items-center justify-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium"
                  >
                    <Copy className="w-4 h-4" />
                    <span>Copy to Excel</span>
                  </button>

                  {result && (
                    <button
                      type="button"
                      onClick={() => {
                        const filename = `factorial-results-${activeTab}`
                        exportResultsToJSON(result, filename)
                      }}
                      className="flex items-center justify-center space-x-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors text-sm font-medium"
                    >
                      <FileJson className="w-4 h-4" />
                      <span>Export Results JSON</span>
                    </button>
                  )}
                </div>
                <p className="text-gray-400 text-xs mt-2">
                  Export design matrix to CSV or copy to clipboard for use in Excel. Export results as JSON for record keeping.
                </p>
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-amber-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-amber-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Analyzing...' : 'Analyze Plackett-Burman Design'}
            </button>
          </form>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="bg-red-900/30 backdrop-blur-lg rounded-xl p-4 border border-red-700/50">
          <p className="text-red-200 font-medium">Error: {error}</p>
        </div>
      )}

      {/* Results Display */}
      {result && (
        <>
          <ResultCard result={result} />

          {/* Export PDF Button */}
          <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-xl font-bold text-gray-100 mb-2">Export Analysis Report</h3>
                <p className="text-gray-300 text-sm">
                  Download a comprehensive PDF report including all analysis results, effects, interactions, and visualizations.
                </p>
              </div>
              <button
                onClick={handleExportPDF}
                className="ml-4 flex items-center gap-2 bg-gradient-to-r from-emerald-600 to-teal-600 text-white font-bold py-3 px-6 rounded-lg hover:from-emerald-700 hover:to-teal-700 transition-all shadow-lg hover:shadow-xl"
              >
                <FileDown className="w-5 h-5" />
                Export PDF
              </button>
            </div>
          </div>
        </>
      )}

      {/* Cube Plot for 2^3 and 2^4 Designs */}
      {result && result.cube_data && result.cube_data.length > 0 && (
        <CubePlot
          data={result.cube_data}
          factors={factors}
          responseName={responseName}
        />
      )}

      {/* Half-Normal Plot for Effect Screening (unreplicated designs) */}
      {result && result.lenths_analysis && (
        <HalfNormalPlot lenthsData={result.lenths_analysis} />
      )}

      {/* Main Effects and Interaction Plots */}
      {result && result.interaction_plots_data && Object.keys(result.interaction_plots_data).length > 0 && (
        <FactorialInteractionPlots
          mainEffectsData={result.main_effects_plot_data}
          interactionData={result.interaction_plots_data}
          factors={factors}
        />
      )}

      {/* Diagnostic Plots */}
      {result && result.diagnostic_plots && (
        <DiagnosticPlots diagnosticPlots={result.diagnostic_plots} />
      )}

      {/* Exploratory Data Analysis */}
      {result && result.diagnostic_plots && (
        <div className="space-y-6">
          <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
            <h3 className="text-2xl font-bold text-gray-100 mb-4">Exploratory Data Analysis</h3>
            <p className="text-gray-300 text-sm mb-6">
              Visual exploration of residuals and fitted values to assess model assumptions and identify patterns.
            </p>
            <div className="grid grid-cols-1 gap-6">
              {/* Residuals Histogram */}
              {result.diagnostic_plots.residuals && (
                <Histogram
                  data={result.diagnostic_plots.residuals}
                  variableName="Residuals"
                  title="Distribution of Residuals"
                  showNormalCurve={true}
                  showKDE={true}
                />
              )}

              {/* Fitted Values Histogram */}
              {result.diagnostic_plots.fitted && (
                <Histogram
                  data={result.diagnostic_plots.fitted}
                  variableName="Fitted Values"
                  title="Distribution of Fitted Values"
                  showNormalCurve={false}
                  showKDE={true}
                />
              )}

              {/* Correlation Heatmap */}
              {result.diagnostic_plots.factor_values && result.diagnostic_plots.observed_values && (
                <CorrelationHeatmap
                  data={{
                    ...result.diagnostic_plots.factor_values,
                    [result.response_name || 'Response']: result.diagnostic_plots.observed_values
                  }}
                  title="Factor and Response Correlation Matrix"
                  method="pearson"
                />
              )}

              {/* Scatter Matrix */}
              {result.diagnostic_plots.factor_values && result.diagnostic_plots.observed_values && (
                <ScatterMatrix
                  data={{
                    ...result.diagnostic_plots.factor_values,
                    [result.response_name || 'Response']: result.diagnostic_plots.observed_values
                  }}
                  title="Pairwise Factor and Response Relationships"
                  showDiagonal={true}
                  diagonalType="histogram"
                />
              )}
            </div>
          </div>
        </div>
      )}

      {/* Alias Structure Visualization (for fractional factorial designs) */}
      {result && activeTab === 'fractional' && result.alias_structure && (
        <AliasStructureGraph aliasStructure={result.alias_structure} />
      )}

      {/* Foldover Design Section */}
      {result && activeTab === 'fractional' && result.alias_structure && (
        <div className="bg-gradient-to-br from-indigo-900/30 to-purple-900/30 backdrop-blur-lg rounded-2xl p-6 border border-indigo-700/50">
          <h3 className="text-2xl font-bold text-gray-100 mb-4">Foldover Design</h3>
          <p className="text-gray-300 mb-4">
            Use foldover designs to de-alias confounded effects. A <strong>full foldover</strong> reverses all factor signs and de-aliases main effects from two-factor interactions. A <strong>partial foldover</strong> reverses one factor to clear specific aliases involving that factor.
          </p>

          {!showFoldover ? (
            <>
              <div className="space-y-4">
                <div>
                  <label className="block text-gray-100 font-medium mb-2">
                    Foldover Type
                  </label>
                  <select
                    value={foldoverType}
                    onChange={(e) => {
                      setFoldoverType(e.target.value)
                      if (e.target.value === 'partial' && factors.length > 0) {
                        setFoldoverFactor(factors[0])
                      }
                    }}
                    className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  >
                    <option value="full">Full Foldover (reverse all factors)</option>
                    <option value="partial">Partial Foldover (reverse one factor)</option>
                  </select>
                </div>

                {foldoverType === 'partial' && (
                  <div>
                    <label className="block text-gray-100 font-medium mb-2">
                      Factor to Fold Over
                    </label>
                    <select
                      value={foldoverFactor}
                      onChange={(e) => setFoldoverFactor(e.target.value)}
                      className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    >
                      {factors.map(factor => (
                        <option key={factor} value={factor}>{factor}</option>
                      ))}
                    </select>
                    <p className="text-gray-400 text-xs mt-1">
                      Partial foldover on {foldoverFactor || 'this factor'} will de-alias effects involving {foldoverFactor || 'this factor'}
                    </p>
                  </div>
                )}

                <div className="bg-indigo-900/20 rounded-lg p-4 border border-indigo-700/30">
                  <h4 className="text-indigo-200 font-semibold mb-2">What will be cleared:</h4>
                  {foldoverType === 'full' ? (
                    <ul className="text-gray-300 text-sm space-y-1 list-disc list-inside">
                      <li>All main effects will be de-aliased from two-factor interactions</li>
                      <li>Combined design will have at least Resolution IV</li>
                      <li>Total runs will be: {tableData.length} (original) + {tableData.length} (foldover) = {tableData.length * 2}</li>
                    </ul>
                  ) : (
                    <ul className="text-gray-300 text-sm space-y-1 list-disc list-inside">
                      <li>{foldoverFactor || 'Selected factor'} will be de-aliased from interactions</li>
                      <li>All interactions involving {foldoverFactor || 'selected factor'} will be cleared</li>
                      <li>More economical than full foldover: same {tableData.length} additional runs</li>
                      <li>Other aliases not involving {foldoverFactor || 'selected factor'} remain confounded</li>
                    </ul>
                  )}
                </div>

                <button
                  type="button"
                  onClick={handleGenerateFoldover}
                  disabled={foldoverLoading}
                  className="w-full bg-indigo-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {foldoverLoading ? 'Generating...' : `Generate ${foldoverType === 'full' ? 'Full' : 'Partial'} Foldover`}
                </button>
              </div>
            </>
          ) : (
            <>
              <div className="bg-green-900/20 rounded-lg p-4 border border-green-700/30 mb-4">
                <h4 className="text-green-200 font-semibold mb-2">
                  {foldoverData.foldover_type === 'full' ? 'Full Foldover' : 'Partial Foldover'} Generated
                </h4>
                <p className="text-gray-300 text-sm mb-2">
                  {foldoverData.clearing_info?.description}
                </p>
                <ul className="text-gray-300 text-sm space-y-1">
                  <li><strong>Original runs:</strong> {foldoverData.n_original_runs}</li>
                  <li><strong>Foldover runs:</strong> {foldoverData.n_foldover_runs}</li>
                  <li><strong>Total runs:</strong> {foldoverData.n_total_runs}</li>
                  {foldoverData.foldover_factor && (
                    <li><strong>Folded factor:</strong> {foldoverData.foldover_factor}</li>
                  )}
                </ul>
              </div>

              <div className="mb-4">
                <h4 className="text-gray-100 font-semibold mb-2">Foldover Run Table</h4>
                <p className="text-gray-400 text-sm mb-2">
                  Enter or modify response values for the foldover runs below, then analyze the combined design.
                </p>
                <div className="overflow-x-auto bg-slate-700/30 rounded-lg border-2 border-indigo-600">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr className="bg-indigo-700/70">
                        <th className="px-3 py-2 text-center text-gray-100 font-semibold text-sm border-b-2 border-r border-indigo-600 w-14">
                          #
                        </th>
                        {factors.map((factor, idx) => (
                          <th
                            key={idx}
                            className="px-3 py-2 text-center text-gray-100 font-semibold text-sm border-b-2 border-r border-indigo-600 min-w-[100px]"
                          >
                            {factor}
                          </th>
                        ))}
                        <th className="px-3 py-2 text-center text-gray-100 font-semibold text-sm border-b-2 border-indigo-600 min-w-[100px] bg-indigo-900/20">
                          {responseName}
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {foldoverTableData.map((row, rowIndex) => (
                        <tr
                          key={rowIndex}
                          className="border-b border-slate-700/30 hover:bg-slate-600/10"
                        >
                          <td className="px-3 py-2 text-center text-gray-300 text-sm font-medium bg-indigo-700/30 border-r border-indigo-600">
                            {rowIndex + 1}
                          </td>
                          {row.map((cell, colIndex) => (
                            <td key={colIndex} className="px-1 py-1 border-r border-slate-700/20">
                              <input
                                id={`foldover-cell-${rowIndex}-${colIndex}`}
                                type="text"
                                value={cell}
                                onChange={(e) => handleFoldoverCellChange(rowIndex, colIndex, e.target.value)}
                                onKeyDown={(e) => handleKeyDown(e, rowIndex, colIndex)}
                                onClick={handleCellClick}
                                className="w-full px-2 py-1.5 bg-slate-800/50 text-gray-100 border border-slate-600/50 focus:border-indigo-500 focus:bg-slate-700/50 hover:border-slate-500 rounded-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/50 text-sm transition-all"
                                placeholder={colIndex === row.length - 1 ? '0.0' : 'Level'}
                                autoComplete="off"
                                disabled={colIndex < row.length - 1}
                              />
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="text-gray-400 text-xs mt-2">
                  Factor levels are automatically set based on foldover type. You can only edit response values.
                </p>
              </div>

              <div className="flex space-x-3">
                <button
                  type="button"
                  onClick={handleAnalyzeCombined}
                  disabled={foldoverLoading}
                  className="flex-1 bg-green-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {foldoverLoading ? 'Analyzing...' : 'Analyze Combined Design'}
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setShowFoldover(false)
                    setFoldoverTableData([])
                    setCombinedResult(null)
                  }}
                  className="px-6 py-3 bg-slate-600 text-white font-medium rounded-lg hover:bg-slate-700 transition-colors"
                >
                  Reset
                </button>
              </div>
            </>
          )}
        </div>
      )}

      {/* Combined Analysis Results */}
      {combinedResult && (
        <div>
          <ResultCard result={combinedResult} />
        </div>
      )}
    </div>
  )
}

export default FactorialDesigns
