import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import ResultCard from '../components/ResultCard'
import EfficiencyMetric from '../components/EfficiencyMetric'
import BlockDiagnostics from '../components/BlockDiagnostics'
import BlockStructureVisualization from '../components/BlockStructureVisualization'
import AncovaResults from '../components/AncovaResults'
import MissingDataPanel from '../components/MissingDataPanel'
import CrossoverResults from '../components/CrossoverResults'
import IncompleteBlockResults from '../components/IncompleteBlockResults'
import QuickPreprocessPanel from '../components/QuickPreprocessPanel'
import { parseTableData } from '../utils/clipboardParser'
import { Grid, Plus, Trash2, Shuffle, Activity, Grid3x3 } from 'lucide-react'

const API_URL = import.meta.env.VITE_API_URL || ''

const BlockDesigns = () => {
  const [designType, setDesignType] = useState('rcbd')
  const [nTreatments, setNTreatments] = useState(4)
  const [nBlocks, setNBlocks] = useState(3)
  const [squareSize, setSquareSize] = useState(4)
  const [nSubjects, setNSubjects] = useState(10)
  const [crossoverType, setCrossoverType] = useState('2x2')
  const [incompleteType, setIncompleteType] = useState('bib')
  const [blockSize, setBlockSize] = useState(2)
  const [nRows, setNRows] = useState(3)
  const [nColumns, setNColumns] = useState(3)
  const [responseName, setResponseName] = useState('Response')
  const [alpha, setAlpha] = useState(0.05)
  const [randomize, setRandomize] = useState(true)
  const [randomBlocks, setRandomBlocks] = useState(false)
  const [covariateColumn, setCovariateColumn] = useState('')
  const [imputationMethod, setImputationMethod] = useState('none')
  const [tableData, setTableData] = useState([])
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [generatedDesign, setGeneratedDesign] = useState(null)

  // Generate random normal value
  const randomNormal = (mean = 100, stdDev = 15) => {
    const u1 = Math.random()
    const u2 = Math.random()
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2)
    return mean + z0 * stdDev
  }

  // Generate design based on type
  const handleGenerateDesign = async () => {
    setLoading(true)
    setError(null)

    try {
      let response
      if (designType === 'rcbd') {
        response = await axios.post(`${API_URL}/api/block-designs/generate/rcbd`, {
          n_treatments: nTreatments,
          n_blocks: nBlocks,
          randomize: randomize
        })
      } else if (designType === 'latin') {
        response = await axios.post(`${API_URL}/api/block-designs/generate/latin-square`, {
          size: squareSize
        })
      } else if (designType === 'graeco') {
        response = await axios.post(`${API_URL}/api/block-designs/generate/graeco-latin`, {
          size: squareSize
        })
      } else if (designType === 'crossover') {
        const nTreatmentsForCrossover = crossoverType === '2x2' ? 2 : crossoverType === 'williams3' ? 3 : 4
        response = await axios.post(`${API_URL}/api/block-designs/crossover/generate`, {
          n_subjects: nSubjects,
          n_treatments: nTreatmentsForCrossover,
          design_type: crossoverType === '2x2' ? '2x2' : 'williams'
        })
      } else if (designType === 'incomplete') {
        if (incompleteType === 'bib') {
          response = await axios.post(`${API_URL}/api/block-designs/incomplete/generate/bib`, {
            n_treatments: nTreatments,
            block_size: blockSize
          })
        } else {
          response = await axios.post(`${API_URL}/api/block-designs/incomplete/generate/youden`, {
            n_treatments: nTreatments,
            n_rows: nRows,
            n_columns: nColumns
          })
        }
      }

      setGeneratedDesign(response.data)

      // Convert design to table data and populate with random responses
      const design = response.data.design_table
      const tableRows = design.map(row => {
        const responseValue = (Math.round(randomNormal(100, 15) * 10) / 10).toFixed(1)
        if (designType === 'rcbd') {
          // Add covariate column if specified
          if (covariateColumn && covariateColumn.trim()) {
            const covariateValue = (Math.round(randomNormal(50, 10) * 10) / 10).toFixed(1)
            return [row.run_order, row.block, row.treatment, covariateValue, responseValue]
          }
          return [row.run_order, row.block, row.treatment, responseValue]
        } else if (designType === 'latin') {
          return [row.row, row.column, row.treatment, responseValue]
        } else if (designType === 'crossover') {
          return [row.subject, row.sequence, row.period, row.treatment, responseValue]
        } else if (designType === 'incomplete') {
          if (incompleteType === 'bib') {
            return [row.block, row.treatment, responseValue]
          } else {
            return [row.row, row.column, row.treatment, responseValue]
          }
        } else {
          return [row.row, row.column, row.latin_treatment, row.greek_treatment, responseValue]
        }
      })

      setTableData(tableRows)
      setResult(null)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  // Auto-generate design on mount or when parameters change
  useEffect(() => {
    if (designType && ((designType === 'rcbd' && nTreatments >= 2 && nBlocks >= 2) ||
        ((designType === 'latin' || designType === 'graeco') && squareSize >= 3) ||
        (designType === 'crossover' && nSubjects >= 4) ||
        (designType === 'incomplete' && nTreatments >= 3))) {
      handleGenerateDesign()
    }
  }, [designType, nTreatments, nBlocks, squareSize, randomize, covariateColumn, nSubjects, crossoverType, incompleteType, blockSize, nRows, nColumns])

  const handleCellChange = (rowIndex, colIndex, value) => {
    const newData = [...tableData]
    newData[rowIndex][colIndex] = value
    setTableData(newData)
  }

  // Handle preprocessing data update
  const handlePreprocessUpdate = (columnIndex, processedValues, info) => {
    const newTableData = [...tableData]

    // Fill the column with processed values
    processedValues.forEach((value, rowIndex) => {
      if (rowIndex < newTableData.length) {
        newTableData[rowIndex][columnIndex] = value.toString()
      } else {
        // Add new rows if needed
        const numCols = getHeaders().length
        const newRow = Array(numCols).fill('')
        newRow[columnIndex] = value.toString()
        newTableData.push(newRow)
      }
    })

    // Clear remaining cells in the column if processed data is shorter
    for (let i = processedValues.length; i < newTableData.length; i++) {
      newTableData[i][columnIndex] = ''
    }

    setTableData(newTableData)
  }

  // Excel-like keyboard navigation
  const handleKeyDown = (e, rowIndex, colIndex) => {
    const maxRow = tableData.length - 1
    const maxCol = tableData[0]?.length - 1 || 0

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
        }
        break
      case 'Tab':
        if (!e.shiftKey && colIndex < maxCol) {
          e.preventDefault()
          document.getElementById(`cell-${rowIndex}-${colIndex + 1}`)?.focus()
        } else if (e.shiftKey && colIndex > 0) {
          e.preventDefault()
          document.getElementById(`cell-${rowIndex}-${colIndex - 1}`)?.focus()
        }
        break
      default:
        break
    }
  }

  const handleCellClick = (e) => {
    e.target.select()
  }

  const addRow = () => {
    const numCols = tableData[0]?.length || (designType === 'graeco' ? 5 : 4)
    const newRow = Array(numCols).fill('')
    setTableData([...tableData, newRow])
  }

  const removeRow = (rowIndex) => {
    if (tableData.length > 1) {
      setTableData(tableData.filter((_, idx) => idx !== rowIndex))
    }
  }

  // Handle paste for table
  const handlePaste = async (e) => {
    e.preventDefault()

    try {
      const text = e.clipboardData?.getData('text') ||
                   await navigator.clipboard.readText()

      if (!text) return

      const result = parseTableData(text, {
        expectHeaders: false,
        expectNumeric: false
      })

      if (!result.success) {
        console.error('Paste error:', result.error)
        return
      }

      const expectedCols = tableData[0]?.length || (designType === 'graeco' ? 5 : 4)

      // Check if pasted data matches expected columns
      if (result.columnCount !== expectedCols) {
        console.warn(`Pasted data has ${result.columnCount} columns, but ${expectedCols} columns are expected`)

        // If fewer columns, pad with empty strings
        const paddedData = result.data.map(row => {
          while (row.length < expectedCols) {
            row.push('')
          }
          return row.slice(0, expectedCols)
        })

        setTableData(paddedData)
      } else {
        setTableData(result.data)
      }
    } catch (error) {
      console.error('Failed to paste data:', error)
    }
  }

  const handleAnalyze = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      if (tableData.length < 2) {
        throw new Error('Please provide at least 2 rows of data')
      }

      let payload, endpoint

      if (designType === 'rcbd') {
        const hasCov = covariateColumn && covariateColumn.trim()
        const data = tableData
          .map((row, idx) => {
            const baseRow = {
              run_order: parseInt(row[0]) || idx + 1,
              block: String(row[1] || '').trim(),
              treatment: String(row[2] || '').trim(),
            }

            if (hasCov) {
              // With covariate: [run, block, treatment, covariate, response]
              baseRow[covariateColumn] = parseFloat(row[3])
              baseRow[responseName] = parseFloat(row[4])
            } else {
              // Without covariate: [run, block, treatment, response]
              baseRow[responseName] = parseFloat(row[3])
            }

            return baseRow
          })
          .filter(row => {
            // Filter out rows with missing or invalid data
            const isValid = row.block && row.treatment && !isNaN(row[responseName])
            return isValid
          })

        if (data.length < 2) {
          throw new Error('Please provide at least 2 complete rows of data with valid response values')
        }

        payload = {
          data: data,
          treatment: 'treatment',
          block: 'block',
          response: responseName,
          alpha: alpha,
          random_blocks: randomBlocks,
          imputation_method: imputationMethod
        }

        // Add covariate to payload if specified
        if (hasCov) {
          payload.covariate = covariateColumn
        }

        endpoint = `${API_URL}/api/block-designs/rcbd`

      } else if (designType === 'latin') {
        const data = tableData
          .map((row) => ({
            row: parseInt(row[0]),
            column: parseInt(row[1]),
            treatment: String(row[2] || '').trim(),
            [responseName]: parseFloat(row[3])
          }))
          .filter(row => {
            const isValid = !isNaN(row.row) && !isNaN(row.column) && row.treatment && !isNaN(row[responseName])
            return isValid
          })

        if (data.length < 2) {
          throw new Error('Please provide at least 2 complete rows of data with valid response values')
        }

        payload = {
          data: data,
          treatment: 'treatment',
          row_block: 'row',
          col_block: 'column',
          response: responseName,
          alpha: alpha,
          random_blocks: randomBlocks
        }
        endpoint = `${API_URL}/api/block-designs/latin-square`

      } else if (designType === 'graeco') {
        const data = tableData
          .map((row) => ({
            row: parseInt(row[0]),
            column: parseInt(row[1]),
            latin_treatment: String(row[2] || '').trim(),
            greek_treatment: String(row[3] || '').trim(),
            [responseName]: parseFloat(row[4])
          }))
          .filter(row => {
            const isValid = !isNaN(row.row) && !isNaN(row.column) &&
                           row.latin_treatment && row.greek_treatment &&
                           !isNaN(row[responseName])
            return isValid
          })

        if (data.length < 2) {
          throw new Error('Please provide at least 2 complete rows of data with valid response values')
        }

        payload = {
          data: data,
          latin_treatment: 'latin_treatment',
          greek_treatment: 'greek_treatment',
          row_block: 'row',
          col_block: 'column',
          response: responseName,
          alpha: alpha
        }
        endpoint = `${API_URL}/api/block-designs/graeco-latin`

      } else if (designType === 'crossover') {
        const data = tableData
          .map((row) => ({
            subject: String(row[0] || '').trim(),
            sequence: String(row[1] || '').trim(),
            period: String(row[2] || '').trim(),
            treatment: String(row[3] || '').trim(),
            [responseName]: parseFloat(row[4])
          }))
          .filter(row => {
            const isValid = row.subject && row.sequence && row.period &&
                           row.treatment && !isNaN(row[responseName])
            return isValid
          })

        if (data.length < 2) {
          throw new Error('Please provide at least 2 complete rows of data with valid response values')
        }

        payload = {
          data: data,
          subject: 'subject',
          period: 'period',
          treatment: 'treatment',
          sequence: 'sequence',
          response: responseName,
          alpha: alpha
        }
        endpoint = `${API_URL}/api/block-designs/crossover/analyze`

      } else if (designType === 'incomplete') {
        const data = tableData
          .map((row) => {
            if (incompleteType === 'bib') {
              return {
                block: String(row[0] || '').trim(),
                treatment: String(row[1] || '').trim(),
                [responseName]: parseFloat(row[2])
              }
            } else {
              return {
                row: String(row[0] || '').trim(),
                column: String(row[1] || '').trim(),
                treatment: String(row[2] || '').trim(),
                [responseName]: parseFloat(row[3])
              }
            }
          })
          .filter(row => {
            if (incompleteType === 'bib') {
              return row.block && row.treatment && !isNaN(row[responseName])
            } else {
              return row.row && row.column && row.treatment && !isNaN(row[responseName])
            }
          })

        if (data.length < 2) {
          throw new Error('Please provide at least 2 complete rows of data with valid response values')
        }

        payload = {
          data: data,
          treatment: 'treatment',
          block: incompleteType === 'bib' ? 'block' : 'row',
          response: responseName,
          alpha: alpha
        }
        endpoint = `${API_URL}/api/block-designs/incomplete/analyze`
      }

      const response = await axios.post(endpoint, payload)
      setResult(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  // Get column headers based on design type
  const getHeaders = () => {
    if (designType === 'rcbd') {
      if (covariateColumn && covariateColumn.trim()) {
        return ['Run', 'Block', 'Treatment', covariateColumn, responseName]
      }
      return ['Run', 'Block', 'Treatment', responseName]
    } else if (designType === 'latin') {
      return ['Row', 'Column', 'Treatment', responseName]
    } else if (designType === 'crossover') {
      return ['Subject', 'Sequence', 'Period', 'Treatment', responseName]
    } else if (designType === 'incomplete') {
      if (incompleteType === 'bib') {
        return ['Block', 'Treatment', responseName]
      } else {
        return ['Row', 'Column', 'Treatment', responseName]
      }
    } else {
      return ['Row', 'Column', 'Latin', 'Greek', responseName]
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <div className="flex items-center space-x-3 mb-4">
          <Grid className="w-8 h-8 text-pink-400" />
          <h2 className="text-3xl font-bold text-gray-100">Block Designs</h2>
        </div>
        <p className="text-gray-300">
          Powerful experimental designs for controlling nuisance variability by grouping experimental units into homogeneous blocks. Choose from RCBD, Latin Square, or Graeco-Latin Square based on your blocking structure.
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl border border-slate-700/50 overflow-hidden">
        <div className="grid grid-cols-1 md:grid-cols-5">
          <button
            onClick={() => {
              setDesignType('rcbd')
              setResult(null)
              setError(null)
              setGeneratedDesign(null)
              setTableData([])
            }}
            className={`px-6 py-4 font-semibold text-center transition-all border-b-4 ${
              designType === 'rcbd'
                ? 'bg-cyan-500/20 text-cyan-400 border-cyan-500'
                : 'text-gray-400 hover:text-gray-300 hover:bg-slate-700/30 border-transparent'
            }`}
          >
            <div className="text-lg">RCBD</div>
            <div className="text-xs mt-1 opacity-75">1 Blocking Factor</div>
          </button>
          <button
            onClick={() => {
              setDesignType('latin')
              setResult(null)
              setError(null)
              setGeneratedDesign(null)
              setTableData([])
            }}
            className={`px-6 py-4 font-semibold text-center transition-all border-b-4 ${
              designType === 'latin'
                ? 'bg-purple-500/20 text-purple-400 border-purple-500'
                : 'text-gray-400 hover:text-gray-300 hover:bg-slate-700/30 border-transparent'
            }`}
          >
            <div className="text-lg">Latin Square</div>
            <div className="text-xs mt-1 opacity-75">2 Blocking Factors</div>
          </button>
          <button
            onClick={() => {
              setDesignType('graeco')
              setResult(null)
              setError(null)
              setGeneratedDesign(null)
              setTableData([])
            }}
            className={`px-6 py-4 font-semibold text-center transition-all border-b-4 ${
              designType === 'graeco'
                ? 'bg-green-500/20 text-green-400 border-green-500'
                : 'text-gray-400 hover:text-gray-300 hover:bg-slate-700/30 border-transparent'
            }`}
          >
            <div className="text-lg">Graeco-Latin</div>
            <div className="text-xs mt-1 opacity-75">2 Treatments + 2 Blocks</div>
          </button>
          <button
            onClick={() => {
              setDesignType('crossover')
              setResult(null)
              setError(null)
              setGeneratedDesign(null)
              setTableData([])
            }}
            className={`px-6 py-4 font-semibold text-center transition-all border-b-4 ${
              designType === 'crossover'
                ? 'bg-orange-500/20 text-orange-400 border-orange-500'
                : 'text-gray-400 hover:text-gray-300 hover:bg-slate-700/30 border-transparent'
            }`}
          >
            <div className="text-lg">Crossover</div>
            <div className="text-xs mt-1 opacity-75">Repeated Measures</div>
          </button>
          <button
            onClick={() => {
              setDesignType('incomplete')
              setResult(null)
              setError(null)
              setGeneratedDesign(null)
              setTableData([])
            }}
            className={`px-6 py-4 font-semibold text-center transition-all border-b-4 ${
              designType === 'incomplete'
                ? 'bg-indigo-500/20 text-indigo-400 border-indigo-500'
                : 'text-gray-400 hover:text-gray-300 hover:bg-slate-700/30 border-transparent'
            }`}
          >
            <div className="text-lg">Incomplete</div>
            <div className="text-xs mt-1 opacity-75">BIB / Youden</div>
          </button>
        </div>
      </div>

      {/* Tab Content */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <form onSubmit={handleAnalyze} className="space-y-6">

          {/* Design Parameters */}
          {designType === 'rcbd' ? (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-gray-100 font-medium mb-2">
                    Number of Treatments
                  </label>
                  <input
                    type="number"
                    min="2"
                    max="20"
                    value={nTreatments}
                    onChange={(e) => setNTreatments(parseInt(e.target.value) || 2)}
                    className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-pink-500"
                  />
                </div>
                <div>
                  <label className="block text-gray-100 font-medium mb-2">
                    Number of Blocks
                  </label>
                  <input
                    type="number"
                    min="2"
                    max="20"
                    value={nBlocks}
                    onChange={(e) => setNBlocks(parseInt(e.target.value) || 2)}
                    className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-pink-500"
                  />
                </div>
              </div>

              {/* ANCOVA Covariate Selection */}
              <div className="bg-cyan-900/20 rounded-lg p-4 border border-cyan-700/30">
                <div className="mb-3">
                  <label className="block text-gray-100 font-medium mb-2">
                    ANCOVA: Covariate Name (Optional)
                  </label>
                  <input
                    type="text"
                    value={covariateColumn}
                    onChange={(e) => setCovariateColumn(e.target.value)}
                    placeholder="e.g., Baseline, InitialWeight, PreTest"
                    className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                  />
                </div>
                <p className="text-cyan-200 text-xs">
                  <strong>ANCOVA (Analysis of Covariance):</strong> Provide a covariate column name to adjust treatment means for a continuous variable (e.g., baseline measurements, pre-test scores). This increases precision by removing covariate variability. The covariate column will be added to your data table.
                </p>
              </div>

              {/* Missing Data Imputation */}
              <div className="bg-orange-900/20 rounded-lg p-4 border border-orange-700/30">
                <div className="mb-3">
                  <label className="block text-gray-100 font-medium mb-2">
                    Missing Data Imputation Method
                  </label>
                  <select
                    value={imputationMethod}
                    onChange={(e) => setImputationMethod(e.target.value)}
                    className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-orange-500"
                  >
                    <option value="none">None (Complete Cases Only)</option>
                    <option value="mean">Mean Imputation (Block-Specific)</option>
                    <option value="em">EM Algorithm (Advanced)</option>
                  </select>
                </div>
                <p className="text-orange-200 text-xs">
                  <strong>Missing Data Handling:</strong> Select an imputation method to handle missing response values.
                  Mean imputation uses block-specific means (simple, fast). EM algorithm uses treatment and block information
                  to predict missing values (more sophisticated). Leave empty cells or enter "NA" for missing data in the table.
                </p>
              </div>
            </div>
          ) : designType === 'crossover' ? (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-gray-100 font-medium mb-2">
                    Number of Subjects
                  </label>
                  <input
                    type="number"
                    min="4"
                    max="100"
                    value={nSubjects}
                    onChange={(e) => setNSubjects(parseInt(e.target.value) || 10)}
                    className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-orange-500"
                  />
                </div>
                <div>
                  <label className="block text-gray-100 font-medium mb-2">
                    Crossover Design Type
                  </label>
                  <select
                    value={crossoverType}
                    onChange={(e) => setCrossoverType(e.target.value)}
                    className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-orange-500"
                  >
                    <option value="2x2">2×2 Crossover (AB/BA)</option>
                    <option value="williams3">Williams 3-Treatment</option>
                    <option value="williams4">Williams 4-Treatment</option>
                  </select>
                </div>
              </div>

              <div className="bg-orange-900/20 rounded-lg p-4 border border-orange-700/30">
                <h4 className="text-gray-100 font-semibold mb-2">About Crossover Designs</h4>
                <p className="text-orange-200 text-xs">
                  <strong>Crossover designs:</strong> Each subject receives multiple treatments in sequence across different time periods.
                  Subjects serve as their own controls, increasing precision. The design tests for carryover effects
                  (residual effects from previous treatments) and period effects (time-dependent changes).
                </p>
              </div>
            </div>
          ) : designType === 'incomplete' ? (
            <div className="space-y-4">
              <div>
                <label className="block text-gray-100 font-medium mb-2">
                  Incomplete Design Type
                </label>
                <select
                  value={incompleteType}
                  onChange={(e) => setIncompleteType(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  <option value="bib">Balanced Incomplete Block (BIB)</option>
                  <option value="youden">Youden Square</option>
                </select>
              </div>

              {incompleteType === 'bib' ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-gray-100 font-medium mb-2">
                      Number of Treatments
                    </label>
                    <input
                      type="number"
                      min="3"
                      max="15"
                      value={nTreatments}
                      onChange={(e) => setNTreatments(parseInt(e.target.value) || 4)}
                      className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    />
                  </div>
                  <div>
                    <label className="block text-gray-100 font-medium mb-2">
                      Block Size (k)
                    </label>
                    <input
                      type="number"
                      min="2"
                      max={nTreatments - 1}
                      value={blockSize}
                      onChange={(e) => setBlockSize(parseInt(e.target.value) || 2)}
                      className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    />
                    <p className="text-gray-400 text-xs mt-1">
                      Block size must be less than number of treatments
                    </p>
                  </div>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label className="block text-gray-100 font-medium mb-2">
                      Number of Treatments
                    </label>
                    <input
                      type="number"
                      min="4"
                      max="12"
                      value={nTreatments}
                      onChange={(e) => setNTreatments(parseInt(e.target.value) || 5)}
                      className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    />
                  </div>
                  <div>
                    <label className="block text-gray-100 font-medium mb-2">
                      Number of Rows
                    </label>
                    <input
                      type="number"
                      min="2"
                      max={nTreatments}
                      value={nRows}
                      onChange={(e) => setNRows(parseInt(e.target.value) || 3)}
                      className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    />
                  </div>
                  <div>
                    <label className="block text-gray-100 font-medium mb-2">
                      Number of Columns
                    </label>
                    <input
                      type="number"
                      min="2"
                      max={nTreatments - 1}
                      value={nColumns}
                      onChange={(e) => setNColumns(parseInt(e.target.value) || 3)}
                      className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    />
                    <p className="text-gray-400 text-xs mt-1">
                      Columns &lt; treatments for incomplete design
                    </p>
                  </div>
                </div>
              )}

              <div className="bg-indigo-900/20 rounded-lg p-4 border border-indigo-700/30">
                <h4 className="text-gray-100 font-semibold mb-2">
                  About {incompleteType === 'bib' ? 'BIB Designs' : 'Youden Squares'}
                </h4>
                <p className="text-indigo-200 text-xs">
                  {incompleteType === 'bib' ? (
                    <>
                      <strong>Balanced Incomplete Block (BIB):</strong> Not all treatments appear in every block.
                      Each treatment appears the same number of times (r), and every pair of treatments occurs together
                      in the same number of blocks (λ). Useful when block size must be smaller than the number of treatments.
                    </>
                  ) : (
                    <>
                      <strong>Youden Square:</strong> An incomplete Latin square where rows are incomplete blocks.
                      Each treatment appears once per column, but not all treatments appear in each row.
                      Controls for two blocking factors with fewer runs than a full Latin square.
                    </>
                  )}
                </p>
              </div>
            </div>
          ) : (
            <div>
              <label className="block text-gray-100 font-medium mb-2">
                Square Size (n × n)
              </label>
              <input
                type="number"
                min={designType === 'graeco' ? 3 : 2}
                max="12"
                value={squareSize}
                onChange={(e) => setSquareSize(parseInt(e.target.value) || 3)}
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-pink-500"
              />
              <p className="text-gray-400 text-xs mt-1">
                {designType === 'graeco'
                  ? 'Graeco-Latin squares do not exist for n=2 or n=6. Size must be 3-12.'
                  : 'Latin square size (treatments = rows = columns). Total runs = n²'}
              </p>
            </div>
          )}

          {/* Response Variable Name */}
          <div>
            <label className="block text-gray-100 font-medium mb-2">
              Response Variable Name
            </label>
            <input
              type="text"
              value={responseName}
              onChange={(e) => setResponseName(e.target.value)}
              className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-pink-500"
              placeholder="e.g., Yield, Strength, Quality"
              required
            />
          </div>

          {/* Randomization Option (RCBD only) */}
          {designType === 'rcbd' && (
            <div className="flex items-center space-x-3">
              <input
                type="checkbox"
                id="randomize"
                checked={randomize}
                onChange={(e) => setRandomize(e.target.checked)}
                className="w-4 h-4 text-pink-600 bg-slate-700 border-slate-600 rounded focus:ring-pink-500"
              />
              <label htmlFor="randomize" className="text-gray-100 font-medium">
                Randomize Run Order
              </label>
              <Shuffle className="w-4 h-4 text-pink-400" />
            </div>
          )}

          {/* Random Blocks Option (RCBD and Latin Square) */}
          {(designType === 'rcbd' || designType === 'latin') && (
            <div className="bg-purple-900/20 rounded-lg p-4 border border-purple-700/30">
              <div className="flex items-start space-x-3">
                <input
                  type="checkbox"
                  id="randomBlocks"
                  checked={randomBlocks}
                  onChange={(e) => setRandomBlocks(e.target.checked)}
                  className="w-4 h-4 text-purple-600 bg-slate-700 border-slate-600 rounded focus:ring-purple-500 mt-0.5"
                />
                <div className="flex-1">
                  <label htmlFor="randomBlocks" className="text-gray-100 font-medium block mb-1">
                    Treat Blocks as Random Effects
                  </label>
                  <p className="text-gray-400 text-xs">
                    <strong>Fixed blocks:</strong> Block levels are the only ones of interest (standard ANOVA).<br />
                    <strong>Random blocks:</strong> Blocks are a random sample from a larger population (mixed model with variance components).
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Generated Design Info */}
          {generatedDesign && (
            <div className="bg-pink-900/20 rounded-lg p-4 border border-pink-700/30">
              <h4 className="text-gray-100 font-semibold mb-2">Generated Design</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="bg-slate-800/50 rounded-lg p-2">
                  <p className="text-gray-400 text-xs">Design Type</p>
                  <p className="text-gray-100 text-sm font-semibold">{generatedDesign.design_type}</p>
                </div>
                <div className="bg-slate-800/50 rounded-lg p-2">
                  <p className="text-gray-400 text-xs">Total Runs</p>
                  <p className="text-gray-100 text-sm font-semibold">{generatedDesign.n_runs}</p>
                </div>
                {generatedDesign.n_treatments && (
                  <div className="bg-slate-800/50 rounded-lg p-2">
                    <p className="text-gray-400 text-xs">Treatments</p>
                    <p className="text-gray-100 text-sm font-semibold">{generatedDesign.n_treatments}</p>
                  </div>
                )}
                {generatedDesign.n_blocks && (
                  <div className="bg-slate-800/50 rounded-lg p-2">
                    <p className="text-gray-400 text-xs">Blocks</p>
                    <p className="text-gray-100 text-sm font-semibold">{generatedDesign.n_blocks}</p>
                  </div>
                )}
                {generatedDesign.size && (
                  <div className="bg-slate-800/50 rounded-lg p-2">
                    <p className="text-gray-400 text-xs">Square Size</p>
                    <p className="text-gray-100 text-sm font-semibold">{generatedDesign.size}×{generatedDesign.size}</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Block Structure Visualization */}
          {generatedDesign && generatedDesign.design_table && (
            <BlockStructureVisualization
              designType={designType}
              designTable={generatedDesign.design_table}
              nBlocks={nBlocks}
              nTreatments={nTreatments}
              squareSize={squareSize}
            />
          )}

          {/* Quick Preprocessing Panel */}
          <QuickPreprocessPanel
            tableData={tableData}
            columnNames={getHeaders()}
            onDataUpdate={handlePreprocessUpdate}
            className="mb-4"
          />

          {/* Data Table */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="block text-gray-100 font-medium">
                Experimental Data
              </label>
              <div className="flex space-x-2">
                <button
                  type="button"
                  onClick={handleGenerateDesign}
                  className="flex items-center space-x-1 px-3 py-1 bg-pink-600 text-white rounded-lg hover:bg-pink-700 transition-colors text-sm"
                >
                  <Shuffle className="w-4 h-4" />
                  <span>Regenerate</span>
                </button>
                <button
                  type="button"
                  onClick={addRow}
                  className="flex items-center space-x-1 px-3 py-1 bg-pink-600 text-white rounded-lg hover:bg-pink-700 transition-colors text-sm"
                >
                  <Plus className="w-4 h-4" />
                  <span>Add Row</span>
                </button>
              </div>
            </div>

            <div
              className="overflow-x-auto bg-slate-700/30 rounded-lg border-2 border-slate-600"
              onPaste={handlePaste}
            >
              <table className="w-full border-collapse">
                <thead>
                  <tr className="bg-slate-700/70">
                    <th className="px-3 py-2 text-center text-gray-100 font-semibold text-sm border-b-2 border-r border-slate-600 w-14 sticky left-0 bg-slate-700/70">
                      #
                    </th>
                    {getHeaders().map((header, idx) => (
                      <th
                        key={idx}
                        className={`px-3 py-2 text-center text-gray-100 font-semibold text-sm border-b-2 border-r border-slate-600 min-w-[100px] ${
                          idx === getHeaders().length - 1 ? 'bg-pink-900/20' : ''
                        }`}
                      >
                        {header}
                      </th>
                    ))}
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
                            className="w-full px-2 py-1.5 bg-slate-800/50 text-gray-100 border border-slate-600/50 focus:border-pink-500 focus:bg-slate-700/50 hover:border-slate-500 rounded-sm focus:outline-none focus:ring-2 focus:ring-pink-500/50 text-sm transition-all"
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
                {designType === 'rcbd'
                  ? `RCBD with ${nTreatments} treatments × ${nBlocks} blocks = ${nTreatments * nBlocks} runs. Data auto-generated with random responses.`
                  : designType === 'latin'
                  ? `Latin Square ${squareSize}×${squareSize} = ${squareSize * squareSize} runs. Each treatment appears once per row and column.`
                  : designType === 'crossover'
                  ? `Crossover design: ${nSubjects} subjects, each receiving multiple treatments in sequence across time periods. Data auto-generated with random responses.`
                  : designType === 'incomplete'
                  ? incompleteType === 'bib'
                    ? `Balanced Incomplete Block (BIB): ${nTreatments} treatments, block size ${blockSize}. Not all treatments appear in each block.`
                    : `Youden Square: ${nTreatments} treatments, ${nRows}×${nColumns} incomplete Latin square. Each treatment appears once per column.`
                  : `Graeco-Latin Square ${squareSize}×${squareSize} = ${squareSize * squareSize} runs. Two orthogonal Latin squares superimposed.`
                }
              </p>
              <p className="text-pink-400 text-xs">
                <strong>Excel-like navigation:</strong> Use Arrow keys, Enter, Tab. Click to select all. Paste data from Excel (Ctrl+V / Cmd+V).
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
              className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-pink-500"
            />
          </div>

          {/* Analyze Button */}
          <button
            type="submit"
            disabled={loading || tableData.length === 0}
            className="w-full bg-pink-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-pink-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Analyzing...' : 'Analyze Block Design'}
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
      {result && (
        <div className="space-y-6">
          <ResultCard result={result} />

          {/* Efficiency Metric */}
          {result.relative_efficiency && (
            <EfficiencyMetric
              relativeEfficiency={result.relative_efficiency}
              designType={designType === 'rcbd' ? 'RCBD' : designType === 'latin' ? 'Latin Square' : 'Graeco-Latin Square'}
            />
          )}

          {/* Block Diagnostics */}
          {(result.normality_test || result.homogeneity_test || result.interaction_means) && (
            <BlockDiagnostics
              normalityTest={result.normality_test}
              homogeneityTest={result.homogeneity_test}
              interactionMeans={result.interaction_means}
              blockType={randomBlocks ? 'random' : 'fixed'}
            />
          )}

          {/* Missing Data Analysis */}
          {result.missing_data && designType === 'rcbd' && (
            <MissingDataPanel missingData={result.missing_data} />
          )}

          {/* ANCOVA Results */}
          {result.ancova && designType === 'rcbd' && (
            <AncovaResults
              ancovaData={result.ancova}
              unadjustedMeans={result.treatment_means || {}}
            />
          )}

          {/* Crossover Results */}
          {designType === 'crossover' && result && (
            <CrossoverResults crossoverData={result} />
          )}

          {/* Incomplete Block Results */}
          {designType === 'incomplete' && result && (
            <IncompleteBlockResults incompleteData={result} />
          )}
        </div>
      )}
    </div>
  )
}

export default BlockDesigns
