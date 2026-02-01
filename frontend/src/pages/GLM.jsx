import { useState, useCallback, useEffect } from 'react'
import axios from 'axios'
import Plot from 'react-plotly.js'
import { Upload, Settings, BarChart3, Activity, Info, Clipboard, AlertTriangle, CheckCircle } from 'lucide-react'
import * as XLSX from 'xlsx'

const API_URL = import.meta.env.VITE_API_URL || ''

const GLM = () => {
  // State
  const [activeTab, setActiveTab] = useState('data')
  const [tableData, setTableData] = useState(
    Array(20).fill(null).map(() => Array(3).fill(''))
  )
  const [predictorNames, setPredictorNames] = useState(['X1', 'X2'])
  const [responseName, setResponseName] = useState('Y')
  const [numPredictors, setNumPredictors] = useState(2)

  // Model settings
  const [families, setFamilies] = useState([])
  const [selectedFamily, setSelectedFamily] = useState('poisson')
  const [selectedLink, setSelectedLink] = useState('')
  const [addIntercept, setAddIntercept] = useState(true)

  // Results
  const [fitResult, setFitResult] = useState(null)
  const [compareResult, setCompareResult] = useState(null)
  const [diagnosticsResult, setDiagnosticsResult] = useState(null)

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  // Load available families on mount
  useEffect(() => {
    loadFamilies()
  }, [])

  const loadFamilies = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/glm/families`)
      setFamilies(response.data.families)
    } catch (err) {
      console.error('Failed to load families:', err)
    }
  }

  // Update table columns when predictor count changes
  useEffect(() => {
    const newNames = Array.from({ length: numPredictors }, (_, i) =>
      predictorNames[i] || `X${i + 1}`
    )
    setPredictorNames(newNames)

    // Update table data to match new column count
    setTableData(prev => prev.map(row => {
      const newRow = Array(numPredictors + 1).fill('')
      row.forEach((val, i) => {
        if (i < newRow.length) newRow[i] = val
      })
      return newRow
    }))
  }, [numPredictors])

  // Excel-like table handlers
  const handleCellChange = useCallback((rowIndex, colIndex, value) => {
    setTableData(prev => {
      const newData = prev.map(row => [...row])
      newData[rowIndex][colIndex] = value
      if (rowIndex === newData.length - 1 && value.trim() !== '') {
        newData.push(Array(numPredictors + 1).fill(''))
      }
      return newData
    })
  }, [numPredictors])

  const handleKeyDown = useCallback((e, rowIndex, colIndex) => {
    const numRows = tableData.length
    const numCols = numPredictors + 1
    let newRow = rowIndex
    let newCol = colIndex

    switch (e.key) {
      case 'ArrowUp':
        e.preventDefault()
        newRow = Math.max(0, rowIndex - 1)
        break
      case 'ArrowDown':
        e.preventDefault()
        newRow = Math.min(numRows - 1, rowIndex + 1)
        if (rowIndex === numRows - 1) {
          setTableData(prev => [...prev, Array(numCols).fill('')])
          newRow = numRows
        }
        break
      case 'ArrowLeft':
        e.preventDefault()
        newCol = Math.max(0, colIndex - 1)
        break
      case 'ArrowRight':
        e.preventDefault()
        newCol = Math.min(numCols - 1, colIndex + 1)
        break
      case 'Tab':
        e.preventDefault()
        if (e.shiftKey) {
          if (colIndex > 0) {
            newCol = colIndex - 1
          } else if (rowIndex > 0) {
            newRow = rowIndex - 1
            newCol = numCols - 1
          }
        } else {
          if (colIndex < numCols - 1) {
            newCol = colIndex + 1
          } else if (rowIndex < numRows - 1) {
            newRow = rowIndex + 1
            newCol = 0
          } else {
            setTableData(prev => [...prev, Array(numCols).fill('')])
            newRow = numRows
            newCol = 0
          }
        }
        break
      case 'Enter':
        e.preventDefault()
        newRow = Math.min(numRows - 1, rowIndex + 1)
        if (rowIndex === numRows - 1) {
          setTableData(prev => [...prev, Array(numCols).fill('')])
          newRow = numRows
        }
        newCol = 0
        break
      default:
        return
    }

    setTimeout(() => {
      const input = document.getElementById(`glm-cell-${newRow}-${newCol}`)
      if (input) {
        input.focus()
        input.select()
      }
    }, 0)
  }, [tableData.length, numPredictors])

  const handlePaste = useCallback((e) => {
    e.preventDefault()
    const pastedData = e.clipboardData.getData('text')
    const lines = pastedData.trim().split('\n')
    const newTableData = []
    const numCols = numPredictors + 1

    lines.forEach(line => {
      const parts = line.trim().split(/[\s,\t]+/)
      const row = Array(numCols).fill('')
      parts.forEach((val, i) => {
        if (i < numCols) row[i] = val
      })
      newTableData.push(row)
    })

    while (newTableData.length < 20) {
      newTableData.push(Array(numCols).fill(''))
    }
    newTableData.push(Array(numCols).fill(''))

    setTableData(newTableData)
  }, [numPredictors])

  // Parse table data
  const parseTableData = useCallback(() => {
    const yData = []
    const xData = []

    tableData.forEach(row => {
      const y = parseFloat(row[numPredictors])
      const xRow = row.slice(0, numPredictors).map(v => parseFloat(v))

      if (!isNaN(y) && xRow.every(v => !isNaN(v))) {
        yData.push(y)
        xData.push(xRow)
      }
    })

    return { yData, xData }
  }, [tableData, numPredictors])

  // Load example data
  const loadExampleData = (exampleType) => {
    let data, names, response, family

    if (exampleType === 'poisson') {
      // Count data example (defects per batch)
      data = [
        [1, 10, 3], [2, 15, 5], [3, 20, 8], [4, 25, 12], [5, 30, 15],
        [1, 35, 4], [2, 40, 7], [3, 45, 11], [4, 50, 16], [5, 55, 20],
        [1, 60, 5], [2, 65, 9], [3, 70, 13], [4, 75, 18], [5, 80, 24]
      ]
      names = ['ShiftHours', 'Temperature']
      response = 'DefectCount'
      family = 'poisson'
      setNumPredictors(2)
    } else if (exampleType === 'binomial') {
      // Binary outcome (pass/fail)
      data = [
        [1, 50, 0], [2, 55, 0], [3, 60, 0], [4, 65, 1], [5, 70, 1],
        [1, 75, 0], [2, 80, 1], [3, 85, 1], [4, 90, 1], [5, 95, 1],
        [1, 100, 1], [2, 105, 1], [3, 110, 1], [4, 115, 1], [5, 120, 1]
      ]
      names = ['Concentration', 'Temperature']
      response = 'Success'
      family = 'binomial'
      setNumPredictors(2)
    } else if (exampleType === 'gamma') {
      // Positive continuous (wait times)
      data = [
        [1, 5, 2.3], [2, 10, 4.1], [3, 15, 5.8], [4, 20, 8.2], [5, 25, 11.5],
        [1, 30, 3.1], [2, 35, 5.4], [3, 40, 7.9], [4, 45, 11.2], [5, 50, 15.6],
        [1, 55, 4.2], [2, 60, 7.1], [3, 65, 10.5], [4, 70, 14.8], [5, 75, 20.1]
      ]
      names = ['Priority', 'QueueLength']
      response = 'WaitTime'
      family = 'gamma'
      setNumPredictors(2)
    } else if (exampleType === 'negativebinomial') {
      // Overdispersed counts
      data = [
        [1, 100, 5], [2, 200, 12], [3, 300, 25], [4, 400, 45], [5, 500, 80],
        [1, 150, 8], [2, 250, 18], [3, 350, 35], [4, 450, 60], [5, 550, 100],
        [1, 175, 10], [2, 275, 22], [3, 375, 42], [4, 475, 72], [5, 575, 120]
      ]
      names = ['Region', 'Population']
      response = 'Claims'
      family = 'negativebinomial'
      setNumPredictors(2)
    }

    const newTableData = data.map(row => row.map(v => v.toString()))
    while (newTableData.length < 20) {
      newTableData.push(Array(3).fill(''))
    }
    newTableData.push(Array(3).fill(''))

    setTableData(newTableData)
    setPredictorNames(names)
    setResponseName(response)
    setSelectedFamily(family)
    setSelectedLink('')
    setActiveTab('model')
  }

  // File upload
  const handleFileUpload = (event) => {
    const file = event.target.files[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (e) => {
      try {
        const fileData = new Uint8Array(e.target.result)
        const workbook = XLSX.read(fileData, { type: 'array' })
        const sheetName = workbook.SheetNames[0]
        const worksheet = workbook.Sheets[sheetName]
        const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 })

        if (jsonData.length < 2) {
          setError('File must have header row and at least one data row')
          return
        }

        // First row is headers
        const headers = jsonData[0].map(h => String(h))
        const numCols = headers.length

        if (numCols < 2) {
          setError('File must have at least 2 columns (1 predictor + 1 response)')
          return
        }

        // Set predictor names (all but last column) and response name (last column)
        setNumPredictors(numCols - 1)
        setPredictorNames(headers.slice(0, -1))
        setResponseName(headers[numCols - 1])

        // Parse data rows
        const newTableData = []
        for (let i = 1; i < jsonData.length; i++) {
          const row = jsonData[i]
          const tableRow = Array(numCols).fill('')
          for (let j = 0; j < numCols; j++) {
            if (row[j] !== undefined && row[j] !== null) {
              tableRow[j] = String(row[j])
            }
          }
          newTableData.push(tableRow)
        }

        while (newTableData.length < 20) {
          newTableData.push(Array(numCols).fill(''))
        }
        newTableData.push(Array(numCols).fill(''))

        setTableData(newTableData)
        setError('')
        setActiveTab('model')
      } catch (err) {
        setError('Failed to read file: ' + err.message)
      }
    }
    reader.readAsArrayBuffer(file)
  }

  // Fit GLM
  const handleFitModel = async () => {
    const { yData, xData } = parseTableData()

    if (yData.length < 3) {
      setError('Need at least 3 complete observations')
      return
    }

    setLoading(true)
    setError('')

    try {
      const response = await axios.post(`${API_URL}/api/glm/fit`, {
        y_data: yData,
        x_data: xData,
        predictor_names: predictorNames,
        response_name: responseName,
        family: selectedFamily,
        link: selectedLink || null,
        add_intercept: addIntercept
      })

      setFitResult(response.data)
      setActiveTab('results')
    } catch (err) {
      setError('Model fitting failed: ' + (err.response?.data?.detail || err.message))
    } finally {
      setLoading(false)
    }
  }

  // Compare families
  const handleCompareModels = async () => {
    const { yData, xData } = parseTableData()

    if (yData.length < 3) {
      setError('Need at least 3 complete observations')
      return
    }

    setLoading(true)
    setError('')

    try {
      const response = await axios.post(`${API_URL}/api/glm/compare`, {
        y_data: yData,
        x_data: xData,
        predictor_names: predictorNames,
        families: ['poisson', 'negativebinomial', 'gamma', 'gaussian'],
        add_intercept: addIntercept
      })

      setCompareResult(response.data)
    } catch (err) {
      setError('Model comparison failed: ' + (err.response?.data?.detail || err.message))
    } finally {
      setLoading(false)
    }
  }

  // Run diagnostics
  const handleRunDiagnostics = async () => {
    const { yData, xData } = parseTableData()

    if (yData.length < 3) {
      setError('Need at least 3 complete observations')
      return
    }

    setLoading(true)
    setError('')

    try {
      const response = await axios.post(`${API_URL}/api/glm/diagnostics`, {
        y_data: yData,
        x_data: xData,
        family: selectedFamily,
        link: selectedLink || null,
        add_intercept: addIntercept
      })

      setDiagnosticsResult(response.data)
      setActiveTab('diagnostics')
    } catch (err) {
      setError('Diagnostics failed: ' + (err.response?.data?.detail || err.message))
    } finally {
      setLoading(false)
    }
  }

  // Get selected family info
  const selectedFamilyInfo = families.find(f => f.id === selectedFamily) || {}

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-pink-600 text-transparent bg-clip-text">
            Generalized Linear Models
          </h1>
          <p className="text-gray-300 text-lg">
            Fit GLMs with Poisson, Binomial, Gamma, Negative Binomial, and other distributions
          </p>
        </div>

        {/* Tab Navigation */}
        <div className="flex space-x-2 bg-slate-800/50 p-2 rounded-lg mb-6">
          <button
            onClick={() => setActiveTab('data')}
            className={`flex-1 px-4 py-3 rounded-lg font-medium transition-all duration-200 flex items-center justify-center space-x-2 ${
              activeTab === 'data'
                ? 'bg-purple-600 text-white shadow-lg'
                : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
            }`}
          >
            <Upload size={20} />
            <span>1. Data</span>
          </button>
          <button
            onClick={() => setActiveTab('model')}
            className={`flex-1 px-4 py-3 rounded-lg font-medium transition-all duration-200 flex items-center justify-center space-x-2 ${
              activeTab === 'model'
                ? 'bg-purple-600 text-white shadow-lg'
                : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
            }`}
          >
            <Settings size={20} />
            <span>2. Model</span>
          </button>
          <button
            onClick={() => setActiveTab('results')}
            disabled={!fitResult}
            className={`flex-1 px-4 py-3 rounded-lg font-medium transition-all duration-200 flex items-center justify-center space-x-2 ${
              activeTab === 'results'
                ? 'bg-purple-600 text-white shadow-lg'
                : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700 disabled:opacity-40'
            }`}
          >
            <BarChart3 size={20} />
            <span>3. Results</span>
          </button>
          <button
            onClick={() => setActiveTab('diagnostics')}
            disabled={!diagnosticsResult && !fitResult}
            className={`flex-1 px-4 py-3 rounded-lg font-medium transition-all duration-200 flex items-center justify-center space-x-2 ${
              activeTab === 'diagnostics'
                ? 'bg-purple-600 text-white shadow-lg'
                : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700 disabled:opacity-40'
            }`}
          >
            <Activity size={20} />
            <span>4. Diagnostics</span>
          </button>
          <button
            onClick={() => setActiveTab('info')}
            className={`flex-1 px-4 py-3 rounded-lg font-medium transition-all duration-200 flex items-center justify-center space-x-2 ${
              activeTab === 'info'
                ? 'bg-purple-600 text-white shadow-lg'
                : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
            }`}
          >
            <Info size={20} />
            <span>Info</span>
          </button>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-6 bg-red-900/50 border border-red-600 text-red-200 px-4 py-3 rounded-lg flex items-center space-x-2">
            <AlertTriangle size={20} />
            <span>{error}</span>
          </div>
        )}

        {/* Data Tab */}
        {activeTab === 'data' && (
          <div className="space-y-6">
            {/* Example Data */}
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <h2 className="text-2xl font-bold mb-4 text-gray-100">Quick Start: Load Example Data</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <button
                  onClick={() => loadExampleData('poisson')}
                  className="px-6 py-4 bg-gradient-to-br from-blue-600 to-blue-700 hover:from-blue-500 hover:to-blue-600 text-white rounded-xl font-semibold transition-all duration-200 shadow-lg"
                >
                  <div className="text-lg">Poisson</div>
                  <div className="text-xs opacity-75">Count data</div>
                </button>
                <button
                  onClick={() => loadExampleData('binomial')}
                  className="px-6 py-4 bg-gradient-to-br from-green-600 to-green-700 hover:from-green-500 hover:to-green-600 text-white rounded-xl font-semibold transition-all duration-200 shadow-lg"
                >
                  <div className="text-lg">Binomial</div>
                  <div className="text-xs opacity-75">Binary outcome</div>
                </button>
                <button
                  onClick={() => loadExampleData('gamma')}
                  className="px-6 py-4 bg-gradient-to-br from-orange-600 to-orange-700 hover:from-orange-500 hover:to-orange-600 text-white rounded-xl font-semibold transition-all duration-200 shadow-lg"
                >
                  <div className="text-lg">Gamma</div>
                  <div className="text-xs opacity-75">Positive continuous</div>
                </button>
                <button
                  onClick={() => loadExampleData('negativebinomial')}
                  className="px-6 py-4 bg-gradient-to-br from-purple-600 to-purple-700 hover:from-purple-500 hover:to-purple-600 text-white rounded-xl font-semibold transition-all duration-200 shadow-lg"
                >
                  <div className="text-lg">Neg. Binomial</div>
                  <div className="text-xs opacity-75">Overdispersed counts</div>
                </button>
              </div>
            </div>

            {/* Data Setup */}
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <h2 className="text-2xl font-bold mb-4 text-gray-100">Data Setup</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Number of Predictors
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="10"
                    value={numPredictors}
                    onChange={(e) => setNumPredictors(Math.max(1, Math.min(10, parseInt(e.target.value) || 1)))}
                    className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Response Variable Name
                  </label>
                  <input
                    type="text"
                    value={responseName}
                    onChange={(e) => setResponseName(e.target.value)}
                    className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>
                <div className="flex items-end">
                  <input
                    type="file"
                    accept=".xlsx,.xls,.csv"
                    onChange={handleFileUpload}
                    className="block w-full text-sm text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-purple-600 file:text-white hover:file:bg-purple-700 cursor-pointer"
                  />
                </div>
              </div>

              {/* Predictor Names */}
              <div className="grid grid-cols-2 md:grid-cols-5 gap-2 mb-4">
                {predictorNames.map((name, i) => (
                  <div key={i}>
                    <label className="block text-xs text-gray-400 mb-1">Predictor {i + 1}</label>
                    <input
                      type="text"
                      value={name}
                      onChange={(e) => {
                        const newNames = [...predictorNames]
                        newNames[i] = e.target.value
                        setPredictorNames(newNames)
                      }}
                      className="w-full px-2 py-1 bg-slate-700/50 border border-slate-600 rounded text-white text-sm focus:outline-none focus:ring-1 focus:ring-purple-500"
                    />
                  </div>
                ))}
              </div>
            </div>

            {/* Data Entry Table */}
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-2xl font-bold text-gray-100">Enter Data</h2>
                <button
                  onClick={() => {
                    navigator.clipboard.readText().then(text => {
                      const lines = text.trim().split('\n')
                      const numCols = numPredictors + 1
                      const newTableData = []
                      lines.forEach(line => {
                        const parts = line.trim().split(/[\s,\t]+/)
                        const row = Array(numCols).fill('')
                        parts.forEach((val, i) => { if (i < numCols) row[i] = val })
                        newTableData.push(row)
                      })
                      while (newTableData.length < 20) newTableData.push(Array(numCols).fill(''))
                      newTableData.push(Array(numCols).fill(''))
                      setTableData(newTableData)
                    }).catch(() => setError('Failed to read clipboard'))
                  }}
                  className="flex items-center gap-1 px-3 py-1 bg-slate-700 hover:bg-slate-600 text-gray-300 rounded text-sm transition-colors"
                >
                  <Clipboard size={14} />
                  Paste
                </button>
              </div>
              <p className="text-sm text-gray-400 mb-3">
                Enter predictor values followed by response. Use arrow keys or Tab to navigate.
              </p>
              <div className="max-h-80 overflow-y-auto border border-slate-600 rounded-lg">
                <table className="w-full">
                  <thead className="sticky top-0 bg-slate-700">
                    <tr>
                      <th className="w-12 px-2 py-2 text-left text-xs font-semibold text-gray-400 border-b border-slate-600">#</th>
                      {predictorNames.map((name, i) => (
                        <th key={i} className="px-3 py-2 text-left text-xs font-semibold text-gray-400 border-b border-slate-600">
                          {name}
                        </th>
                      ))}
                      <th className="px-3 py-2 text-left text-xs font-semibold text-purple-400 border-b border-slate-600">
                        {responseName}
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {tableData.map((row, rowIndex) => (
                      <tr key={rowIndex} className="border-b border-slate-700/50 hover:bg-slate-700/30">
                        <td className="px-2 py-1 text-xs text-gray-500 font-mono">{rowIndex + 1}</td>
                        {row.map((cell, colIndex) => (
                          <td key={colIndex} className={`p-0 ${colIndex < numPredictors ? 'border-r border-slate-700/50' : ''}`}>
                            <input
                              id={`glm-cell-${rowIndex}-${colIndex}`}
                              type="text"
                              value={cell}
                              onChange={(e) => handleCellChange(rowIndex, colIndex, e.target.value)}
                              onKeyDown={(e) => handleKeyDown(e, rowIndex, colIndex)}
                              onPaste={colIndex === 0 && rowIndex === 0 ? handlePaste : undefined}
                              className={`w-full px-3 py-2 bg-transparent focus:bg-slate-700/50 focus:outline-none focus:ring-1 focus:ring-purple-500/50 font-mono text-sm ${
                                colIndex === numPredictors ? 'text-purple-300' : 'text-gray-100'
                              }`}
                              placeholder={rowIndex === 0 ? (colIndex === numPredictors ? 'Response' : 'Predictor') : ''}
                            />
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="mt-4 text-sm text-gray-400">
                {parseTableData().yData.length} complete observations
              </div>
            </div>
          </div>
        )}

        {/* Model Tab */}
        {activeTab === 'model' && (
          <div className="space-y-6">
            {/* Family Selection */}
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <h2 className="text-2xl font-bold mb-4 text-gray-100">Select Distribution Family</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {families.map(family => (
                  <button
                    key={family.id}
                    onClick={() => {
                      setSelectedFamily(family.id)
                      setSelectedLink('')
                    }}
                    className={`p-4 rounded-lg border-2 text-left transition-all duration-200 ${
                      selectedFamily === family.id
                        ? 'border-purple-500 bg-purple-900/30'
                        : 'border-slate-600 bg-slate-700/30 hover:border-slate-500'
                    }`}
                  >
                    <h3 className="font-bold text-lg text-gray-100">{family.name}</h3>
                    <p className="text-purple-400 text-sm mb-1">{family.description}</p>
                    <p className="text-gray-400 text-xs">{family.typical_use}</p>
                    <p className="text-gray-500 text-xs mt-2">Response: {family.response_type}</p>
                  </button>
                ))}
              </div>
            </div>

            {/* Link Function */}
            {selectedFamilyInfo.available_links && (
              <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
                <h2 className="text-2xl font-bold mb-4 text-gray-100">Link Function</h2>
                <div className="flex flex-wrap gap-2">
                  <button
                    onClick={() => setSelectedLink('')}
                    className={`px-4 py-2 rounded-lg transition-colors ${
                      selectedLink === ''
                        ? 'bg-purple-600 text-white'
                        : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
                    }`}
                  >
                    Default ({selectedFamilyInfo.default_link})
                  </button>
                  {selectedFamilyInfo.available_links.map(link => (
                    <button
                      key={link}
                      onClick={() => setSelectedLink(link)}
                      className={`px-4 py-2 rounded-lg transition-colors ${
                        selectedLink === link
                          ? 'bg-purple-600 text-white'
                          : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
                      }`}
                    >
                      {link}
                    </button>
                  ))}
                </div>
                <p className="text-gray-400 text-sm mt-2">
                  Variance function: {selectedFamilyInfo.variance_function}
                </p>
              </div>
            )}

            {/* Model Options */}
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <h2 className="text-2xl font-bold mb-4 text-gray-100">Model Options</h2>
              <label className="flex items-center space-x-2 text-gray-300">
                <input
                  type="checkbox"
                  checked={addIntercept}
                  onChange={(e) => setAddIntercept(e.target.checked)}
                  className="w-4 h-4 text-purple-600 bg-slate-700 border-slate-600 rounded focus:ring-purple-500"
                />
                <span>Include intercept term</span>
              </label>
            </div>

            {/* Action Buttons */}
            <div className="flex flex-wrap gap-4 justify-center">
              <button
                onClick={handleFitModel}
                disabled={loading}
                className="px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white rounded-xl font-bold text-lg shadow-lg hover:shadow-purple-500/50 transition-all duration-200 disabled:opacity-50"
              >
                {loading ? 'Fitting Model...' : 'Fit GLM'}
              </button>
              <button
                onClick={handleCompareModels}
                disabled={loading}
                className="px-6 py-4 bg-slate-700 hover:bg-slate-600 text-white rounded-xl font-semibold transition-colors disabled:opacity-50"
              >
                Compare Families
              </button>
              <button
                onClick={handleRunDiagnostics}
                disabled={loading}
                className="px-6 py-4 bg-slate-700 hover:bg-slate-600 text-white rounded-xl font-semibold transition-colors disabled:opacity-50"
              >
                Run Diagnostics
              </button>
            </div>

            {/* Compare Results */}
            {compareResult && (
              <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
                <h2 className="text-2xl font-bold mb-4 text-gray-100">Family Comparison</h2>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-slate-600">
                        <th className="px-4 py-2 text-left text-gray-400">Family</th>
                        <th className="px-4 py-2 text-right text-gray-400">AIC</th>
                        <th className="px-4 py-2 text-right text-gray-400">BIC</th>
                        <th className="px-4 py-2 text-right text-gray-400">Pseudo R²</th>
                        <th className="px-4 py-2 text-right text-gray-400">ΔAIC</th>
                        <th className="px-4 py-2 text-center text-gray-400">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {compareResult.results.map((r, i) => (
                        <tr key={i} className={`border-b border-slate-700/50 ${r.family === compareResult.best_family ? 'bg-green-900/20' : ''}`}>
                          <td className="px-4 py-2 text-white font-medium">
                            {r.family_name}
                            {r.family === compareResult.best_family && (
                              <span className="ml-2 text-xs bg-green-600 px-2 py-0.5 rounded">Best</span>
                            )}
                          </td>
                          <td className="px-4 py-2 text-right text-gray-300">
                            {r.aic?.toFixed(2) ?? 'N/A'}
                          </td>
                          <td className="px-4 py-2 text-right text-gray-300">
                            {r.bic?.toFixed(2) ?? 'N/A'}
                          </td>
                          <td className="px-4 py-2 text-right text-gray-300">
                            {r.pseudo_r_squared?.toFixed(4) ?? 'N/A'}
                          </td>
                          <td className="px-4 py-2 text-right text-gray-300">
                            {r.delta_aic?.toFixed(2) ?? 'N/A'}
                          </td>
                          <td className="px-4 py-2 text-center">
                            {r.converged ? (
                              <CheckCircle size={16} className="inline text-green-400" />
                            ) : (
                              <AlertTriangle size={16} className="inline text-red-400" />
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="text-gray-400 mt-4">{compareResult.recommendation}</p>
              </div>
            )}
          </div>
        )}

        {/* Results Tab */}
        {activeTab === 'results' && fitResult && (
          <div className="space-y-6">
            {/* Model Summary */}
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <h2 className="text-2xl font-bold mb-4 text-gray-100">
                {fitResult.family_info?.name || fitResult.family} Model Results
              </h2>
              <p className="text-purple-400 mb-4">{fitResult.interpretation}</p>

              {/* Fit Statistics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div className="bg-slate-700/30 rounded-lg p-4">
                  <p className="text-gray-400 text-sm">Pseudo R²</p>
                  <p className="text-2xl font-bold text-green-400">
                    {fitResult.statistics.pseudo_r_squared?.toFixed(4) ?? 'N/A'}
                  </p>
                </div>
                <div className="bg-slate-700/30 rounded-lg p-4">
                  <p className="text-gray-400 text-sm">AIC</p>
                  <p className="text-2xl font-bold text-white">
                    {fitResult.statistics.aic?.toFixed(2) ?? 'N/A'}
                  </p>
                </div>
                <div className="bg-slate-700/30 rounded-lg p-4">
                  <p className="text-gray-400 text-sm">BIC</p>
                  <p className="text-2xl font-bold text-white">
                    {fitResult.statistics.bic?.toFixed(2) ?? 'N/A'}
                  </p>
                </div>
                <div className="bg-slate-700/30 rounded-lg p-4">
                  <p className="text-gray-400 text-sm">Deviance</p>
                  <p className="text-2xl font-bold text-white">
                    {fitResult.statistics.deviance?.toFixed(2) ?? 'N/A'}
                  </p>
                </div>
              </div>

              {/* Coefficients Table */}
              <h3 className="text-xl font-bold mb-3 text-gray-100">Coefficients</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-600">
                      <th className="px-4 py-2 text-left text-gray-400">Parameter</th>
                      <th className="px-4 py-2 text-right text-gray-400">Estimate</th>
                      <th className="px-4 py-2 text-right text-gray-400">Std Error</th>
                      <th className="px-4 py-2 text-right text-gray-400">z-value</th>
                      <th className="px-4 py-2 text-right text-gray-400">p-value</th>
                      {fitResult.link?.toLowerCase() === 'log' || fitResult.link?.toLowerCase() === 'logit' ? (
                        <th className="px-4 py-2 text-right text-gray-400">
                          {fitResult.link.toLowerCase() === 'logit' ? 'Odds Ratio' : 'exp(β)'}
                        </th>
                      ) : null}
                      <th className="px-4 py-2 text-center text-gray-400">Sig.</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(fitResult.coefficients).map(([name, data]) => (
                      <tr key={name} className="border-b border-slate-700/50">
                        <td className="px-4 py-2 text-white font-medium">{name}</td>
                        <td className="px-4 py-2 text-right text-gray-300">
                          {data.estimate?.toFixed(4)}
                        </td>
                        <td className="px-4 py-2 text-right text-gray-300">
                          {data.std_error?.toFixed(4)}
                        </td>
                        <td className="px-4 py-2 text-right text-gray-300">
                          {data.z_value?.toFixed(3)}
                        </td>
                        <td className="px-4 py-2 text-right text-gray-300">
                          {data.p_value?.toExponential(3)}
                        </td>
                        {(fitResult.link?.toLowerCase() === 'log' || fitResult.link?.toLowerCase() === 'logit') && (
                          <td className="px-4 py-2 text-right text-purple-300">
                            {data.exp_estimate?.toFixed(3)}
                          </td>
                        )}
                        <td className="px-4 py-2 text-center">
                          {data.significant ? (
                            <span className="text-green-400">✓</span>
                          ) : (
                            <span className="text-gray-500">-</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Interpretations */}
              <div className="mt-6">
                <h3 className="text-lg font-bold mb-2 text-gray-100">Interpretations</h3>
                <div className="space-y-2">
                  {Object.entries(fitResult.coefficients).map(([name, data]) => (
                    <p key={name} className="text-gray-400 text-sm">
                      <span className="text-white font-medium">{name}:</span> {data.interpretation}
                    </p>
                  ))}
                </div>
              </div>
            </div>

            {/* Fitted vs Observed Plot */}
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <h2 className="text-xl font-bold mb-4 text-gray-100">Fitted vs Observed</h2>
              <Plot
                data={[
                  {
                    x: fitResult.fitted_values,
                    y: fitResult.y_data,
                    type: 'scatter',
                    mode: 'markers',
                    marker: { size: 8, color: '#a855f7' },
                    name: 'Observations'
                  },
                  {
                    x: [Math.min(...fitResult.fitted_values), Math.max(...fitResult.fitted_values)],
                    y: [Math.min(...fitResult.fitted_values), Math.max(...fitResult.fitted_values)],
                    type: 'scatter',
                    mode: 'lines',
                    line: { color: '#ef4444', dash: 'dash' },
                    name: 'Perfect Fit'
                  }
                ]}
                layout={{
                  xaxis: { title: 'Fitted Values', gridcolor: '#334155' },
                  yaxis: { title: 'Observed Values', gridcolor: '#334155' },
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(15,23,42,0.5)',
                  font: { color: '#e2e8f0' },
                  autosize: true,
                  legend: { x: 0.02, y: 0.98 }
                }}
                style={{ width: '100%', height: '400px' }}
                useResizeHandler={true}
              />
            </div>

            {/* Residual Plots */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
                <h2 className="text-xl font-bold mb-4 text-gray-100">Deviance Residuals vs Fitted</h2>
                <Plot
                  data={[
                    {
                      x: fitResult.fitted_values,
                      y: fitResult.residuals.deviance,
                      type: 'scatter',
                      mode: 'markers',
                      marker: { size: 8, color: '#a855f7' },
                      name: 'Residuals'
                    },
                    {
                      x: [Math.min(...fitResult.fitted_values), Math.max(...fitResult.fitted_values)],
                      y: [0, 0],
                      type: 'scatter',
                      mode: 'lines',
                      line: { color: '#ef4444', dash: 'dash' },
                      name: 'Zero'
                    }
                  ]}
                  layout={{
                    xaxis: { title: 'Fitted Values', gridcolor: '#334155' },
                    yaxis: { title: 'Deviance Residuals', gridcolor: '#334155' },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(15,23,42,0.5)',
                    font: { color: '#e2e8f0' },
                    autosize: true,
                    showlegend: false
                  }}
                  style={{ width: '100%', height: '350px' }}
                  useResizeHandler={true}
                />
              </div>

              <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
                <h2 className="text-xl font-bold mb-4 text-gray-100">Pearson Residuals Distribution</h2>
                <Plot
                  data={[
                    {
                      x: fitResult.residuals.pearson,
                      type: 'histogram',
                      marker: { color: '#a855f7' },
                      name: 'Residuals'
                    }
                  ]}
                  layout={{
                    xaxis: { title: 'Pearson Residuals', gridcolor: '#334155' },
                    yaxis: { title: 'Frequency', gridcolor: '#334155' },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(15,23,42,0.5)',
                    font: { color: '#e2e8f0' },
                    autosize: true,
                    showlegend: false
                  }}
                  style={{ width: '100%', height: '350px' }}
                  useResizeHandler={true}
                />
              </div>
            </div>
          </div>
        )}

        {/* Diagnostics Tab */}
        {activeTab === 'diagnostics' && diagnosticsResult && (
          <div className="space-y-6">
            {/* Summary */}
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <h2 className="text-2xl font-bold mb-4 text-gray-100">Diagnostic Summary</h2>
              <div className="grid grid-cols-3 gap-4">
                <div className={`rounded-lg p-4 ${diagnosticsResult.summary.n_outliers > 0 ? 'bg-yellow-900/30 border border-yellow-600' : 'bg-green-900/30 border border-green-600'}`}>
                  <p className="text-gray-300 text-sm">Potential Outliers</p>
                  <p className="text-2xl font-bold text-white">{diagnosticsResult.summary.n_outliers}</p>
                </div>
                <div className={`rounded-lg p-4 ${diagnosticsResult.summary.n_high_leverage > 0 ? 'bg-yellow-900/30 border border-yellow-600' : 'bg-green-900/30 border border-green-600'}`}>
                  <p className="text-gray-300 text-sm">High Leverage Points</p>
                  <p className="text-2xl font-bold text-white">{diagnosticsResult.summary.n_high_leverage}</p>
                </div>
                <div className={`rounded-lg p-4 ${diagnosticsResult.summary.n_influential > 0 ? 'bg-red-900/30 border border-red-600' : 'bg-green-900/30 border border-green-600'}`}>
                  <p className="text-gray-300 text-sm">Influential Points</p>
                  <p className="text-2xl font-bold text-white">{diagnosticsResult.summary.n_influential}</p>
                </div>
              </div>
            </div>

            {/* Dispersion Test */}
            {diagnosticsResult.diagnostics.dispersion_test && (
              <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
                <h2 className="text-xl font-bold mb-4 text-gray-100">Dispersion Test (Poisson)</h2>
                <div className="grid grid-cols-3 gap-4 mb-4">
                  <div className="bg-slate-700/30 rounded-lg p-3">
                    <p className="text-gray-400 text-sm">Pearson χ²</p>
                    <p className="text-xl font-bold text-white">
                      {diagnosticsResult.diagnostics.dispersion_test.pearson_chi2?.toFixed(2)}
                    </p>
                  </div>
                  <div className="bg-slate-700/30 rounded-lg p-3">
                    <p className="text-gray-400 text-sm">Degrees of Freedom</p>
                    <p className="text-xl font-bold text-white">
                      {diagnosticsResult.diagnostics.dispersion_test.df}
                    </p>
                  </div>
                  <div className="bg-slate-700/30 rounded-lg p-3">
                    <p className="text-gray-400 text-sm">Dispersion Ratio</p>
                    <p className={`text-xl font-bold ${
                      diagnosticsResult.diagnostics.dispersion_test.dispersion_ratio > 1.5 ? 'text-yellow-400' : 'text-green-400'
                    }`}>
                      {diagnosticsResult.diagnostics.dispersion_test.dispersion_ratio?.toFixed(3)}
                    </p>
                  </div>
                </div>
                <p className="text-gray-300">{diagnosticsResult.diagnostics.dispersion_test.interpretation}</p>
              </div>
            )}

            {/* Link Test */}
            {diagnosticsResult.diagnostics.link_test && (
              <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
                <h2 className="text-xl font-bold mb-4 text-gray-100">Link Function Test</h2>
                <p className="text-gray-300">{diagnosticsResult.diagnostics.link_test.interpretation}</p>
              </div>
            )}

            {/* Influence Plots */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
                <h2 className="text-xl font-bold mb-4 text-gray-100">Leverage (Hat Values)</h2>
                <Plot
                  data={[
                    {
                      y: diagnosticsResult.influence.hat_values,
                      type: 'bar',
                      marker: {
                        color: diagnosticsResult.influence.hat_values.map(h =>
                          h > diagnosticsResult.influence.leverage_threshold ? '#ef4444' : '#a855f7'
                        )
                      }
                    },
                    {
                      y: Array(diagnosticsResult.influence.hat_values.length).fill(diagnosticsResult.influence.leverage_threshold),
                      type: 'scatter',
                      mode: 'lines',
                      line: { color: '#ef4444', dash: 'dash' },
                      name: 'Threshold'
                    }
                  ]}
                  layout={{
                    xaxis: { title: 'Observation', gridcolor: '#334155' },
                    yaxis: { title: 'Leverage', gridcolor: '#334155' },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(15,23,42,0.5)',
                    font: { color: '#e2e8f0' },
                    autosize: true,
                    showlegend: false
                  }}
                  style={{ width: '100%', height: '350px' }}
                  useResizeHandler={true}
                />
              </div>

              <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
                <h2 className="text-xl font-bold mb-4 text-gray-100">Cook's Distance</h2>
                <Plot
                  data={[
                    {
                      y: diagnosticsResult.influence.cooks_distance,
                      type: 'bar',
                      marker: {
                        color: diagnosticsResult.influence.cooks_distance.map(c =>
                          c > diagnosticsResult.influence.cooks_threshold ? '#ef4444' : '#a855f7'
                        )
                      }
                    },
                    {
                      y: Array(diagnosticsResult.influence.cooks_distance.length).fill(diagnosticsResult.influence.cooks_threshold),
                      type: 'scatter',
                      mode: 'lines',
                      line: { color: '#ef4444', dash: 'dash' },
                      name: 'Threshold'
                    }
                  ]}
                  layout={{
                    xaxis: { title: 'Observation', gridcolor: '#334155' },
                    yaxis: { title: "Cook's Distance", gridcolor: '#334155' },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(15,23,42,0.5)',
                    font: { color: '#e2e8f0' },
                    autosize: true,
                    showlegend: false
                  }}
                  style={{ width: '100%', height: '350px' }}
                  useResizeHandler={true}
                />
              </div>
            </div>

            {/* Problem Points Tables */}
            {diagnosticsResult.diagnostics.influential_points.length > 0 && (
              <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-red-700/50">
                <h2 className="text-xl font-bold mb-4 text-red-400">Influential Points</h2>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-slate-600">
                        <th className="px-4 py-2 text-left text-gray-400">Index</th>
                        <th className="px-4 py-2 text-right text-gray-400">Value</th>
                        <th className="px-4 py-2 text-right text-gray-400">Cook's D</th>
                        <th className="px-4 py-2 text-right text-gray-400">Threshold</th>
                      </tr>
                    </thead>
                    <tbody>
                      {diagnosticsResult.diagnostics.influential_points.map((p, i) => (
                        <tr key={i} className="border-b border-slate-700/50">
                          <td className="px-4 py-2 text-white">{p.index + 1}</td>
                          <td className="px-4 py-2 text-right text-gray-300">{p.value?.toFixed(4)}</td>
                          <td className="px-4 py-2 text-right text-red-400">{p.cooks_d?.toFixed(4)}</td>
                          <td className="px-4 py-2 text-right text-gray-400">{p.threshold?.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Info Tab */}
        {activeTab === 'info' && (
          <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
            <h2 className="text-2xl font-bold mb-4 text-gray-100">About Generalized Linear Models</h2>
            <div className="space-y-4 text-gray-300">
              <p>
                Generalized Linear Models (GLMs) extend linear regression to response variables
                that follow distributions other than normal. A GLM has three components:
              </p>

              <ol className="list-decimal list-inside space-y-2 ml-4">
                <li><strong className="text-white">Random Component:</strong> The probability distribution of the response (e.g., Poisson, Binomial)</li>
                <li><strong className="text-white">Systematic Component:</strong> The linear predictor η = β₀ + β₁X₁ + ...</li>
                <li><strong className="text-white">Link Function:</strong> g(μ) = η, connecting the mean to the linear predictor</li>
              </ol>

              <h3 className="text-xl font-bold text-white mt-6">Distribution Families</h3>
              <ul className="list-disc list-inside space-y-2 ml-4">
                <li><strong className="text-purple-400">Poisson:</strong> Count data (0, 1, 2, ...). Mean = Variance. Use log link.</li>
                <li><strong className="text-purple-400">Binomial:</strong> Binary (0/1) or proportion data. Use logit link for logistic regression.</li>
                <li><strong className="text-purple-400">Negative Binomial:</strong> Overdispersed counts (variance &gt; mean). Like Poisson but allows extra variation.</li>
                <li><strong className="text-purple-400">Gamma:</strong> Positive continuous data, often right-skewed (times, costs).</li>
                <li><strong className="text-purple-400">Gaussian:</strong> Regular normal distribution. GLM with identity link = linear regression.</li>
              </ul>

              <h3 className="text-xl font-bold text-white mt-6">Link Functions</h3>
              <ul className="list-disc list-inside space-y-2 ml-4">
                <li><strong>Log:</strong> g(μ) = log(μ). For positive responses. exp(β) = multiplicative effect.</li>
                <li><strong>Logit:</strong> g(μ) = log(μ/(1-μ)). For proportions. exp(β) = odds ratio.</li>
                <li><strong>Identity:</strong> g(μ) = μ. Direct linear relationship.</li>
                <li><strong>Inverse:</strong> g(μ) = 1/μ. For Gamma with inverse link.</li>
                <li><strong>Probit:</strong> g(μ) = Φ⁻¹(μ). Alternative to logit for binary data.</li>
              </ul>

              <h3 className="text-xl font-bold text-white mt-6">Model Selection</h3>
              <p>
                Use AIC (Akaike Information Criterion) or BIC to compare models:
              </p>
              <ul className="list-disc list-inside space-y-2 ml-4">
                <li>Lower AIC/BIC indicates better fit</li>
                <li>ΔAIC &lt; 2: Models are essentially equivalent</li>
                <li>ΔAIC 2-10: Some evidence for better model</li>
                <li>ΔAIC &gt; 10: Strong evidence for better model</li>
              </ul>

              <h3 className="text-xl font-bold text-white mt-6">Diagnostics</h3>
              <ul className="list-disc list-inside space-y-2 ml-4">
                <li><strong>Deviance Residuals:</strong> Should be randomly scattered around zero</li>
                <li><strong>Pearson Residuals:</strong> Approximately normal for good fit</li>
                <li><strong>Cook's Distance:</strong> Identifies influential observations</li>
                <li><strong>Dispersion Test:</strong> For Poisson, checks if variance = mean</li>
              </ul>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default GLM
