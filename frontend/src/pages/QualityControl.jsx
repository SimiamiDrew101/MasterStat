import { useState, useCallback } from 'react'
import axios from 'axios'
import Plot from 'react-plotly.js'
import { Upload, BarChart3, TrendingUp, Info, AlertTriangle, Clipboard, Gauge, Users } from 'lucide-react'
import * as XLSX from 'xlsx'

const API_URL = import.meta.env.VITE_API_URL || ''

const QualityControl = () => {
  // State
  const [activeTab, setActiveTab] = useState('data')
  const [data, setData] = useState([])
  // Excel-like table data (single column for measurements)
  const [tableData, setTableData] = useState(
    Array(20).fill(null).map(() => [''])
  )
  const [chartType, setChartType] = useState('i-mr')
  const [subgroupSize, setSubgroupSize] = useState(5)
  const [chartResult, setChartResult] = useState(null)
  const [capabilityResult, setCapabilityResult] = useState(null)
  const [specLimits, setSpecLimits] = useState({ lsl: '', usl: '', target: '' })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  // CUSUM/EWMA parameters
  const [cusumParams, setCusumParams] = useState({ k: 0.5, h: 5.0, target: '', sigma: '' })
  const [ewmaParams, setEwmaParams] = useState({ lambda: 0.2, L: 3.0, target: '', sigma: '' })

  // MSA state
  const [msaType, setMsaType] = useState('gauge-rr-crossed')
  const [msaData, setMsaData] = useState({ operators: 3, parts: 10, replicates: 2 })
  const [msaTableData, setMsaTableData] = useState([])
  const [msaResult, setMsaResult] = useState(null)
  const [msaTolerance, setMsaTolerance] = useState('')

  // Excel-like table cell change handler
  const handleCellChange = useCallback((rowIndex, value) => {
    setTableData(prev => {
      const newData = prev.map(row => [...row])
      newData[rowIndex][0] = value

      // Auto-add row if typing in last row
      if (rowIndex === newData.length - 1 && value.trim() !== '') {
        newData.push([''])
      }
      return newData
    })
  }, [])

  // Arrow key navigation handler
  const handleKeyDown = useCallback((e, rowIndex) => {
    const numRows = tableData.length
    let newRow = rowIndex

    switch (e.key) {
      case 'ArrowUp':
        e.preventDefault()
        newRow = Math.max(0, rowIndex - 1)
        break
      case 'ArrowDown':
      case 'Enter':
        e.preventDefault()
        newRow = Math.min(numRows - 1, rowIndex + 1)
        // Add new row if at end and pressing down/enter
        if (rowIndex === numRows - 1) {
          setTableData(prev => [...prev, ['']])
          newRow = numRows
        }
        break
      case 'Tab':
        // Allow default tab behavior to move between elements
        return
      default:
        return
    }

    // Focus the new cell after a brief delay to allow state update
    setTimeout(() => {
      const input = document.getElementById(`qc-cell-${newRow}`)
      if (input) {
        input.focus()
        input.select()
      }
    }, 0)
  }, [tableData.length])

  // Handle paste from clipboard
  const handlePaste = useCallback((e) => {
    e.preventDefault()
    const pastedData = e.clipboardData.getData('text')
    const lines = pastedData.trim().split('\n')
    const newTableData = []

    lines.forEach(line => {
      const value = line.trim().split(/[\s,\t]+/)[0] // Take first value if multiple columns
      if (value) {
        newTableData.push([value])
      }
    })

    // Ensure at least 20 rows
    while (newTableData.length < 20) {
      newTableData.push([''])
    }
    // Add extra empty row at end
    newTableData.push([''])

    setTableData(newTableData)
  }, [])

  // Parse table data to numeric array
  const parseTableData = useCallback(() => {
    const parsedData = []
    tableData.forEach(row => {
      const value = parseFloat(row[0])
      if (!isNaN(value)) {
        parsedData.push(value)
      }
    })
    return parsedData
  }, [tableData])

  // Apply data from table
  const handleApplyData = () => {
    const parsedData = parseTableData()
    if (parsedData.length < 3) {
      setError('Need at least 3 data points')
      return
    }
    setData(parsedData)
    setError('')
    setActiveTab('chart')
  }

  // Chart type information
  const chartTypes = [
    {
      id: 'i-mr',
      name: 'I-MR Chart',
      description: 'Individuals and Moving Range',
      icon: '📊',
      useCase: 'Individual measurements (subgroup size = 1)',
      dataType: 'Continuous',
      needsSubgroup: false
    },
    {
      id: 'xbar-r',
      name: 'X̄-R Chart',
      description: 'X-bar and Range',
      icon: '📈',
      useCase: 'Subgrouped data (2-10 samples)',
      dataType: 'Continuous',
      needsSubgroup: true
    },
    {
      id: 'xbar-s',
      name: 'X̄-S Chart',
      description: 'X-bar and Standard Deviation',
      icon: '📉',
      useCase: 'Larger subgroups (>10 samples)',
      dataType: 'Continuous',
      needsSubgroup: true
    },
    {
      id: 'p',
      name: 'P Chart',
      description: 'Proportion Defective',
      icon: '🎯',
      useCase: 'Fraction defective items',
      dataType: 'Attribute',
      needsSubgroup: true
    },
    {
      id: 'c',
      name: 'C Chart',
      description: 'Count of Defects',
      icon: '🔢',
      useCase: 'Number of defects per unit',
      dataType: 'Attribute',
      needsSubgroup: false
    },
    {
      id: 'cusum',
      name: 'CUSUM Chart',
      description: 'Cumulative Sum',
      icon: '📊',
      useCase: 'Detect small, persistent shifts',
      dataType: 'Continuous',
      needsSubgroup: false,
      advanced: true
    },
    {
      id: 'ewma',
      name: 'EWMA Chart',
      description: 'Exponentially Weighted Moving Average',
      icon: '📈',
      useCase: 'Detect small shifts, smooth response',
      dataType: 'Continuous',
      needsSubgroup: false,
      advanced: true
    }
  ]

  // Load example data
  const loadExampleData = (exampleType) => {
    let exampleData, exampleSubgroup

    if (exampleType === 'stable') {
      // Stable process data
      exampleData = [20.1, 19.8, 20.3, 19.9, 20.2, 20.0, 19.7, 20.1, 20.4, 19.8, 20.2, 19.9, 20.0, 20.1, 19.9, 20.3, 19.8, 20.2, 20.0, 19.9]
      exampleSubgroup = 1
      setChartType('i-mr')
    } else if (exampleType === 'outofcontrol') {
      // Process with special cause variation
      exampleData = [20.0, 19.9, 20.1, 19.8, 20.2, 22.5, 20.0, 19.9, 20.1, 19.8, 20.0, 19.9, 18.2, 20.1, 19.9, 20.0, 19.8, 20.2, 20.0, 19.9]
      exampleSubgroup = 1
      setChartType('i-mr')
    } else if (exampleType === 'trending') {
      // Process with trend
      exampleData = [19.0, 19.2, 19.4, 19.6, 19.8, 20.0, 20.2, 20.4, 20.6, 20.8, 21.0, 21.2, 21.4, 21.6, 21.8, 22.0, 22.2, 22.4, 22.6, 22.8]
      exampleSubgroup = 1
      setChartType('i-mr')
    } else if (exampleType === 'subgroups') {
      // Subgrouped data for Xbar-R chart
      exampleData = [19.8, 20.2, 20.0, 19.9, 20.1, 19.7, 20.3, 19.9, 20.0, 20.2, 20.1, 19.8, 20.2, 20.0, 19.9, 20.3, 19.8, 20.1, 20.0, 19.9, 20.2, 20.0, 19.8, 20.1, 19.9]
      exampleSubgroup = 5
      setChartType('xbar-r')
    }

    // Populate table data
    const newTableData = exampleData.map(val => [val.toString()])
    newTableData.push(['']) // Add empty row at end
    setTableData(newTableData)

    setData(exampleData)
    setSubgroupSize(exampleSubgroup)
    setActiveTab('chart')
  }

  // Upload Excel/CSV file
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

        const parsedData = []
        const newTableData = []
        jsonData.forEach((row, i) => {
          if (i === 0) return // Skip header
          const value = parseFloat(row[0])
          if (!isNaN(value)) {
            parsedData.push(value)
            newTableData.push([value.toString()])
          }
        })

        if (parsedData.length < 3) {
          setError('Need at least 3 valid data points in file')
          return
        }

        // Ensure minimum rows and add empty row at end
        while (newTableData.length < 20) {
          newTableData.push([''])
        }
        newTableData.push([''])

        setTableData(newTableData)
        setData(parsedData)
        setError('')
        setActiveTab('chart')
      } catch (err) {
        setError('Failed to read file: ' + err.message)
      }
    }

    reader.readAsArrayBuffer(file)
  }

  // Generate control chart
  const handleGenerateChart = async () => {
    if (data.length < 3) {
      setError('Need at least 3 data points')
      return
    }

    setLoading(true)
    setError('')

    try {
      let response

      if (chartType === 'cusum') {
        const requestData = {
          data: data,
          k: cusumParams.k,
          h: cusumParams.h
        }
        if (cusumParams.target) requestData.target = parseFloat(cusumParams.target)
        if (cusumParams.sigma) requestData.sigma = parseFloat(cusumParams.sigma)

        response = await axios.post(`${API_URL}/api/quality-control/cusum`, requestData)
      } else if (chartType === 'ewma') {
        const requestData = {
          data: data,
          lambda_: ewmaParams.lambda,
          L: ewmaParams.L
        }
        if (ewmaParams.target) requestData.target = parseFloat(ewmaParams.target)
        if (ewmaParams.sigma) requestData.sigma = parseFloat(ewmaParams.sigma)

        response = await axios.post(`${API_URL}/api/quality-control/ewma`, requestData)
      } else {
        const requestData = {
          data: data,
          chart_type: chartType
        }

        const selectedChart = chartTypes.find(c => c.id === chartType)
        if (selectedChart && selectedChart.needsSubgroup) {
          requestData.subgroup_size = subgroupSize
        }

        response = await axios.post(`${API_URL}/api/quality-control/control-chart`, requestData)
      }

      setChartResult(response.data)
      setActiveTab('results')
    } catch (err) {
      setError('Chart generation failed: ' + (err.response?.data?.detail || err.message))
    } finally {
      setLoading(false)
    }
  }

  // Calculate capability
  const handleCalculateCapability = async () => {
    if (data.length < 3) {
      setError('Need at least 3 data points')
      return
    }

    const lsl = parseFloat(specLimits.lsl)
    const usl = parseFloat(specLimits.usl)

    if (isNaN(lsl) && isNaN(usl)) {
      setError('Need at least one specification limit (LSL or USL)')
      return
    }

    setLoading(true)
    setError('')

    try {
      const requestData = {
        data: data,
        spec_limits: {}
      }

      if (!isNaN(lsl)) requestData.spec_limits.lsl = lsl
      if (!isNaN(usl)) requestData.spec_limits.usl = usl

      const target = parseFloat(specLimits.target)
      if (!isNaN(target)) {
        requestData.target = target
      }

      const response = await axios.post(`${API_URL}/api/quality-control/capability`, requestData)
      setCapabilityResult(response.data)
    } catch (err) {
      setError('Capability calculation failed: ' + (err.response?.data?.detail || err.message))
    } finally {
      setLoading(false)
    }
  }

  // Get violation color
  const getViolationColor = (severity) => {
    if (severity === 'critical') return '#ef4444'
    if (severity === 'warning') return '#f59e0b'
    return '#3b82f6'
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-pink-600 text-transparent bg-clip-text">
            Quality Control & SPC Charts
          </h1>
          <p className="text-gray-300 text-lg">
            Statistical Process Control charts and capability analysis for manufacturing quality
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
            onClick={() => setActiveTab('chart')}
            disabled={data.length === 0}
            className={`flex-1 px-4 py-3 rounded-lg font-medium transition-all duration-200 flex items-center justify-center space-x-2 ${
              activeTab === 'chart'
                ? 'bg-purple-600 text-white shadow-lg'
                : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700 disabled:opacity-40'
            }`}
          >
            <BarChart3 size={20} />
            <span>2. Chart Type</span>
          </button>
          <button
            onClick={() => setActiveTab('results')}
            disabled={!chartResult}
            className={`flex-1 px-4 py-3 rounded-lg font-medium transition-all duration-200 flex items-center justify-center space-x-2 ${
              activeTab === 'results'
                ? 'bg-purple-600 text-white shadow-lg'
                : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700 disabled:opacity-40'
            }`}
          >
            <TrendingUp size={20} />
            <span>3. Results</span>
          </button>
          <button
            onClick={() => setActiveTab('capability')}
            disabled={data.length === 0}
            className={`flex-1 px-4 py-3 rounded-lg font-medium transition-all duration-200 flex items-center justify-center space-x-2 ${
              activeTab === 'capability'
                ? 'bg-purple-600 text-white shadow-lg'
                : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700 disabled:opacity-40'
            }`}
          >
            <Gauge size={20} />
            <span>4. Capability</span>
          </button>
          <button
            onClick={() => setActiveTab('msa')}
            className={`flex-1 px-4 py-3 rounded-lg font-medium transition-all duration-200 flex items-center justify-center space-x-2 ${
              activeTab === 'msa'
                ? 'bg-purple-600 text-white shadow-lg'
                : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
            }`}
          >
            <Users size={20} />
            <span>5. MSA</span>
          </button>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-6 bg-red-900/50 border border-red-600 text-red-200 px-4 py-3 rounded-lg flex items-center space-x-2">
            <AlertTriangle size={20} />
            <span>{error}</span>
          </div>
        )}

        {/* Tab Content */}
        {activeTab === 'data' && (
          <div className="space-y-6">
            {/* Example Data */}
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <h2 className="text-2xl font-bold mb-4 text-gray-100">Quick Start: Load Example Data</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <button
                  onClick={() => loadExampleData('stable')}
                  className="px-6 py-4 bg-gradient-to-br from-green-600 to-green-700 hover:from-green-500 hover:to-green-600 text-white rounded-xl font-semibold transition-all duration-200 shadow-lg hover:shadow-green-500/50"
                >
                  Stable Process
                </button>
                <button
                  onClick={() => loadExampleData('outofcontrol')}
                  className="px-6 py-4 bg-gradient-to-br from-red-600 to-red-700 hover:from-red-500 hover:to-red-600 text-white rounded-xl font-semibold transition-all duration-200 shadow-lg hover:shadow-red-500/50"
                >
                  Out of Control
                </button>
                <button
                  onClick={() => loadExampleData('trending')}
                  className="px-6 py-4 bg-gradient-to-br from-orange-600 to-orange-700 hover:from-orange-500 hover:to-orange-600 text-white rounded-xl font-semibold transition-all duration-200 shadow-lg hover:shadow-orange-500/50"
                >
                  Trending Process
                </button>
                <button
                  onClick={() => loadExampleData('subgroups')}
                  className="px-6 py-4 bg-gradient-to-br from-blue-600 to-blue-700 hover:from-blue-500 hover:to-blue-600 text-white rounded-xl font-semibold transition-all duration-200 shadow-lg hover:shadow-blue-500/50"
                >
                  Subgrouped Data
                </button>
              </div>
            </div>

            {/* Manual Data Entry - Excel-like table */}
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <h2 className="text-2xl font-bold mb-4 text-gray-100">Enter Process Data</h2>
              <div className="flex items-center gap-2 mb-3">
                <p className="text-sm text-gray-400">
                  Enter measurements in the table below. Use arrow keys to navigate, or paste data from Excel.
                </p>
                <button
                  onClick={() => {
                    navigator.clipboard.readText().then(text => {
                      const lines = text.trim().split('\n')
                      const newTableData = []
                      lines.forEach(line => {
                        const value = line.trim().split(/[\s,\t]+/)[0]
                        if (value) newTableData.push([value])
                      })
                      while (newTableData.length < 20) newTableData.push([''])
                      newTableData.push([''])
                      setTableData(newTableData)
                    }).catch(() => setError('Failed to read clipboard'))
                  }}
                  className="flex items-center gap-1 px-3 py-1 bg-slate-700 hover:bg-slate-600 text-gray-300 rounded text-sm transition-colors"
                >
                  <Clipboard size={14} />
                  Paste
                </button>
              </div>
              <div className="max-h-80 overflow-y-auto border border-slate-600 rounded-lg">
                <table className="w-full">
                  <thead className="sticky top-0 bg-slate-700">
                    <tr>
                      <th className="w-16 px-3 py-2 text-left text-xs font-semibold text-gray-400 border-b border-slate-600">#</th>
                      <th className="px-3 py-2 text-left text-xs font-semibold text-gray-400 border-b border-slate-600">Measurement</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tableData.map((row, rowIndex) => (
                      <tr key={rowIndex} className="border-b border-slate-700/50 hover:bg-slate-700/30">
                        <td className="px-3 py-1 text-xs text-gray-500 font-mono">{rowIndex + 1}</td>
                        <td className="p-0">
                          <input
                            id={`qc-cell-${rowIndex}`}
                            type="text"
                            value={row[0]}
                            onChange={(e) => handleCellChange(rowIndex, e.target.value)}
                            onKeyDown={(e) => handleKeyDown(e, rowIndex)}
                            onPaste={handlePaste}
                            className="w-full px-3 py-2 bg-transparent text-gray-100 focus:bg-slate-700/50 focus:outline-none focus:ring-1 focus:ring-purple-500/50 font-mono text-sm"
                            placeholder={rowIndex === 0 ? 'Enter value...' : ''}
                          />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="flex items-center gap-4 mt-4">
                <button
                  onClick={handleApplyData}
                  className="px-6 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-semibold transition-colors"
                >
                  Apply Data
                </button>
                <span className="text-sm text-gray-400">
                  {parseTableData().length} valid measurements
                </span>
              </div>
            </div>

            {/* File Upload */}
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <h2 className="text-2xl font-bold mb-4 text-gray-100">Or Upload File</h2>
              <input
                type="file"
                accept=".xlsx,.xls,.csv"
                onChange={handleFileUpload}
                className="block w-full text-sm text-gray-300
                  file:mr-4 file:py-2 file:px-4
                  file:rounded-lg file:border-0
                  file:text-sm file:font-semibold
                  file:bg-purple-600 file:text-white
                  hover:file:bg-purple-700
                  cursor-pointer"
              />
              <p className="text-gray-400 text-sm mt-2">
                Upload Excel or CSV file with measurements in first column
              </p>
            </div>

            {/* Data Preview */}
            {data.length > 0 && (
              <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
                <h2 className="text-2xl font-bold mb-4 text-gray-100">Data Summary</h2>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-slate-700/30 rounded-lg p-4">
                    <p className="text-gray-400 text-sm mb-1">Sample Size</p>
                    <p className="text-2xl font-bold text-white">{data.length}</p>
                  </div>
                  <div className="bg-slate-700/30 rounded-lg p-4">
                    <p className="text-gray-400 text-sm mb-1">Mean</p>
                    <p className="text-2xl font-bold text-white">
                      {(data.reduce((a, b) => a + b, 0) / data.length).toFixed(3)}
                    </p>
                  </div>
                  <div className="bg-slate-700/30 rounded-lg p-4">
                    <p className="text-gray-400 text-sm mb-1">Min</p>
                    <p className="text-2xl font-bold text-white">{Math.min(...data).toFixed(3)}</p>
                  </div>
                  <div className="bg-slate-700/30 rounded-lg p-4">
                    <p className="text-gray-400 text-sm mb-1">Max</p>
                    <p className="text-2xl font-bold text-white">{Math.max(...data).toFixed(3)}</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'chart' && (
          <div className="space-y-6">
            {/* Chart Type Selection */}
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <h2 className="text-2xl font-bold mb-4 text-gray-100">Select Control Chart Type</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {chartTypes.map(chart => (
                  <button
                    key={chart.id}
                    onClick={() => setChartType(chart.id)}
                    className={`p-4 rounded-lg border-2 text-left transition-all duration-200 ${
                      chartType === chart.id
                        ? 'border-purple-500 bg-purple-900/30'
                        : 'border-slate-600 bg-slate-700/30 hover:border-slate-500'
                    }`}
                  >
                    <div className="flex items-center space-x-3 mb-2">
                      <span className="text-3xl">{chart.icon}</span>
                      <h3 className="font-bold text-lg text-gray-100">{chart.name}</h3>
                    </div>
                    <p className="text-purple-400 text-sm font-medium mb-1">{chart.description}</p>
                    <p className="text-gray-400 text-sm mb-1">{chart.useCase}</p>
                    <p className="text-gray-500 text-xs">Data Type: {chart.dataType}</p>
                  </button>
                ))}
              </div>
            </div>

            {/* Subgroup Size */}
            {chartTypes.find(c => c.id === chartType)?.needsSubgroup && (
              <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
                <h2 className="text-2xl font-bold mb-4 text-gray-100">Subgroup Size</h2>
                <div className="max-w-md">
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Number of samples per subgroup
                  </label>
                  <input
                    type="number"
                    min="2"
                    max="25"
                    value={subgroupSize}
                    onChange={(e) => setSubgroupSize(parseInt(e.target.value))}
                    className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                  <p className="text-gray-400 text-sm mt-2">
                    {chartType === 'xbar-r' && 'Recommended: 2-10 samples per subgroup'}
                    {chartType === 'xbar-s' && 'Recommended: 10+ samples per subgroup'}
                    {chartType === 'p' && 'Sample size for proportion calculation'}
                  </p>
                </div>
              </div>
            )}

            {/* CUSUM Parameters */}
            {chartType === 'cusum' && (
              <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
                <h2 className="text-2xl font-bold mb-4 text-gray-100">CUSUM Parameters</h2>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      k (Slack Value)
                    </label>
                    <input
                      type="number"
                      step="0.1"
                      value={cusumParams.k}
                      onChange={(e) => setCusumParams({ ...cusumParams, k: parseFloat(e.target.value) })}
                      className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                    <p className="text-gray-500 text-xs mt-1">Typically 0.5σ</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      h (Decision Interval)
                    </label>
                    <input
                      type="number"
                      step="0.5"
                      value={cusumParams.h}
                      onChange={(e) => setCusumParams({ ...cusumParams, h: parseFloat(e.target.value) })}
                      className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                    <p className="text-gray-500 text-xs mt-1">Typically 4-5σ</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Target (optional)
                    </label>
                    <input
                      type="number"
                      step="any"
                      value={cusumParams.target}
                      onChange={(e) => setCusumParams({ ...cusumParams, target: e.target.value })}
                      placeholder="Auto"
                      className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Sigma (optional)
                    </label>
                    <input
                      type="number"
                      step="any"
                      value={cusumParams.sigma}
                      onChange={(e) => setCusumParams({ ...cusumParams, sigma: e.target.value })}
                      placeholder="Auto"
                      className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                </div>
              </div>
            )}

            {/* EWMA Parameters */}
            {chartType === 'ewma' && (
              <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
                <h2 className="text-2xl font-bold mb-4 text-gray-100">EWMA Parameters</h2>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      λ (Smoothing)
                    </label>
                    <input
                      type="number"
                      step="0.05"
                      min="0.01"
                      max="1"
                      value={ewmaParams.lambda}
                      onChange={(e) => setEwmaParams({ ...ewmaParams, lambda: parseFloat(e.target.value) })}
                      className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                    <p className="text-gray-500 text-xs mt-1">0.05-0.25 typical</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      L (Control Width)
                    </label>
                    <input
                      type="number"
                      step="0.1"
                      value={ewmaParams.L}
                      onChange={(e) => setEwmaParams({ ...ewmaParams, L: parseFloat(e.target.value) })}
                      className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                    <p className="text-gray-500 text-xs mt-1">Typically 2.5-3.0</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Target (optional)
                    </label>
                    <input
                      type="number"
                      step="any"
                      value={ewmaParams.target}
                      onChange={(e) => setEwmaParams({ ...ewmaParams, target: e.target.value })}
                      placeholder="Auto"
                      className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Sigma (optional)
                    </label>
                    <input
                      type="number"
                      step="any"
                      value={ewmaParams.sigma}
                      onChange={(e) => setEwmaParams({ ...ewmaParams, sigma: e.target.value })}
                      placeholder="Auto"
                      className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Generate Button */}
            <div className="flex justify-center">
              <button
                onClick={handleGenerateChart}
                disabled={loading || data.length === 0}
                className="px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white rounded-xl font-bold text-lg shadow-lg hover:shadow-purple-500/50 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? 'Generating Chart...' : 'Generate Control Chart'}
              </button>
            </div>
          </div>
        )}

        {activeTab === 'results' && chartResult && (
          <div className="space-y-6">
            {/* Chart Info */}
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <h2 className="text-2xl font-bold mb-4 text-gray-100">
                {chartTypes.find(c => c.id === chartResult.chart_type)?.name || 'Control Chart'}
              </h2>

              {/* Primary Chart (I, Xbar, P, or C) */}
              {(() => {
                let chartData, chartName, yAxisTitle

                if (chartResult.chart_type === 'i-mr') {
                  chartData = chartResult.i_chart
                  chartName = 'Individuals Chart'
                  yAxisTitle = 'Individual Value'
                } else if (chartResult.chart_type === 'xbar-r' || chartResult.chart_type === 'xbar-s') {
                  chartData = chartResult.xbar_chart
                  chartName = 'X-bar Chart'
                  yAxisTitle = 'Subgroup Mean'
                } else if (chartResult.chart_type === 'p') {
                  chartData = chartResult.p_chart
                  chartName = 'P Chart'
                  yAxisTitle = 'Proportion'
                } else {
                  chartData = chartResult.c_chart
                  chartName = 'C Chart'
                  yAxisTitle = 'Count'
                }

                const xValues = chartData.values.map((_, i) => i + 1)

                // Find violation points
                const violationIndices = chartResult.violations?.map(v => v.point) || []

                return (
                  <div className="mb-6">
                    <h3 className="text-xl font-bold mb-4 text-gray-100">{chartName}</h3>
                    <Plot
                      data={[
                        {
                          x: xValues,
                          y: chartData.values,
                          type: 'scatter',
                          mode: 'lines+markers',
                          marker: {
                            size: 8,
                            color: xValues.map(i => violationIndices.includes(i - 1) ? '#ef4444' : '#a855f7')
                          },
                          line: { color: '#a855f7' },
                          name: 'Measurements'
                        },
                        {
                          x: [1, xValues.length],
                          y: [chartData.center_line, chartData.center_line],
                          type: 'scatter',
                          mode: 'lines',
                          line: { color: '#22c55e', width: 2, dash: 'solid' },
                          name: 'Center Line'
                        },
                        {
                          x: [1, xValues.length],
                          y: [chartData.ucl, chartData.ucl],
                          type: 'scatter',
                          mode: 'lines',
                          line: { color: '#ef4444', width: 2, dash: 'dash' },
                          name: 'UCL'
                        },
                        {
                          x: [1, xValues.length],
                          y: [chartData.lcl, chartData.lcl],
                          type: 'scatter',
                          mode: 'lines',
                          line: { color: '#ef4444', width: 2, dash: 'dash' },
                          name: 'LCL'
                        }
                      ]}
                      layout={{
                        xaxis: { title: 'Sample Number', gridcolor: '#334155' },
                        yaxis: { title: yAxisTitle, gridcolor: '#334155' },
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(15,23,42,0.5)',
                        font: { color: '#e2e8f0' },
                        autosize: true,
                        legend: { x: 0.02, y: 0.98 }
                      }}
                      style={{ width: '100%', height: '500px' }}
                      useResizeHandler={true}
                    />
                    <div className="grid grid-cols-3 gap-4 mt-4">
                      <div className="bg-slate-700/30 rounded-lg p-3">
                        <p className="text-gray-400 text-sm">Center Line</p>
                        <p className="text-xl font-bold text-green-400">{chartData.center_line.toFixed(4)}</p>
                      </div>
                      <div className="bg-slate-700/30 rounded-lg p-3">
                        <p className="text-gray-400 text-sm">UCL</p>
                        <p className="text-xl font-bold text-red-400">{chartData.ucl.toFixed(4)}</p>
                      </div>
                      <div className="bg-slate-700/30 rounded-lg p-3">
                        <p className="text-gray-400 text-sm">LCL</p>
                        <p className="text-xl font-bold text-red-400">{chartData.lcl.toFixed(4)}</p>
                      </div>
                    </div>
                  </div>
                )
              })()}

              {/* Secondary Chart (MR or R or S) */}
              {(chartResult.chart_type === 'i-mr' && chartResult.mr_chart) && (
                <div className="mt-8">
                  <h3 className="text-xl font-bold mb-4 text-gray-100">Moving Range Chart</h3>
                  <Plot
                    data={[
                      {
                        x: chartResult.mr_chart.values.map((_, i) => i + 1),
                        y: chartResult.mr_chart.values,
                        type: 'scatter',
                        mode: 'lines+markers',
                        marker: { size: 8, color: '#a855f7' },
                        line: { color: '#a855f7' },
                        name: 'Moving Range'
                      },
                      {
                        x: [1, chartResult.mr_chart.values.length],
                        y: [chartResult.mr_chart.center_line, chartResult.mr_chart.center_line],
                        type: 'scatter',
                        mode: 'lines',
                        line: { color: '#22c55e', width: 2 },
                        name: 'MR-bar'
                      },
                      {
                        x: [1, chartResult.mr_chart.values.length],
                        y: [chartResult.mr_chart.ucl, chartResult.mr_chart.ucl],
                        type: 'scatter',
                        mode: 'lines',
                        line: { color: '#ef4444', width: 2, dash: 'dash' },
                        name: 'UCL'
                      }
                    ]}
                    layout={{
                      xaxis: { title: 'Sample Number', gridcolor: '#334155' },
                      yaxis: { title: 'Moving Range', gridcolor: '#334155' },
                      paper_bgcolor: 'rgba(0,0,0,0)',
                      plot_bgcolor: 'rgba(15,23,42,0.5)',
                      font: { color: '#e2e8f0' },
                      autosize: true,
                      showlegend: false
                    }}
                    style={{ width: '100%', height: '400px' }}
                    useResizeHandler={true}
                  />
                </div>
              )}

              {(chartResult.r_chart) && (
                <div className="mt-8">
                  <h3 className="text-xl font-bold mb-4 text-gray-100">Range Chart</h3>
                  <Plot
                    data={[
                      {
                        x: chartResult.r_chart.values.map((_, i) => i + 1),
                        y: chartResult.r_chart.values,
                        type: 'scatter',
                        mode: 'lines+markers',
                        marker: { size: 8, color: '#a855f7' },
                        line: { color: '#a855f7' },
                        name: 'Range'
                      },
                      {
                        x: [1, chartResult.r_chart.values.length],
                        y: [chartResult.r_chart.center_line, chartResult.r_chart.center_line],
                        type: 'scatter',
                        mode: 'lines',
                        line: { color: '#22c55e', width: 2 },
                        name: 'R-bar'
                      },
                      {
                        x: [1, chartResult.r_chart.values.length],
                        y: [chartResult.r_chart.ucl, chartResult.r_chart.ucl],
                        type: 'scatter',
                        mode: 'lines',
                        line: { color: '#ef4444', width: 2, dash: 'dash' },
                        name: 'UCL'
                      }
                    ]}
                    layout={{
                      xaxis: { title: 'Subgroup Number', gridcolor: '#334155' },
                      yaxis: { title: 'Range', gridcolor: '#334155' },
                      paper_bgcolor: 'rgba(0,0,0,0)',
                      plot_bgcolor: 'rgba(15,23,42,0.5)',
                      font: { color: '#e2e8f0' },
                      autosize: true,
                      showlegend: false
                    }}
                    style={{ width: '100%', height: '400px' }}
                    useResizeHandler={true}
                  />
                </div>
              )}

              {(chartResult.s_chart) && (
                <div className="mt-8">
                  <h3 className="text-xl font-bold mb-4 text-gray-100">Standard Deviation Chart</h3>
                  <Plot
                    data={[
                      {
                        x: chartResult.s_chart.values.map((_, i) => i + 1),
                        y: chartResult.s_chart.values,
                        type: 'scatter',
                        mode: 'lines+markers',
                        marker: { size: 8, color: '#a855f7' },
                        line: { color: '#a855f7' },
                        name: 'Std Dev'
                      },
                      {
                        x: [1, chartResult.s_chart.values.length],
                        y: [chartResult.s_chart.center_line, chartResult.s_chart.center_line],
                        type: 'scatter',
                        mode: 'lines',
                        line: { color: '#22c55e', width: 2 },
                        name: 'S-bar'
                      },
                      {
                        x: [1, chartResult.s_chart.values.length],
                        y: [chartResult.s_chart.ucl, chartResult.s_chart.ucl],
                        type: 'scatter',
                        mode: 'lines',
                        line: { color: '#ef4444', width: 2, dash: 'dash' },
                        name: 'UCL'
                      }
                    ]}
                    layout={{
                      xaxis: { title: 'Subgroup Number', gridcolor: '#334155' },
                      yaxis: { title: 'Standard Deviation', gridcolor: '#334155' },
                      paper_bgcolor: 'rgba(0,0,0,0)',
                      plot_bgcolor: 'rgba(15,23,42,0.5)',
                      font: { color: '#e2e8f0' },
                      autosize: true,
                      showlegend: false
                    }}
                    style={{ width: '100%', height: '400px' }}
                    useResizeHandler={true}
                  />
                </div>
              )}

              {/* CUSUM Chart */}
              {chartResult.chart_type === 'cusum' && chartResult.cusum_chart && (
                <div className="mt-8">
                  <h3 className="text-xl font-bold mb-4 text-gray-100">CUSUM Chart</h3>
                  <Plot
                    data={[
                      {
                        x: chartResult.cusum_chart.cusum_upper.map((_, i) => i + 1),
                        y: chartResult.cusum_chart.cusum_upper,
                        type: 'scatter',
                        mode: 'lines+markers',
                        marker: { size: 6, color: '#3b82f6' },
                        line: { color: '#3b82f6' },
                        name: 'CUSUM+ (Upper)'
                      },
                      {
                        x: chartResult.cusum_chart.cusum_lower.map((_, i) => i + 1),
                        y: chartResult.cusum_chart.cusum_lower,
                        type: 'scatter',
                        mode: 'lines+markers',
                        marker: { size: 6, color: '#f59e0b' },
                        line: { color: '#f59e0b' },
                        name: 'CUSUM- (Lower)'
                      },
                      {
                        x: [1, chartResult.cusum_chart.cusum_upper.length],
                        y: [chartResult.cusum_chart.h, chartResult.cusum_chart.h],
                        type: 'scatter',
                        mode: 'lines',
                        line: { color: '#ef4444', width: 2, dash: 'dash' },
                        name: 'UCL (+h)'
                      },
                      {
                        x: [1, chartResult.cusum_chart.cusum_upper.length],
                        y: [-chartResult.cusum_chart.h, -chartResult.cusum_chart.h],
                        type: 'scatter',
                        mode: 'lines',
                        line: { color: '#ef4444', width: 2, dash: 'dash' },
                        name: 'LCL (-h)'
                      },
                      {
                        x: [1, chartResult.cusum_chart.cusum_upper.length],
                        y: [0, 0],
                        type: 'scatter',
                        mode: 'lines',
                        line: { color: '#22c55e', width: 1 },
                        name: 'Center'
                      }
                    ]}
                    layout={{
                      xaxis: { title: 'Sample Number', gridcolor: '#334155' },
                      yaxis: { title: 'Cumulative Sum', gridcolor: '#334155' },
                      paper_bgcolor: 'rgba(0,0,0,0)',
                      plot_bgcolor: 'rgba(15,23,42,0.5)',
                      font: { color: '#e2e8f0' },
                      autosize: true,
                      legend: { x: 0.02, y: 0.98 }
                    }}
                    style={{ width: '100%', height: '400px' }}
                    useResizeHandler={true}
                  />
                  <div className="grid grid-cols-4 gap-4 mt-4">
                    <div className="bg-slate-700/30 rounded-lg p-3">
                      <p className="text-gray-400 text-sm">Target</p>
                      <p className="text-lg font-bold text-white">{chartResult.cusum_chart.target?.toFixed(4)}</p>
                    </div>
                    <div className="bg-slate-700/30 rounded-lg p-3">
                      <p className="text-gray-400 text-sm">k (Slack)</p>
                      <p className="text-lg font-bold text-white">{chartResult.cusum_chart.k?.toFixed(2)}σ</p>
                    </div>
                    <div className="bg-slate-700/30 rounded-lg p-3">
                      <p className="text-gray-400 text-sm">h (Threshold)</p>
                      <p className="text-lg font-bold text-white">±{chartResult.cusum_chart.h?.toFixed(2)}σ</p>
                    </div>
                    <div className="bg-slate-700/30 rounded-lg p-3">
                      <p className="text-gray-400 text-sm">Sigma</p>
                      <p className="text-lg font-bold text-white">{chartResult.cusum_chart.sigma?.toFixed(4)}</p>
                    </div>
                  </div>
                  {(chartResult.cusum_chart.upper_violations?.length > 0 || chartResult.cusum_chart.lower_violations?.length > 0) && (
                    <div className="mt-4 p-4 bg-red-900/30 border border-red-700/50 rounded-lg">
                      <p className="text-red-400 font-semibold">
                        Shift detected! Points exceeding threshold: {' '}
                        {[...chartResult.cusum_chart.upper_violations, ...chartResult.cusum_chart.lower_violations].map(p => p + 1).join(', ')}
                      </p>
                    </div>
                  )}
                </div>
              )}

              {/* EWMA Chart */}
              {chartResult.chart_type === 'ewma' && chartResult.ewma_chart && (
                <div className="mt-8">
                  <h3 className="text-xl font-bold mb-4 text-gray-100">EWMA Chart</h3>
                  <Plot
                    data={[
                      {
                        x: chartResult.ewma_chart.ewma.map((_, i) => i + 1),
                        y: chartResult.ewma_chart.ewma,
                        type: 'scatter',
                        mode: 'lines+markers',
                        marker: { size: 6, color: '#a855f7' },
                        line: { color: '#a855f7' },
                        name: 'EWMA'
                      },
                      {
                        x: chartResult.ewma_chart.ucl.map((_, i) => i + 1),
                        y: chartResult.ewma_chart.ucl,
                        type: 'scatter',
                        mode: 'lines',
                        line: { color: '#ef4444', width: 2, dash: 'dash' },
                        name: 'UCL'
                      },
                      {
                        x: chartResult.ewma_chart.lcl.map((_, i) => i + 1),
                        y: chartResult.ewma_chart.lcl,
                        type: 'scatter',
                        mode: 'lines',
                        line: { color: '#ef4444', width: 2, dash: 'dash' },
                        name: 'LCL'
                      },
                      {
                        x: [1, chartResult.ewma_chart.ewma.length],
                        y: [chartResult.ewma_chart.target, chartResult.ewma_chart.target],
                        type: 'scatter',
                        mode: 'lines',
                        line: { color: '#22c55e', width: 2 },
                        name: 'Target'
                      }
                    ]}
                    layout={{
                      xaxis: { title: 'Sample Number', gridcolor: '#334155' },
                      yaxis: { title: 'EWMA Value', gridcolor: '#334155' },
                      paper_bgcolor: 'rgba(0,0,0,0)',
                      plot_bgcolor: 'rgba(15,23,42,0.5)',
                      font: { color: '#e2e8f0' },
                      autosize: true,
                      legend: { x: 0.02, y: 0.98 }
                    }}
                    style={{ width: '100%', height: '400px' }}
                    useResizeHandler={true}
                  />
                  <div className="grid grid-cols-4 gap-4 mt-4">
                    <div className="bg-slate-700/30 rounded-lg p-3">
                      <p className="text-gray-400 text-sm">Target</p>
                      <p className="text-lg font-bold text-white">{chartResult.ewma_chart.target?.toFixed(4)}</p>
                    </div>
                    <div className="bg-slate-700/30 rounded-lg p-3">
                      <p className="text-gray-400 text-sm">λ (Smoothing)</p>
                      <p className="text-lg font-bold text-white">{chartResult.ewma_chart.lambda?.toFixed(2)}</p>
                    </div>
                    <div className="bg-slate-700/30 rounded-lg p-3">
                      <p className="text-gray-400 text-sm">L (Width)</p>
                      <p className="text-lg font-bold text-white">{chartResult.ewma_chart.L?.toFixed(1)}</p>
                    </div>
                    <div className="bg-slate-700/30 rounded-lg p-3">
                      <p className="text-gray-400 text-sm">Asymptotic UCL</p>
                      <p className="text-lg font-bold text-white">{chartResult.ewma_chart.ucl_asymptotic?.toFixed(4)}</p>
                    </div>
                  </div>
                  {chartResult.ewma_chart.violations?.length > 0 && (
                    <div className="mt-4 p-4 bg-red-900/30 border border-red-700/50 rounded-lg">
                      <p className="text-red-400 font-semibold">
                        Out of control at points: {chartResult.ewma_chart.violations.map(p => p + 1).join(', ')}
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Violations */}
            {chartResult.violations && chartResult.violations.length > 0 && (
              <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-red-700/50">
                <h2 className="text-2xl font-bold mb-4 text-red-400 flex items-center space-x-2">
                  <AlertTriangle size={24} />
                  <span>Western Electric Rules Violations ({chartResult.violations.length})</span>
                </h2>
                <div className="space-y-2">
                  {chartResult.violations.map((violation, idx) => (
                    <div
                      key={idx}
                      className="bg-slate-700/30 rounded-lg p-3 border-l-4"
                      style={{ borderLeftColor: getViolationColor(violation.severity) }}
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-semibold text-white">
                            Point {violation.point + 1}: {violation.description}
                          </p>
                          <p className="text-gray-400 text-sm">Rule {violation.rule} violation</p>
                        </div>
                        <span
                          className="px-3 py-1 rounded-full text-xs font-semibold"
                          style={{
                            backgroundColor: getViolationColor(violation.severity) + '30',
                            color: getViolationColor(violation.severity)
                          }}
                        >
                          {violation.severity}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {chartResult.violations && chartResult.violations.length === 0 && (
              <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-green-700/50">
                <h2 className="text-2xl font-bold mb-2 text-green-400 flex items-center space-x-2">
                  <span>✓</span>
                  <span>Process In Statistical Control</span>
                </h2>
                <p className="text-gray-300">
                  No Western Electric rules violations detected. The process appears stable.
                </p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'capability' && (
          <div className="space-y-6">
            {/* Specification Limits Input */}
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <h2 className="text-2xl font-bold mb-4 text-gray-100">Specification Limits</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-3xl">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Lower Spec Limit (LSL)
                  </label>
                  <input
                    type="number"
                    step="any"
                    value={specLimits.lsl}
                    onChange={(e) => setSpecLimits({ ...specLimits, lsl: e.target.value })}
                    className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    placeholder="e.g., 19.0"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Upper Spec Limit (USL)
                  </label>
                  <input
                    type="number"
                    step="any"
                    value={specLimits.usl}
                    onChange={(e) => setSpecLimits({ ...specLimits, usl: e.target.value })}
                    className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    placeholder="e.g., 21.0"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Target (optional)
                  </label>
                  <input
                    type="number"
                    step="any"
                    value={specLimits.target}
                    onChange={(e) => setSpecLimits({ ...specLimits, target: e.target.value })}
                    className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    placeholder="e.g., 20.0"
                  />
                </div>
              </div>
              <button
                onClick={handleCalculateCapability}
                disabled={loading}
                className="mt-4 px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-semibold transition-colors disabled:opacity-50"
              >
                {loading ? 'Calculating...' : 'Calculate Capability'}
              </button>
            </div>

            {/* Capability Results */}
            {capabilityResult && (
              <div className="space-y-6">
                <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
                  <h2 className="text-2xl font-bold mb-4 text-gray-100">Process Capability Indices</h2>
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                    {capabilityResult.cp !== null && (
                      <div className="bg-slate-700/30 rounded-lg p-4">
                        <p className="text-gray-400 text-sm mb-1">Cp</p>
                        <p className="text-3xl font-bold text-white">{capabilityResult.cp.toFixed(3)}</p>
                        <p className="text-gray-500 text-xs mt-1">Potential</p>
                      </div>
                    )}
                    {capabilityResult.cpk !== null && (
                      <div className="bg-slate-700/30 rounded-lg p-4">
                        <p className="text-gray-400 text-sm mb-1">Cpk</p>
                        <p className={`text-3xl font-bold ${
                          capabilityResult.cpk >= 1.33 ? 'text-green-400' :
                          capabilityResult.cpk >= 1.0 ? 'text-yellow-400' : 'text-red-400'
                        }`}>
                          {capabilityResult.cpk.toFixed(3)}
                        </p>
                        <p className="text-gray-500 text-xs mt-1">Actual</p>
                      </div>
                    )}
                    {capabilityResult.pp !== null && (
                      <div className="bg-slate-700/30 rounded-lg p-4">
                        <p className="text-gray-400 text-sm mb-1">Pp</p>
                        <p className="text-3xl font-bold text-white">{capabilityResult.pp.toFixed(3)}</p>
                        <p className="text-gray-500 text-xs mt-1">Long-term</p>
                      </div>
                    )}
                    {capabilityResult.ppk !== null && (
                      <div className="bg-slate-700/30 rounded-lg p-4">
                        <p className="text-gray-400 text-sm mb-1">Ppk</p>
                        <p className="text-3xl font-bold text-white">{capabilityResult.ppk.toFixed(3)}</p>
                        <p className="text-gray-500 text-xs mt-1">Long-term</p>
                      </div>
                    )}
                    {capabilityResult.cpm !== null && (
                      <div className="bg-slate-700/30 rounded-lg p-4">
                        <p className="text-gray-400 text-sm mb-1">Cpm</p>
                        <p className="text-3xl font-bold text-white">{capabilityResult.cpm.toFixed(3)}</p>
                        <p className="text-gray-500 text-xs mt-1">Taguchi</p>
                      </div>
                    )}
                  </div>

                  {capabilityResult.interpretation && (
                    <div className="mt-6 p-4 bg-slate-700/30 rounded-lg">
                      <p className="text-lg font-semibold text-gray-100">
                        Interpretation: <span className={
                          capabilityResult.interpretation.includes('Excellent') ? 'text-green-400' :
                          capabilityResult.interpretation.includes('good') ? 'text-blue-400' :
                          capabilityResult.interpretation.includes('Adequate') ? 'text-yellow-400' :
                          capabilityResult.interpretation.includes('Marginal') ? 'text-orange-400' : 'text-red-400'
                        }>
                          {capabilityResult.interpretation}
                        </span>
                      </p>
                      {capabilityResult.estimated_ppm !== undefined && (
                        <p className="text-gray-400 mt-2">
                          Estimated defect rate: {capabilityResult.estimated_ppm.toFixed(2)} PPM
                        </p>
                      )}
                    </div>
                  )}
                </div>

                <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
                  <h2 className="text-2xl font-bold mb-4 text-gray-100">Process Statistics</h2>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="bg-slate-700/30 rounded-lg p-4">
                      <p className="text-gray-400 text-sm mb-1">Process Mean</p>
                      <p className="text-2xl font-bold text-white">{capabilityResult.process_mean.toFixed(4)}</p>
                    </div>
                    <div className="bg-slate-700/30 rounded-lg p-4">
                      <p className="text-gray-400 text-sm mb-1">Process Std Dev</p>
                      <p className="text-2xl font-bold text-white">{capabilityResult.process_std_within.toFixed(4)}</p>
                    </div>
                    <div className="bg-slate-700/30 rounded-lg p-4">
                      <p className="text-gray-400 text-sm mb-1">Sample Size</p>
                      <p className="text-2xl font-bold text-white">{capabilityResult.sample_size}</p>
                    </div>
                    <div className="bg-slate-700/30 rounded-lg p-4">
                      <p className="text-gray-400 text-sm mb-1">Spec Width</p>
                      <p className="text-2xl font-bold text-white">
                        {capabilityResult.spec_limits.usl && capabilityResult.spec_limits.lsl
                          ? (capabilityResult.spec_limits.usl - capabilityResult.spec_limits.lsl).toFixed(4)
                          : 'N/A'}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Confidence Intervals */}
                {(capabilityResult.cpk_ci || capabilityResult.cp_ci) && (
                  <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
                    <h2 className="text-2xl font-bold mb-4 text-gray-100">
                      {capabilityResult.confidence_level ? `${(capabilityResult.confidence_level * 100).toFixed(0)}%` : '95%'} Confidence Intervals
                    </h2>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      {capabilityResult.cp_ci && (
                        <div className="bg-slate-700/30 rounded-lg p-4">
                          <p className="text-gray-400 text-sm mb-1">Cp CI</p>
                          <p className="text-lg font-bold text-white">
                            [{capabilityResult.cp_ci.lower.toFixed(3)}, {capabilityResult.cp_ci.upper.toFixed(3)}]
                          </p>
                        </div>
                      )}
                      {capabilityResult.cpk_ci && (
                        <div className="bg-slate-700/30 rounded-lg p-4">
                          <p className="text-gray-400 text-sm mb-1">Cpk CI</p>
                          <p className="text-lg font-bold text-white">
                            [{capabilityResult.cpk_ci.lower.toFixed(3)}, {capabilityResult.cpk_ci.upper.toFixed(3)}]
                          </p>
                        </div>
                      )}
                      {capabilityResult.pp_ci && (
                        <div className="bg-slate-700/30 rounded-lg p-4">
                          <p className="text-gray-400 text-sm mb-1">Pp CI</p>
                          <p className="text-lg font-bold text-white">
                            [{capabilityResult.pp_ci.lower.toFixed(3)}, {capabilityResult.pp_ci.upper.toFixed(3)}]
                          </p>
                        </div>
                      )}
                      {capabilityResult.ppk_ci && (
                        <div className="bg-slate-700/30 rounded-lg p-4">
                          <p className="text-gray-400 text-sm mb-1">Ppk CI</p>
                          <p className="text-lg font-bold text-white">
                            [{capabilityResult.ppk_ci.lower.toFixed(3)}, {capabilityResult.ppk_ci.upper.toFixed(3)}]
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Capability Histogram */}
                {capabilityResult.histogram && capabilityResult.normal_curve && (
                  <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
                    <h2 className="text-2xl font-bold mb-4 text-gray-100">Process Distribution</h2>
                    <Plot
                      data={[
                        {
                          x: capabilityResult.histogram.bin_centers,
                          y: capabilityResult.histogram.counts,
                          type: 'bar',
                          marker: { color: '#a855f7', opacity: 0.7 },
                          name: 'Histogram'
                        },
                        {
                          x: capabilityResult.normal_curve.x,
                          y: capabilityResult.normal_curve.y,
                          type: 'scatter',
                          mode: 'lines',
                          line: { color: '#22c55e', width: 2 },
                          name: 'Normal Fit'
                        },
                        ...(capabilityResult.spec_limits.lsl ? [{
                          x: [capabilityResult.spec_limits.lsl, capabilityResult.spec_limits.lsl],
                          y: [0, Math.max(...capabilityResult.histogram.counts) * 1.1],
                          type: 'scatter',
                          mode: 'lines',
                          line: { color: '#ef4444', width: 2, dash: 'dash' },
                          name: 'LSL'
                        }] : []),
                        ...(capabilityResult.spec_limits.usl ? [{
                          x: [capabilityResult.spec_limits.usl, capabilityResult.spec_limits.usl],
                          y: [0, Math.max(...capabilityResult.histogram.counts) * 1.1],
                          type: 'scatter',
                          mode: 'lines',
                          line: { color: '#ef4444', width: 2, dash: 'dash' },
                          name: 'USL'
                        }] : []),
                        {
                          x: [capabilityResult.process_mean, capabilityResult.process_mean],
                          y: [0, Math.max(...capabilityResult.histogram.counts) * 1.1],
                          type: 'scatter',
                          mode: 'lines',
                          line: { color: '#3b82f6', width: 2 },
                          name: 'Mean'
                        }
                      ]}
                      layout={{
                        xaxis: { title: 'Value', gridcolor: '#334155' },
                        yaxis: { title: 'Frequency', gridcolor: '#334155' },
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(15,23,42,0.5)',
                        font: { color: '#e2e8f0' },
                        autosize: true,
                        bargap: 0.05,
                        legend: { x: 0.02, y: 0.98 }
                      }}
                      style={{ width: '100%', height: '400px' }}
                      useResizeHandler={true}
                    />
                  </div>
                )}
              </div>
            )}

            {/* Info Box */}
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <h2 className="text-2xl font-bold mb-4 text-gray-100">About Process Capability</h2>
              <div className="space-y-3 text-gray-300">
                <p>
                  <strong className="text-white">Cp (Potential Capability):</strong> Measures how well the process could perform if perfectly centered. Ratio of spec width to process width.
                </p>
                <p>
                  <strong className="text-white">Cpk (Actual Capability):</strong> Accounts for process centering. The smaller of CPU or CPL. This is the most commonly used index.
                </p>
                <p>
                  <strong className="text-white">Pp/Ppk (Performance):</strong> Long-term capability using overall variation instead of within-subgroup variation.
                </p>
                <p>
                  <strong className="text-white">Cpm (Taguchi Index):</strong> Penalizes deviation from target value, not just spec limits.
                </p>
                <div className="mt-4 p-3 bg-slate-700/30 rounded-lg">
                  <p className="font-semibold text-white mb-2">Interpretation Guidelines:</p>
                  <ul className="space-y-1 text-sm">
                    <li>Cpk ≥ 2.0: Excellent (6-sigma quality)</li>
                    <li>Cpk ≥ 1.67: Very good (5-sigma quality)</li>
                    <li>Cpk ≥ 1.33: Adequate (4-sigma quality)</li>
                    <li>Cpk ≥ 1.0: Marginal (3-sigma quality)</li>
                    <li>Cpk &lt; 1.0: Inadequate (improvement needed)</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'msa' && (
          <div className="space-y-6">
            {/* MSA Type Selection */}
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <h2 className="text-2xl font-bold mb-4 text-gray-100">Measurement Systems Analysis</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <button
                  onClick={() => setMsaType('gauge-rr-crossed')}
                  className={`p-4 rounded-lg border-2 text-left transition-all ${
                    msaType === 'gauge-rr-crossed'
                      ? 'border-purple-500 bg-purple-900/30'
                      : 'border-slate-600 bg-slate-700/30 hover:border-slate-500'
                  }`}
                >
                  <h3 className="font-bold text-lg text-gray-100">Gauge R&R (Crossed)</h3>
                  <p className="text-gray-400 text-sm">Each operator measures each part multiple times</p>
                  <p className="text-purple-400 text-xs mt-1">Standard study for continuous measurements</p>
                </button>
                <button
                  onClick={() => setMsaType('gauge-rr-nested')}
                  className={`p-4 rounded-lg border-2 text-left transition-all ${
                    msaType === 'gauge-rr-nested'
                      ? 'border-purple-500 bg-purple-900/30'
                      : 'border-slate-600 bg-slate-700/30 hover:border-slate-500'
                  }`}
                >
                  <h3 className="font-bold text-lg text-gray-100">Gauge R&R (Nested)</h3>
                  <p className="text-gray-400 text-sm">Destructive testing - parts nested within operators</p>
                  <p className="text-purple-400 text-xs mt-1">For testing that consumes/destroys parts</p>
                </button>
              </div>
            </div>

            {/* MSA Data Entry */}
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <h2 className="text-2xl font-bold mb-4 text-gray-100">Study Setup</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Operators</label>
                  <input
                    type="number"
                    min="2"
                    max="10"
                    value={msaData.operators}
                    onChange={(e) => setMsaData({ ...msaData, operators: parseInt(e.target.value) || 2 })}
                    className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Parts</label>
                  <input
                    type="number"
                    min="2"
                    max="30"
                    value={msaData.parts}
                    onChange={(e) => setMsaData({ ...msaData, parts: parseInt(e.target.value) || 5 })}
                    className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Replicates</label>
                  <input
                    type="number"
                    min="2"
                    max="5"
                    value={msaData.replicates}
                    onChange={(e) => setMsaData({ ...msaData, replicates: parseInt(e.target.value) || 2 })}
                    className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Tolerance (optional)</label>
                  <input
                    type="number"
                    step="any"
                    value={msaTolerance}
                    onChange={(e) => setMsaTolerance(e.target.value)}
                    placeholder="e.g., 2.0"
                    className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>
              </div>

              <div className="flex gap-4 mb-4">
                <button
                  onClick={() => {
                    // Generate example data
                    const ops = msaData.operators
                    const parts = msaData.parts
                    const reps = msaData.replicates
                    const exampleData = []
                    const baseMean = 50
                    const partEffect = 2
                    const opEffect = 0.5
                    const repeatability = 0.3

                    for (let o = 0; o < ops; o++) {
                      const opData = []
                      for (let p = 0; p < parts; p++) {
                        const repData = []
                        const partMean = baseMean + (p - parts / 2) * partEffect + (o - ops / 2) * opEffect
                        for (let r = 0; r < reps; r++) {
                          repData.push(+(partMean + (Math.random() - 0.5) * repeatability * 2).toFixed(3))
                        }
                        opData.push(repData)
                      }
                      exampleData.push(opData)
                    }
                    setMsaTableData(exampleData)
                  }}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold transition-colors"
                >
                  Load Example Data
                </button>
                <button
                  onClick={() => {
                    // Initialize empty table
                    const ops = msaData.operators
                    const parts = msaData.parts
                    const reps = msaData.replicates
                    const emptyData = []
                    for (let o = 0; o < ops; o++) {
                      const opData = []
                      for (let p = 0; p < parts; p++) {
                        opData.push(Array(reps).fill(''))
                      }
                      emptyData.push(opData)
                    }
                    setMsaTableData(emptyData)
                  }}
                  className="px-4 py-2 bg-slate-600 hover:bg-slate-500 text-white rounded-lg font-semibold transition-colors"
                >
                  Initialize Empty Table
                </button>
              </div>

              {/* Data Table */}
              {msaTableData.length > 0 && (
                <div className="overflow-x-auto">
                  <p className="text-gray-400 text-sm mb-2">
                    Enter measurements: Rows = Parts, Columns = Replicates (grouped by Operator)
                  </p>
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="bg-slate-700">
                        <th className="px-2 py-2 text-left text-gray-300 border border-slate-600">Part</th>
                        {msaTableData.map((_, opIdx) => (
                          Array(msaData.replicates).fill(null).map((_, repIdx) => (
                            <th key={`op${opIdx}-rep${repIdx}`} className="px-2 py-2 text-center text-gray-300 border border-slate-600">
                              Op{opIdx + 1}-R{repIdx + 1}
                            </th>
                          ))
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {Array(msaData.parts).fill(null).map((_, partIdx) => (
                        <tr key={partIdx} className="hover:bg-slate-700/30">
                          <td className="px-2 py-1 text-gray-400 border border-slate-600 font-mono">{partIdx + 1}</td>
                          {msaTableData.map((opData, opIdx) => (
                            opData[partIdx]?.map((val, repIdx) => (
                              <td key={`${opIdx}-${partIdx}-${repIdx}`} className="p-0 border border-slate-600">
                                <input
                                  type="text"
                                  value={val}
                                  onChange={(e) => {
                                    const newData = [...msaTableData]
                                    newData[opIdx][partIdx][repIdx] = e.target.value
                                    setMsaTableData(newData)
                                  }}
                                  className="w-full px-2 py-1 bg-transparent text-gray-100 text-center focus:bg-slate-700/50 focus:outline-none font-mono text-xs"
                                />
                              </td>
                            ))
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              <button
                onClick={async () => {
                  if (msaTableData.length === 0) {
                    setError('Please enter or load MSA data first')
                    return
                  }

                  setLoading(true)
                  setError('')

                  try {
                    // Convert table data to numeric
                    const measurements = msaTableData.map(opData =>
                      opData.map(partData =>
                        partData.map(val => parseFloat(val))
                      )
                    )

                    // Check for NaN values
                    const hasNaN = measurements.some(op =>
                      op.some(part => part.some(val => isNaN(val)))
                    )
                    if (hasNaN) {
                      throw new Error('All cells must contain valid numbers')
                    }

                    const endpoint = msaType === 'gauge-rr-crossed'
                      ? `${API_URL}/api/msa/gauge-rr/crossed`
                      : `${API_URL}/api/msa/gauge-rr/nested`

                    const requestData = {
                      measurements,
                      tolerance: msaTolerance ? parseFloat(msaTolerance) : null
                    }

                    const response = await axios.post(endpoint, requestData)
                    setMsaResult(response.data)
                  } catch (err) {
                    setError('MSA analysis failed: ' + (err.response?.data?.detail || err.message))
                  } finally {
                    setLoading(false)
                  }
                }}
                disabled={loading || msaTableData.length === 0}
                className="mt-4 px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-semibold transition-colors disabled:opacity-50"
              >
                {loading ? 'Analyzing...' : 'Run Gauge R&R Analysis'}
              </button>
            </div>

            {/* MSA Results */}
            {msaResult && (
              <div className="space-y-6">
                {/* Summary */}
                <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
                  <h2 className="text-2xl font-bold mb-4 text-gray-100">Gauge R&R Results</h2>

                  {/* Key Metrics */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                    <div className="bg-slate-700/30 rounded-lg p-4">
                      <p className="text-gray-400 text-sm mb-1">Total Gauge R&R</p>
                      <p className={`text-3xl font-bold ${
                        msaResult.percent_study_variation?.total_gauge_rr < 10 ? 'text-green-400' :
                        msaResult.percent_study_variation?.total_gauge_rr < 30 ? 'text-yellow-400' : 'text-red-400'
                      }`}>
                        {msaResult.percent_study_variation?.total_gauge_rr?.toFixed(1)}%
                      </p>
                      <p className="text-gray-500 text-xs mt-1">% Study Variation</p>
                    </div>
                    <div className="bg-slate-700/30 rounded-lg p-4">
                      <p className="text-gray-400 text-sm mb-1">Repeatability</p>
                      <p className="text-2xl font-bold text-white">
                        {msaResult.percent_study_variation?.repeatability?.toFixed(1)}%
                      </p>
                      <p className="text-gray-500 text-xs mt-1">Equipment Variation</p>
                    </div>
                    <div className="bg-slate-700/30 rounded-lg p-4">
                      <p className="text-gray-400 text-sm mb-1">Reproducibility</p>
                      <p className="text-2xl font-bold text-white">
                        {msaResult.percent_study_variation?.reproducibility?.toFixed(1)}%
                      </p>
                      <p className="text-gray-500 text-xs mt-1">Appraiser Variation</p>
                    </div>
                    <div className="bg-slate-700/30 rounded-lg p-4">
                      <p className="text-gray-400 text-sm mb-1">Distinct Categories</p>
                      <p className={`text-2xl font-bold ${
                        msaResult.number_distinct_categories >= 5 ? 'text-green-400' : 'text-red-400'
                      }`}>
                        {msaResult.number_distinct_categories?.toFixed(0)}
                      </p>
                      <p className="text-gray-500 text-xs mt-1">ndc (need ≥5)</p>
                    </div>
                  </div>

                  {/* Interpretation */}
                  <div className={`p-4 rounded-lg ${
                    msaResult.interpretation?.includes('Excellent') ? 'bg-green-900/30 border border-green-700/50' :
                    msaResult.interpretation?.includes('Acceptable') ? 'bg-yellow-900/30 border border-yellow-700/50' :
                    'bg-red-900/30 border border-red-700/50'
                  }`}>
                    <p className="text-lg font-semibold text-white">{msaResult.interpretation}</p>
                    {msaResult.ndc_interpretation && (
                      <p className="text-gray-300 text-sm mt-1">
                        Number of distinct categories: {msaResult.ndc_interpretation}
                      </p>
                    )}
                  </div>
                </div>

                {/* ANOVA Table */}
                {msaResult.anova_table && (
                  <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
                    <h2 className="text-2xl font-bold mb-4 text-gray-100">ANOVA Table</h2>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="bg-slate-700">
                            <th className="px-4 py-2 text-left text-gray-300">Source</th>
                            <th className="px-4 py-2 text-right text-gray-300">DF</th>
                            <th className="px-4 py-2 text-right text-gray-300">SS</th>
                            <th className="px-4 py-2 text-right text-gray-300">MS</th>
                            <th className="px-4 py-2 text-right text-gray-300">F</th>
                            <th className="px-4 py-2 text-right text-gray-300">P-Value</th>
                          </tr>
                        </thead>
                        <tbody>
                          {msaResult.anova_table.sources.map((source, idx) => (
                            <tr key={idx} className="border-b border-slate-700 hover:bg-slate-700/30">
                              <td className="px-4 py-2 text-gray-100 font-medium">{source}</td>
                              <td className="px-4 py-2 text-right text-gray-300">{msaResult.anova_table.df[idx]}</td>
                              <td className="px-4 py-2 text-right text-gray-300">{msaResult.anova_table.ss[idx]?.toFixed(4)}</td>
                              <td className="px-4 py-2 text-right text-gray-300">
                                {msaResult.anova_table.ms[idx] !== null ? msaResult.anova_table.ms[idx]?.toFixed(4) : '-'}
                              </td>
                              <td className="px-4 py-2 text-right text-gray-300">
                                {msaResult.anova_table.f_value[idx] !== null ? msaResult.anova_table.f_value[idx]?.toFixed(4) : '-'}
                              </td>
                              <td className={`px-4 py-2 text-right ${
                                msaResult.anova_table.p_value[idx] !== null && msaResult.anova_table.p_value[idx] < 0.05
                                  ? 'text-yellow-400' : 'text-gray-300'
                              }`}>
                                {msaResult.anova_table.p_value[idx] !== null
                                  ? msaResult.anova_table.p_value[idx]?.toFixed(4)
                                  : '-'}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Variance Components */}
                {msaResult.variance_components && (
                  <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
                    <h2 className="text-2xl font-bold mb-4 text-gray-100">Variance Components</h2>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                      <div className="bg-slate-700/30 rounded-lg p-4">
                        <p className="text-gray-400 text-sm mb-1">Total Gauge R&R</p>
                        <p className="text-xl font-bold text-white">{msaResult.variance_components.total_gauge_rr?.toFixed(6)}</p>
                      </div>
                      <div className="bg-slate-700/30 rounded-lg p-4">
                        <p className="text-gray-400 text-sm mb-1">Repeatability</p>
                        <p className="text-xl font-bold text-white">{msaResult.variance_components.repeatability?.toFixed(6)}</p>
                      </div>
                      <div className="bg-slate-700/30 rounded-lg p-4">
                        <p className="text-gray-400 text-sm mb-1">Reproducibility</p>
                        <p className="text-xl font-bold text-white">{msaResult.variance_components.reproducibility?.toFixed(6)}</p>
                      </div>
                      <div className="bg-slate-700/30 rounded-lg p-4">
                        <p className="text-gray-400 text-sm mb-1">Part-to-Part</p>
                        <p className="text-xl font-bold text-white">{msaResult.variance_components.part_to_part?.toFixed(6)}</p>
                      </div>
                      <div className="bg-slate-700/30 rounded-lg p-4">
                        <p className="text-gray-400 text-sm mb-1">Total Variation</p>
                        <p className="text-xl font-bold text-white">{msaResult.variance_components.total_variation?.toFixed(6)}</p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Gauge R&R Chart */}
                {msaResult.percent_study_variation && (
                  <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
                    <h2 className="text-2xl font-bold mb-4 text-gray-100">Components of Variation</h2>
                    <Plot
                      data={[{
                        x: ['Gauge R&R', 'Repeatability', 'Reproducibility', 'Part-to-Part'],
                        y: [
                          msaResult.percent_study_variation.total_gauge_rr,
                          msaResult.percent_study_variation.repeatability,
                          msaResult.percent_study_variation.reproducibility,
                          msaResult.percent_study_variation.part_to_part || 0
                        ],
                        type: 'bar',
                        marker: {
                          color: [
                            msaResult.percent_study_variation.total_gauge_rr < 10 ? '#22c55e' :
                            msaResult.percent_study_variation.total_gauge_rr < 30 ? '#f59e0b' : '#ef4444',
                            '#a855f7',
                            '#3b82f6',
                            '#22c55e'
                          ]
                        }
                      }]}
                      layout={{
                        xaxis: { title: 'Source', gridcolor: '#334155' },
                        yaxis: { title: '% Study Variation', gridcolor: '#334155' },
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(15,23,42,0.5)',
                        font: { color: '#e2e8f0' },
                        autosize: true,
                        shapes: [
                          { type: 'line', x0: -0.5, x1: 3.5, y0: 10, y1: 10, line: { color: '#22c55e', width: 2, dash: 'dash' } },
                          { type: 'line', x0: -0.5, x1: 3.5, y0: 30, y1: 30, line: { color: '#ef4444', width: 2, dash: 'dash' } }
                        ],
                        annotations: [
                          { x: 3.5, y: 10, text: '10% (Excellent)', showarrow: false, xanchor: 'left', font: { color: '#22c55e', size: 10 } },
                          { x: 3.5, y: 30, text: '30% (Acceptable)', showarrow: false, xanchor: 'left', font: { color: '#ef4444', size: 10 } }
                        ]
                      }}
                      style={{ width: '100%', height: '400px' }}
                      useResizeHandler={true}
                    />
                  </div>
                )}
              </div>
            )}

            {/* MSA Info */}
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <h2 className="text-2xl font-bold mb-4 text-gray-100">About Gauge R&R</h2>
              <div className="space-y-3 text-gray-300">
                <p>
                  <strong className="text-white">Repeatability (EV):</strong> Variation when the same operator measures the same part multiple times with the same gauge.
                </p>
                <p>
                  <strong className="text-white">Reproducibility (AV):</strong> Variation when different operators measure the same parts with the same gauge.
                </p>
                <p>
                  <strong className="text-white">Gauge R&R:</strong> Combined repeatability and reproducibility, representing total measurement system variation.
                </p>
                <p>
                  <strong className="text-white">Number of Distinct Categories (ndc):</strong> Number of non-overlapping confidence intervals that span the range of product variation.
                </p>
                <div className="mt-4 p-3 bg-slate-700/30 rounded-lg">
                  <p className="font-semibold text-white mb-2">Acceptance Criteria:</p>
                  <ul className="space-y-1 text-sm">
                    <li className="text-green-400">%GRR &lt; 10%: Excellent - acceptable for all applications</li>
                    <li className="text-yellow-400">%GRR 10-30%: Acceptable - may be suitable depending on importance</li>
                    <li className="text-red-400">%GRR &gt; 30%: Unacceptable - measurement system needs improvement</li>
                    <li className="mt-2">ndc ≥ 5: Adequate for distinguishing parts</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default QualityControl
