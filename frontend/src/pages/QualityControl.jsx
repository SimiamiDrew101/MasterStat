import { useState } from 'react'
import axios from 'axios'
import Plot from 'react-plotly.js'
import { Upload, BarChart3, TrendingUp, Info, AlertTriangle } from 'lucide-react'
import * as XLSX from 'xlsx'

const QualityControl = () => {
  // State
  const [activeTab, setActiveTab] = useState('data')
  const [data, setData] = useState([])
  const [dataText, setDataText] = useState('')
  const [chartType, setChartType] = useState('i-mr')
  const [subgroupSize, setSubgroupSize] = useState(5)
  const [chartResult, setChartResult] = useState(null)
  const [capabilityResult, setCapabilityResult] = useState(null)
  const [specLimits, setSpecLimits] = useState({ lsl: '', usl: '', target: '' })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

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

    setData(exampleData)
    setDataText(exampleData.join('\n'))
    setSubgroupSize(exampleSubgroup)
    setActiveTab('chart')
  }

  // Parse pasted data
  const handleParseData = () => {
    try {
      const lines = dataText.trim().split('\n')
      const parsedData = []

      lines.forEach(line => {
        const value = parseFloat(line.trim())
        if (!isNaN(value)) {
          parsedData.push(value)
        }
      })

      if (parsedData.length < 3) {
        setError('Need at least 3 data points')
        return
      }

      setData(parsedData)
      setError('')
      setActiveTab('chart')
    } catch (err) {
      setError('Failed to parse data. Enter one number per line')
    }
  }

  // Upload Excel/CSV file
  const handleFileUpload = (event) => {
    const file = event.target.files[0]
    if (!file) return

    const reader = new FileReader()

    reader.onload = (e) => {
      try {
        const data = new Uint8Array(e.target.result)
        const workbook = XLSX.read(data, { type: 'array' })
        const sheetName = workbook.SheetNames[0]
        const worksheet = workbook.Sheets[sheetName]
        const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 })

        const parsedData = []
        jsonData.forEach((row, i) => {
          if (i === 0) return // Skip header
          const value = parseFloat(row[0])
          if (!isNaN(value)) {
            parsedData.push(value)
          }
        })

        if (parsedData.length < 3) {
          setError('Need at least 3 valid data points in file')
          return
        }

        setData(parsedData)
        setDataText(parsedData.join('\n'))
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
      const requestData = {
        data: data,
        chart_type: chartType
      }

      const selectedChart = chartTypes.find(c => c.id === chartType)
      if (selectedChart && selectedChart.needsSubgroup) {
        requestData.subgroup_size = subgroupSize
      }

      const response = await axios.post('/api/quality-control/control-chart', requestData)
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

      const response = await axios.post('/api/quality-control/capability', requestData)
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
            <Info size={20} />
            <span>4. Capability</span>
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

            {/* Manual Data Entry */}
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <h2 className="text-2xl font-bold mb-4 text-gray-100">Enter Process Data</h2>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Paste measurements (one per line)
              </label>
              <textarea
                value={dataText}
                onChange={(e) => setDataText(e.target.value)}
                className="w-full h-64 px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white font-mono focus:outline-none focus:ring-2 focus:ring-purple-500"
                placeholder="20.1&#10;19.8&#10;20.3&#10;19.9&#10;20.2&#10;..."
              />
              <button
                onClick={handleParseData}
                className="mt-4 px-6 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-semibold transition-colors"
              >
                Parse Data
              </button>
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
      </div>
    </div>
  )
}

export default QualityControl
