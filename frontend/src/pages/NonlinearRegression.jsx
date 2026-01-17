import { useState } from 'react'
import axios from 'axios'
import Plot from 'react-plotly.js'
import { Upload, TrendingUp, Activity, Info } from 'lucide-react'
import * as XLSX from 'xlsx'

const NonlinearRegression = () => {
  // State
  const [activeTab, setActiveTab] = useState('data')
  const [xData, setXData] = useState([])
  const [yData, setYData] = useState([])
  const [xLabel, setXLabel] = useState('X')
  const [yLabel, setYLabel] = useState('Y')
  const [dataText, setDataText] = useState('')
  const [availableModels, setAvailableModels] = useState([])
  const [selectedModel, setSelectedModel] = useState('')
  const [modelInfo, setModelInfo] = useState(null)
  const [initialParams, setInitialParams] = useState({})
  const [useSuggestedParams, setUseSuggestedParams] = useState(true)
  const [fitResult, setFitResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  // Load example datasets
  const loadExampleData = (exampleType) => {
    let exampleX, exampleY, exampleXLabel, exampleYLabel

    if (exampleType === 'exponential') {
      exampleX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      exampleY = [2.1, 2.8, 3.9, 5.5, 8.0, 11.8, 17.2, 25.1, 36.8, 53.9, 78.6]
      exampleXLabel = 'Time (hours)'
      exampleYLabel = 'Population'
    } else if (exampleType === 'logistic') {
      exampleX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
      exampleY = [2, 3, 5, 9, 16, 28, 45, 65, 82, 92, 97, 99, 99.5]
      exampleXLabel = 'Time (days)'
      exampleYLabel = 'Market Penetration (%)'
    } else if (exampleType === 'michaelis_menten') {
      exampleX = [0.5, 1, 2, 5, 10, 20, 50, 100]
      exampleY = [1.2, 2.0, 3.2, 5.5, 7.8, 9.5, 10.8, 11.2]
      exampleXLabel = 'Substrate Concentration'
      exampleYLabel = 'Reaction Rate'
    } else if (exampleType === 'power_law') {
      exampleX = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      exampleY = [2.1, 5.8, 11.2, 18.5, 27.6, 38.4, 51.0, 65.3, 81.4, 99.2]
      exampleXLabel = 'X'
      exampleYLabel = 'Y'
    }

    setXData(exampleX)
    setYData(exampleY)
    setXLabel(exampleXLabel)
    setYLabel(exampleYLabel)
    setDataText(exampleX.map((x, i) => `${x}\t${exampleY[i]}`).join('\n'))
    setActiveTab('model')
  }

  // Parse pasted data
  const handleParseData = () => {
    try {
      const lines = dataText.trim().split('\n')
      const parsedX = []
      const parsedY = []

      lines.forEach(line => {
        const parts = line.trim().split(/[\s,\t]+/)
        if (parts.length >= 2) {
          const x = parseFloat(parts[0])
          const y = parseFloat(parts[1])
          if (!isNaN(x) && !isNaN(y)) {
            parsedX.push(x)
            parsedY.push(y)
          }
        }
      })

      if (parsedX.length < 3) {
        setError('Need at least 3 data points')
        return
      }

      setXData(parsedX)
      setYData(parsedY)
      setError('')
      setActiveTab('model')
    } catch (err) {
      setError('Failed to parse data. Use format: X Y (one pair per line)')
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

        // Extract X and Y columns (first two columns)
        const parsedX = []
        const parsedY = []

        jsonData.forEach((row, i) => {
          if (i === 0) return // Skip header
          if (row.length >= 2) {
            const x = parseFloat(row[0])
            const y = parseFloat(row[1])
            if (!isNaN(x) && !isNaN(y)) {
              parsedX.push(x)
              parsedY.push(y)
            }
          }
        })

        if (parsedX.length < 3) {
          setError('Need at least 3 valid data points in file')
          return
        }

        setXData(parsedX)
        setYData(parsedY)
        setError('')
        setActiveTab('model')
      } catch (err) {
        setError('Failed to read file: ' + err.message)
      }
    }

    reader.readAsArrayBuffer(file)
  }

  // Fetch available models
  const loadModels = async () => {
    try {
      const response = await axios.get('/api/nonlinear-regression/models')
      setAvailableModels(response.data.models)
    } catch (err) {
      setError('Failed to load models: ' + err.message)
    }
  }

  // Select model and suggest parameters
  const handleSelectModel = async (modelName) => {
    setSelectedModel(modelName)
    const model = availableModels.find(m => m.name === modelName)
    setModelInfo(model)

    // Auto-suggest initial parameters
    if (useSuggestedParams) {
      try {
        const response = await axios.post('/api/nonlinear-regression/suggest-initial', {
          x_data: xData,
          y_data: yData,
          model_name: modelName
        })
        setInitialParams(response.data.suggested_parameters)
      } catch (err) {
        console.error('Failed to suggest parameters:', err)
      }
    }
  }

  // Fit model
  const handleFitModel = async () => {
    if (!selectedModel) {
      setError('Please select a model first')
      return
    }

    setLoading(true)
    setError('')

    try {
      const requestData = {
        x_data: xData,
        y_data: yData,
        model_name: selectedModel,
        x_label: xLabel,
        y_label: yLabel
      }

      if (!useSuggestedParams && Object.keys(initialParams).length > 0) {
        requestData.initial_params = initialParams
      }

      const response = await axios.post('/api/nonlinear-regression/fit', requestData)
      setFitResult(response.data)
      setActiveTab('results')
    } catch (err) {
      setError('Model fitting failed: ' + (err.response?.data?.detail || err.message))
    } finally {
      setLoading(false)
    }
  }

  // Load models on mount
  useState(() => {
    loadModels()
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-pink-600 text-transparent bg-clip-text">
            Nonlinear Regression
          </h1>
          <p className="text-gray-300 text-lg">
            Fit custom nonlinear models to your data with parameter estimation and diagnostics
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
            disabled={xData.length === 0}
            className={`flex-1 px-4 py-3 rounded-lg font-medium transition-all duration-200 flex items-center justify-center space-x-2 ${
              activeTab === 'model'
                ? 'bg-purple-600 text-white shadow-lg'
                : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700 disabled:opacity-40'
            }`}
          >
            <TrendingUp size={20} />
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
            <Activity size={20} />
            <span>3. Results</span>
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
          <div className="mb-6 bg-red-900/50 border border-red-600 text-red-200 px-4 py-3 rounded-lg">
            {error}
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
                  onClick={() => loadExampleData('exponential')}
                  className="px-6 py-4 bg-gradient-to-br from-purple-600 to-purple-700 hover:from-purple-500 hover:to-purple-600 text-white rounded-xl font-semibold transition-all duration-200 shadow-lg hover:shadow-purple-500/50"
                >
                  Exponential Growth
                </button>
                <button
                  onClick={() => loadExampleData('logistic')}
                  className="px-6 py-4 bg-gradient-to-br from-blue-600 to-blue-700 hover:from-blue-500 hover:to-blue-600 text-white rounded-xl font-semibold transition-all duration-200 shadow-lg hover:shadow-blue-500/50"
                >
                  Logistic Curve
                </button>
                <button
                  onClick={() => loadExampleData('michaelis_menten')}
                  className="px-6 py-4 bg-gradient-to-br from-green-600 to-green-700 hover:from-green-500 hover:to-green-600 text-white rounded-xl font-semibold transition-all duration-200 shadow-lg hover:shadow-green-500/50"
                >
                  Enzyme Kinetics
                </button>
                <button
                  onClick={() => loadExampleData('power_law')}
                  className="px-6 py-4 bg-gradient-to-br from-orange-600 to-orange-700 hover:from-orange-500 hover:to-orange-600 text-white rounded-xl font-semibold transition-all duration-200 shadow-lg hover:shadow-orange-500/50"
                >
                  Power Law
                </button>
              </div>
            </div>

            {/* Manual Data Entry */}
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <h2 className="text-2xl font-bold mb-4 text-gray-100">Enter Your Data</h2>
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">X-axis Label</label>
                  <input
                    type="text"
                    value={xLabel}
                    onChange={(e) => setXLabel(e.target.value)}
                    className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    placeholder="e.g., Time"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Y-axis Label</label>
                  <input
                    type="text"
                    value={yLabel}
                    onChange={(e) => setYLabel(e.target.value)}
                    className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    placeholder="e.g., Concentration"
                  />
                </div>
              </div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Paste data (format: X Y, one pair per line)
              </label>
              <textarea
                value={dataText}
                onChange={(e) => setDataText(e.target.value)}
                className="w-full h-64 px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white font-mono focus:outline-none focus:ring-2 focus:ring-purple-500"
                placeholder="0 2.0&#10;1 2.7&#10;2 3.7&#10;3 5.4&#10;4 8.1&#10;5 12.2"
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
                Upload Excel or CSV file with X values in first column, Y values in second column
              </p>
            </div>

            {/* Data Preview */}
            {xData.length > 0 && (
              <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
                <h2 className="text-2xl font-bold mb-4 text-gray-100">Data Preview</h2>
                <p className="text-gray-300 mb-4">Loaded {xData.length} data points</p>
                <Plot
                  data={[{
                    x: xData,
                    y: yData,
                    type: 'scatter',
                    mode: 'markers',
                    marker: { size: 10, color: '#a855f7' },
                    name: 'Data'
                  }]}
                  layout={{
                    title: 'Raw Data',
                    xaxis: { title: xLabel, gridcolor: '#334155' },
                    yaxis: { title: yLabel, gridcolor: '#334155' },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(15,23,42,0.5)',
                    font: { color: '#e2e8f0' },
                    autosize: true
                  }}
                  style={{ width: '100%', height: '400px' }}
                  useResizeHandler={true}
                />
              </div>
            )}
          </div>
        )}

        {activeTab === 'model' && (
          <div className="space-y-6">
            {/* Model Selection */}
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <h2 className="text-2xl font-bold mb-4 text-gray-100">Select Nonlinear Model</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {availableModels.map(model => (
                  <button
                    key={model.name}
                    onClick={() => handleSelectModel(model.name)}
                    className={`p-4 rounded-lg border-2 text-left transition-all duration-200 ${
                      selectedModel === model.name
                        ? 'border-purple-500 bg-purple-900/30'
                        : 'border-slate-600 bg-slate-700/30 hover:border-slate-500'
                    }`}
                  >
                    <h3 className="font-bold text-lg text-gray-100 mb-1">
                      {model.name.replace('_', ' ').toUpperCase()}
                    </h3>
                    <p className="text-purple-400 font-mono text-sm mb-2">{model.equation}</p>
                    <p className="text-gray-400 text-sm mb-1">{model.description}</p>
                    <p className="text-gray-500 text-xs italic">{model.typical_use}</p>
                  </button>
                ))}
              </div>
            </div>

            {/* Initial Parameters */}
            {selectedModel && (
              <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
                <h2 className="text-2xl font-bold mb-4 text-gray-100">Initial Parameter Values</h2>
                <div className="mb-4">
                  <label className="flex items-center space-x-2 text-gray-300">
                    <input
                      type="checkbox"
                      checked={useSuggestedParams}
                      onChange={(e) => setUseSuggestedParams(e.target.checked)}
                      className="w-4 h-4 text-purple-600 bg-slate-700 border-slate-600 rounded focus:ring-purple-500"
                    />
                    <span>Use automatically suggested parameters</span>
                  </label>
                </div>

                {modelInfo && (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {modelInfo.parameters.map(param => (
                      <div key={param}>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                          {param}
                        </label>
                        <input
                          type="number"
                          step="any"
                          value={initialParams[param] || ''}
                          onChange={(e) => setInitialParams({
                            ...initialParams,
                            [param]: parseFloat(e.target.value)
                          })}
                          disabled={useSuggestedParams}
                          className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500 disabled:opacity-50"
                          placeholder="Auto"
                        />
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Fit Button */}
            {selectedModel && (
              <div className="flex justify-center">
                <button
                  onClick={handleFitModel}
                  disabled={loading}
                  className="px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white rounded-xl font-bold text-lg shadow-lg hover:shadow-purple-500/50 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? 'Fitting Model...' : 'Fit Model'}
                </button>
              </div>
            )}
          </div>
        )}

        {activeTab === 'results' && fitResult && (
          <div className="space-y-6">
            {/* Model Equation */}
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <h2 className="text-2xl font-bold mb-4 text-gray-100">Fitted Model</h2>
              <p className="text-purple-400 font-mono text-xl mb-2">{fitResult.equation}</p>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
                {Object.entries(fitResult.parameters).map(([param, data]) => (
                  <div key={param} className="bg-slate-700/30 rounded-lg p-4">
                    <p className="text-gray-400 text-sm mb-1">{param}</p>
                    <p className="text-2xl font-bold text-white">{data.estimate.toFixed(4)}</p>
                    <p className="text-gray-500 text-xs mt-1">
                      95% CI: [{data.ci_lower.toFixed(3)}, {data.ci_upper.toFixed(3)}]
                    </p>
                    <p className="text-gray-500 text-xs">p = {data.p_value.toExponential(3)}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Fit Statistics */}
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <h2 className="text-2xl font-bold mb-4 text-gray-100">Goodness of Fit</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-slate-700/30 rounded-lg p-4">
                  <p className="text-gray-400 text-sm mb-1">R²</p>
                  <p className="text-3xl font-bold text-green-400">
                    {fitResult.statistics.r_squared.toFixed(4)}
                  </p>
                </div>
                <div className="bg-slate-700/30 rounded-lg p-4">
                  <p className="text-gray-400 text-sm mb-1">Adj R²</p>
                  <p className="text-2xl font-bold text-white">
                    {fitResult.statistics.adj_r_squared.toFixed(4)}
                  </p>
                </div>
                <div className="bg-slate-700/30 rounded-lg p-4">
                  <p className="text-gray-400 text-sm mb-1">RMSE</p>
                  <p className="text-2xl font-bold text-white">
                    {fitResult.statistics.rmse.toFixed(4)}
                  </p>
                </div>
                <div className="bg-slate-700/30 rounded-lg p-4">
                  <p className="text-gray-400 text-sm mb-1">MAE</p>
                  <p className="text-2xl font-bold text-white">
                    {fitResult.statistics.mae.toFixed(4)}
                  </p>
                </div>
                <div className="bg-slate-700/30 rounded-lg p-4">
                  <p className="text-gray-400 text-sm mb-1">AIC</p>
                  <p className="text-2xl font-bold text-white">
                    {fitResult.statistics.aic.toFixed(2)}
                  </p>
                </div>
                <div className="bg-slate-700/30 rounded-lg p-4">
                  <p className="text-gray-400 text-sm mb-1">BIC</p>
                  <p className="text-2xl font-bold text-white">
                    {fitResult.statistics.bic.toFixed(2)}
                  </p>
                </div>
                <div className="bg-slate-700/30 rounded-lg p-4">
                  <p className="text-gray-400 text-sm mb-1">N Observations</p>
                  <p className="text-2xl font-bold text-white">
                    {fitResult.statistics.n_observations}
                  </p>
                </div>
                <div className="bg-slate-700/30 rounded-lg p-4">
                  <p className="text-gray-400 text-sm mb-1">DF</p>
                  <p className="text-2xl font-bold text-white">
                    {fitResult.statistics.degrees_of_freedom}
                  </p>
                </div>
              </div>
            </div>

            {/* Fitted Curve Plot */}
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <h2 className="text-2xl font-bold mb-4 text-gray-100">Fitted Curve</h2>
              <Plot
                data={[
                  {
                    x: fitResult.x_data,
                    y: fitResult.y_data,
                    type: 'scatter',
                    mode: 'markers',
                    marker: { size: 10, color: '#a855f7' },
                    name: 'Observed Data'
                  },
                  {
                    x: fitResult.prediction_curve.x,
                    y: fitResult.prediction_curve.y,
                    type: 'scatter',
                    mode: 'lines',
                    line: { color: '#ec4899', width: 3 },
                    name: 'Fitted Model'
                  }
                ]}
                layout={{
                  title: `${fitResult.model} Model Fit (R² = ${fitResult.statistics.r_squared.toFixed(4)})`,
                  xaxis: { title: xLabel, gridcolor: '#334155' },
                  yaxis: { title: yLabel, gridcolor: '#334155' },
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(15,23,42,0.5)',
                  font: { color: '#e2e8f0' },
                  autosize: true,
                  legend: { x: 0.02, y: 0.98 }
                }}
                style={{ width: '100%', height: '500px' }}
                useResizeHandler={true}
              />
            </div>

            {/* Residual Plots */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
                <h2 className="text-xl font-bold mb-4 text-gray-100">Residuals vs Fitted</h2>
                <Plot
                  data={[{
                    x: fitResult.fitted_values,
                    y: fitResult.residuals,
                    type: 'scatter',
                    mode: 'markers',
                    marker: { size: 8, color: '#a855f7' },
                    name: 'Residuals'
                  }, {
                    x: [Math.min(...fitResult.fitted_values), Math.max(...fitResult.fitted_values)],
                    y: [0, 0],
                    type: 'scatter',
                    mode: 'lines',
                    line: { color: '#ef4444', dash: 'dash' },
                    name: 'Zero Line'
                  }]}
                  layout={{
                    xaxis: { title: 'Fitted Values', gridcolor: '#334155' },
                    yaxis: { title: 'Residuals', gridcolor: '#334155' },
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

              <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
                <h2 className="text-xl font-bold mb-4 text-gray-100">Residual Distribution</h2>
                <Plot
                  data={[{
                    x: fitResult.residuals,
                    type: 'histogram',
                    marker: { color: '#a855f7' },
                    name: 'Residuals'
                  }]}
                  layout={{
                    xaxis: { title: 'Residuals', gridcolor: '#334155' },
                    yaxis: { title: 'Frequency', gridcolor: '#334155' },
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
            </div>
          </div>
        )}

        {activeTab === 'info' && (
          <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
            <h2 className="text-2xl font-bold mb-4 text-gray-100">About Nonlinear Regression</h2>
            <div className="space-y-4 text-gray-300">
              <p>
                Nonlinear regression fits a nonlinear equation to data using iterative optimization.
                Unlike linear regression, these models cannot be solved analytically and require
                numerical methods like Levenberg-Marquardt.
              </p>

              <h3 className="text-xl font-bold text-white mt-6">Available Models</h3>
              <ul className="list-disc list-inside space-y-2 ml-4">
                <li><strong>Exponential:</strong> y = a × exp(b×x) - Growth or decay</li>
                <li><strong>Logistic:</strong> y = L / (1 + exp(-k×(x-x₀))) - S-shaped growth with saturation</li>
                <li><strong>Michaelis-Menten:</strong> y = Vmax×x / (Km + x) - Enzyme kinetics</li>
                <li><strong>Power Law:</strong> y = a × x^b - Allometric scaling</li>
                <li><strong>Gompertz:</strong> y = a × exp(-b × exp(-c×x)) - Asymmetric S-curve</li>
                <li><strong>Weibull:</strong> y = a × (1 - exp(-(x/b)^c)) - Reliability analysis</li>
                <li><strong>Logarithmic:</strong> y = a + b×ln(x) - Logarithmic relationship</li>
              </ul>

              <h3 className="text-xl font-bold text-white mt-6">Parameter Estimation</h3>
              <p>
                The system automatically suggests initial parameter values using heuristics based on
                your data. Good initial values help the optimizer converge faster and avoid local minima.
              </p>

              <h3 className="text-xl font-bold text-white mt-6">Fit Statistics</h3>
              <ul className="list-disc list-inside space-y-2 ml-4">
                <li><strong>R²:</strong> Proportion of variance explained (0-1, higher is better)</li>
                <li><strong>RMSE:</strong> Root mean squared error (lower is better)</li>
                <li><strong>AIC/BIC:</strong> Information criteria for model comparison (lower is better)</li>
                <li><strong>p-values:</strong> Statistical significance of parameters (p &lt; 0.05 significant)</li>
              </ul>

              <h3 className="text-xl font-bold text-white mt-6">Diagnostics</h3>
              <p>
                Residual plots help identify model inadequacies. Look for:
              </p>
              <ul className="list-disc list-inside space-y-2 ml-4">
                <li>Random scatter in residuals vs fitted (no patterns)</li>
                <li>Approximately normal residual distribution</li>
                <li>Constant variance across fitted values (homoscedasticity)</li>
              </ul>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default NonlinearRegression
