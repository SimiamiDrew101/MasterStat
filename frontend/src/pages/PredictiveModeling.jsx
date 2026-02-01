import { useState, useCallback } from 'react'
import axios from 'axios'
import Plot from 'react-plotly.js'
import { Upload, Cpu, BarChart3, GitCompare, Info, AlertTriangle, Clipboard, CheckCircle } from 'lucide-react'
import * as XLSX from 'xlsx'

const API_URL = import.meta.env.VITE_API_URL || ''

const PredictiveModeling = () => {
  // State
  const [activeTab, setActiveTab] = useState('data')
  const [tableData, setTableData] = useState(Array(20).fill(null).map(() => Array(5).fill('')))
  const [featureNames, setFeatureNames] = useState(['X1', 'X2', 'X3', 'X4'])
  const [targetName, setTargetName] = useState('Y')
  const [problemType, setProblemType] = useState('regression')
  const [selectedMethod, setSelectedMethod] = useState('random_forest')
  const [testSize, setTestSize] = useState(0.2)
  const [result, setResult] = useState(null)
  const [screeningResult, setScreeningResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  // Model-specific parameters
  const [treeParams, setTreeParams] = useState({ max_depth: '', min_samples_split: 2, min_samples_leaf: 1 })
  const [rfParams, setRfParams] = useState({ n_estimators: 100, max_depth: '', max_features: 'sqrt' })
  const [gbParams, setGbParams] = useState({ n_estimators: 100, learning_rate: 0.1, max_depth: 3, subsample: 1.0 })
  const [regParams, setRegParams] = useState({ method: 'ridge', alpha: 1.0, l1_ratio: 0.5 })

  // Methods info
  const methods = [
    {
      id: 'decision_tree',
      name: 'Decision Tree',
      icon: '🌳',
      description: 'Interpretable tree-based model',
      pros: 'Easy to interpret, visualize splits',
      cons: 'Prone to overfitting'
    },
    {
      id: 'random_forest',
      name: 'Random Forest',
      icon: '🌲',
      description: 'Ensemble of decision trees',
      pros: 'Robust, good accuracy, feature importance',
      cons: 'Less interpretable than single tree'
    },
    {
      id: 'gradient_boosting',
      name: 'Gradient Boosting',
      icon: '🚀',
      description: 'Sequential boosted trees',
      pros: 'Often best accuracy',
      cons: 'Slower training, many parameters'
    },
    {
      id: 'regularized',
      name: 'Regularized Regression',
      icon: '📐',
      description: 'Ridge, Lasso, or ElasticNet',
      pros: 'Feature selection (Lasso), fast',
      cons: 'Assumes linear relationships'
    }
  ]

  // Handle cell change
  const handleCellChange = useCallback((rowIndex, colIndex, value) => {
    setTableData(prev => {
      const newData = prev.map(row => [...row])
      newData[rowIndex][colIndex] = value
      if (rowIndex === newData.length - 1 && value.trim() !== '') {
        newData.push(Array(newData[0].length).fill(''))
      }
      return newData
    })
  }, [])

  // Handle key navigation
  const handleKeyDown = useCallback((e, rowIndex, colIndex) => {
    const numRows = tableData.length
    const numCols = tableData[0].length
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
        if (rowIndex === numRows - 1) {
          setTableData(prev => [...prev, Array(numCols).fill('')])
          newRow = numRows
        }
        break
      case 'ArrowLeft':
        if (e.target.selectionStart === 0) {
          e.preventDefault()
          newCol = Math.max(0, colIndex - 1)
        }
        return
      case 'ArrowRight':
        if (e.target.selectionStart === e.target.value.length) {
          e.preventDefault()
          newCol = Math.min(numCols - 1, colIndex + 1)
        }
        return
      case 'Tab':
        return
      default:
        return
    }

    setTimeout(() => {
      const input = document.getElementById(`pm-cell-${newRow}-${newCol}`)
      if (input) {
        input.focus()
        input.select()
      }
    }, 0)
  }, [tableData])

  // Handle paste
  const handlePaste = useCallback((e) => {
    e.preventDefault()
    const pastedData = e.clipboardData.getData('text')
    const rows = pastedData.trim().split('\n')
    const newTableData = []

    rows.forEach(row => {
      const values = row.split(/[\t,]/).map(v => v.trim())
      if (values.some(v => v !== '')) {
        newTableData.push(values)
      }
    })

    if (newTableData.length > 0) {
      const maxCols = Math.max(...newTableData.map(r => r.length))
      const paddedData = newTableData.map(row => {
        while (row.length < maxCols) row.push('')
        return row
      })
      while (paddedData.length < 20) {
        paddedData.push(Array(maxCols).fill(''))
      }
      paddedData.push(Array(maxCols).fill(''))

      setTableData(paddedData)

      // Update feature names
      const newFeatureNames = []
      for (let i = 0; i < maxCols - 1; i++) {
        newFeatureNames.push(`X${i + 1}`)
      }
      setFeatureNames(newFeatureNames)
    }
  }, [])

  // Add/remove columns
  const addColumn = () => {
    setTableData(prev => prev.map(row => [...row, '']))
    setFeatureNames(prev => [...prev, `X${prev.length + 1}`])
  }

  const removeColumn = () => {
    if (tableData[0].length > 2) {
      setTableData(prev => prev.map(row => row.slice(0, -1)))
      setFeatureNames(prev => prev.slice(0, -1))
    }
  }

  // Parse data
  const parseData = useCallback(() => {
    const X = []
    const y = []
    const numCols = tableData[0].length

    tableData.forEach(row => {
      const values = row.map(v => {
        const num = parseFloat(v)
        return isNaN(num) ? v : num
      })

      const hasData = values.slice(0, -1).some(v => v !== '' && !isNaN(v))
      const hasTarget = values[numCols - 1] !== '' && (problemType === 'classification' || !isNaN(values[numCols - 1]))

      if (hasData && hasTarget) {
        X.push(values.slice(0, -1).map(v => typeof v === 'number' ? v : 0))
        y.push(values[numCols - 1])
      }
    })

    return { X, y }
  }, [tableData, problemType])

  // Load example data
  const loadExampleData = (type) => {
    let exampleData, exampleFeatures

    if (type === 'regression') {
      // Boston-like housing data (simplified)
      exampleData = [
        [6.5, 65, 4.0, 24.0],
        [3.5, 78, 2.3, 21.6],
        [7.2, 45, 6.5, 34.7],
        [4.1, 54, 3.1, 33.4],
        [8.1, 38, 7.8, 36.2],
        [2.9, 82, 1.9, 28.7],
        [5.6, 66, 4.2, 22.9],
        [6.8, 52, 5.8, 27.1],
        [3.2, 75, 2.5, 16.5],
        [7.8, 42, 7.1, 18.9],
        [4.5, 58, 3.6, 15.0],
        [5.9, 48, 5.2, 18.9],
        [6.2, 55, 4.8, 21.7],
        [3.8, 72, 2.8, 20.4],
        [7.5, 40, 6.8, 18.2],
        [4.8, 62, 3.9, 13.6],
        [5.3, 50, 4.5, 19.6],
        [6.9, 44, 6.2, 15.2],
        [3.4, 80, 2.1, 14.5],
        [8.5, 35, 8.2, 17.0]
      ]
      exampleFeatures = ['Rooms', 'Age', 'Distance', 'Price']
      setProblemType('regression')
      setTargetName('Price')
    } else {
      // Iris-like classification data
      exampleData = [
        [5.1, 3.5, 1.4, 'setosa'],
        [4.9, 3.0, 1.4, 'setosa'],
        [4.7, 3.2, 1.3, 'setosa'],
        [5.0, 3.6, 1.4, 'setosa'],
        [5.4, 3.9, 1.7, 'setosa'],
        [4.6, 3.4, 1.4, 'setosa'],
        [5.0, 3.4, 1.5, 'setosa'],
        [7.0, 3.2, 4.7, 'versicolor'],
        [6.4, 3.2, 4.5, 'versicolor'],
        [6.9, 3.1, 4.9, 'versicolor'],
        [5.5, 2.3, 4.0, 'versicolor'],
        [6.5, 2.8, 4.6, 'versicolor'],
        [5.7, 2.8, 4.5, 'versicolor'],
        [6.3, 3.3, 6.0, 'virginica'],
        [5.8, 2.7, 5.1, 'virginica'],
        [7.1, 3.0, 5.9, 'virginica'],
        [6.3, 2.9, 5.6, 'virginica'],
        [6.5, 3.0, 5.8, 'virginica'],
        [7.6, 3.0, 6.6, 'virginica'],
        [4.9, 2.5, 4.5, 'versicolor']
      ]
      exampleFeatures = ['SepalLen', 'SepalWid', 'PetalLen', 'Species']
      setProblemType('classification')
      setTargetName('Species')
    }

    const paddedData = exampleData.map(row => row.map(v => String(v)))
    while (paddedData.length < 20) {
      paddedData.push(Array(exampleData[0].length).fill(''))
    }
    paddedData.push(Array(exampleData[0].length).fill(''))

    setTableData(paddedData)
    setFeatureNames(exampleFeatures.slice(0, -1))
  }

  // Upload file
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
          setError('File must have header row and data')
          return
        }

        // First row is headers
        const headers = jsonData[0].map(h => String(h || ''))
        const dataRows = jsonData.slice(1).map(row =>
          row.map(v => v === null || v === undefined ? '' : String(v))
        )

        // Ensure consistent column count
        const numCols = headers.length
        const paddedData = dataRows.map(row => {
          while (row.length < numCols) row.push('')
          return row.slice(0, numCols)
        })

        while (paddedData.length < 20) {
          paddedData.push(Array(numCols).fill(''))
        }
        paddedData.push(Array(numCols).fill(''))

        setTableData(paddedData)
        setFeatureNames(headers.slice(0, -1))
        setTargetName(headers[headers.length - 1] || 'Y')
        setError('')
      } catch (err) {
        setError('Failed to read file: ' + err.message)
      }
    }
    reader.readAsArrayBuffer(file)
  }

  // Train model
  const handleTrainModel = async () => {
    const { X, y } = parseData()

    if (X.length < 10) {
      setError('Need at least 10 valid data rows')
      return
    }

    setLoading(true)
    setError('')

    try {
      let endpoint = ''
      let requestData = {
        X,
        y,
        feature_names: featureNames,
        problem_type: problemType,
        test_size: testSize,
        random_state: 42
      }

      if (selectedMethod === 'decision_tree') {
        endpoint = `${API_URL}/api/ml/decision-tree`
        requestData.max_depth = treeParams.max_depth ? parseInt(treeParams.max_depth) : null
        requestData.min_samples_split = treeParams.min_samples_split
        requestData.min_samples_leaf = treeParams.min_samples_leaf
      } else if (selectedMethod === 'random_forest') {
        endpoint = `${API_URL}/api/ml/random-forest`
        requestData.n_estimators = rfParams.n_estimators
        requestData.max_depth = rfParams.max_depth ? parseInt(rfParams.max_depth) : null
        requestData.max_features = rfParams.max_features
      } else if (selectedMethod === 'gradient_boosting') {
        endpoint = `${API_URL}/api/ml/gradient-boosting`
        requestData.n_estimators = gbParams.n_estimators
        requestData.learning_rate = gbParams.learning_rate
        requestData.max_depth = gbParams.max_depth
        requestData.subsample = gbParams.subsample
      } else if (selectedMethod === 'regularized') {
        endpoint = `${API_URL}/api/ml/regularized-regression`
        requestData.method = regParams.method
        requestData.alpha = regParams.alpha
        requestData.l1_ratio = regParams.l1_ratio
      }

      const response = await axios.post(endpoint, requestData)
      setResult(response.data)
      setActiveTab('results')
    } catch (err) {
      setError('Training failed: ' + (err.response?.data?.detail || err.message))
    } finally {
      setLoading(false)
    }
  }

  // Run model screening
  const handleModelScreening = async () => {
    const { X, y } = parseData()

    if (X.length < 10) {
      setError('Need at least 10 valid data rows')
      return
    }

    setLoading(true)
    setError('')

    try {
      const response = await axios.post(`${API_URL}/api/ml/model-screening`, {
        X,
        y,
        feature_names: featureNames,
        problem_type: problemType,
        test_size: testSize,
        methods: ['decision_tree', 'random_forest', 'gradient_boosting', 'ridge', 'lasso'],
        cv_folds: 5
      })
      setScreeningResult(response.data)
      setActiveTab('compare')
    } catch (err) {
      setError('Model screening failed: ' + (err.response?.data?.detail || err.message))
    } finally {
      setLoading(false)
    }
  }

  // Get metric color
  const getMetricColor = (value, metric) => {
    if (metric === 'r2' || metric === 'accuracy') {
      if (value >= 0.9) return 'text-green-400'
      if (value >= 0.7) return 'text-blue-400'
      if (value >= 0.5) return 'text-yellow-400'
      return 'text-red-400'
    }
    return 'text-white'
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-pink-600 text-transparent bg-clip-text">
            Predictive Modeling
          </h1>
          <p className="text-gray-300 text-lg">
            Machine learning models for regression and classification tasks
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
            <Cpu size={20} />
            <span>2. Model</span>
          </button>
          <button
            onClick={() => setActiveTab('results')}
            disabled={!result}
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
            onClick={() => setActiveTab('compare')}
            disabled={!screeningResult}
            className={`flex-1 px-4 py-3 rounded-lg font-medium transition-all duration-200 flex items-center justify-center space-x-2 ${
              activeTab === 'compare'
                ? 'bg-purple-600 text-white shadow-lg'
                : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700 disabled:opacity-40'
            }`}
          >
            <GitCompare size={20} />
            <span>4. Compare</span>
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
              <div className="flex gap-4">
                <button
                  onClick={() => loadExampleData('regression')}
                  className="px-6 py-3 bg-gradient-to-br from-blue-600 to-blue-700 hover:from-blue-500 hover:to-blue-600 text-white rounded-xl font-semibold transition-all"
                >
                  Regression Example (Housing)
                </button>
                <button
                  onClick={() => loadExampleData('classification')}
                  className="px-6 py-3 bg-gradient-to-br from-green-600 to-green-700 hover:from-green-500 hover:to-green-600 text-white rounded-xl font-semibold transition-all"
                >
                  Classification Example (Iris)
                </button>
              </div>
            </div>

            {/* Problem Type */}
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <h2 className="text-2xl font-bold mb-4 text-gray-100">Problem Type</h2>
              <div className="flex gap-4">
                <button
                  onClick={() => setProblemType('regression')}
                  className={`flex-1 px-6 py-4 rounded-xl font-semibold transition-all border-2 ${
                    problemType === 'regression'
                      ? 'border-purple-500 bg-purple-900/30 text-white'
                      : 'border-slate-600 bg-slate-700/30 text-gray-300 hover:border-slate-500'
                  }`}
                >
                  <div className="text-2xl mb-2">📈</div>
                  <div className="text-lg">Regression</div>
                  <div className="text-sm text-gray-400">Predict continuous values</div>
                </button>
                <button
                  onClick={() => setProblemType('classification')}
                  className={`flex-1 px-6 py-4 rounded-xl font-semibold transition-all border-2 ${
                    problemType === 'classification'
                      ? 'border-purple-500 bg-purple-900/30 text-white'
                      : 'border-slate-600 bg-slate-700/30 text-gray-300 hover:border-slate-500'
                  }`}
                >
                  <div className="text-2xl mb-2">🏷️</div>
                  <div className="text-lg">Classification</div>
                  <div className="text-sm text-gray-400">Predict categories</div>
                </button>
              </div>
            </div>

            {/* Data Entry */}
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-2xl font-bold text-gray-100">Enter Data</h2>
                <div className="flex gap-2">
                  <button
                    onClick={removeColumn}
                    disabled={tableData[0]?.length <= 2}
                    className="px-3 py-1 bg-slate-700 hover:bg-slate-600 text-gray-300 rounded text-sm disabled:opacity-50"
                  >
                    - Column
                  </button>
                  <button
                    onClick={addColumn}
                    className="px-3 py-1 bg-slate-700 hover:bg-slate-600 text-gray-300 rounded text-sm"
                  >
                    + Column
                  </button>
                  <button
                    onClick={() => navigator.clipboard.readText().then(text => {
                      const rows = text.trim().split('\n')
                      const newTableData = rows.map(r => r.split(/[\t,]/).map(v => v.trim()))
                      const maxCols = Math.max(...newTableData.map(r => r.length))
                      const padded = newTableData.map(r => { while(r.length < maxCols) r.push(''); return r })
                      while(padded.length < 20) padded.push(Array(maxCols).fill(''))
                      padded.push(Array(maxCols).fill(''))
                      setTableData(padded)
                      setFeatureNames(Array(maxCols - 1).fill(null).map((_, i) => `X${i+1}`))
                    })}
                    className="flex items-center gap-1 px-3 py-1 bg-slate-700 hover:bg-slate-600 text-gray-300 rounded text-sm"
                  >
                    <Clipboard size={14} /> Paste
                  </button>
                </div>
              </div>
              <p className="text-sm text-gray-400 mb-3">
                Features in first columns, target in last column. Use arrow keys to navigate.
              </p>
              <div className="overflow-x-auto max-h-80 border border-slate-600 rounded-lg">
                <table className="w-full">
                  <thead className="sticky top-0 bg-slate-700">
                    <tr>
                      <th className="w-12 px-2 py-2 text-left text-xs font-semibold text-gray-400 border-b border-slate-600">#</th>
                      {featureNames.map((name, idx) => (
                        <th key={idx} className="px-2 py-2 text-center text-xs font-semibold text-gray-400 border-b border-slate-600">
                          <input
                            type="text"
                            value={name}
                            onChange={(e) => {
                              const newNames = [...featureNames]
                              newNames[idx] = e.target.value
                              setFeatureNames(newNames)
                            }}
                            className="w-full bg-transparent text-center text-blue-400 focus:outline-none"
                          />
                        </th>
                      ))}
                      <th className="px-2 py-2 text-center text-xs font-semibold text-gray-400 border-b border-slate-600">
                        <input
                          type="text"
                          value={targetName}
                          onChange={(e) => setTargetName(e.target.value)}
                          className="w-full bg-transparent text-center text-green-400 focus:outline-none"
                        />
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {tableData.map((row, rowIdx) => (
                      <tr key={rowIdx} className="border-b border-slate-700/50 hover:bg-slate-700/30">
                        <td className="px-2 py-1 text-xs text-gray-500 font-mono">{rowIdx + 1}</td>
                        {row.map((cell, colIdx) => (
                          <td key={colIdx} className="p-0">
                            <input
                              id={`pm-cell-${rowIdx}-${colIdx}`}
                              type="text"
                              value={cell}
                              onChange={(e) => handleCellChange(rowIdx, colIdx, e.target.value)}
                              onKeyDown={(e) => handleKeyDown(e, rowIdx, colIdx)}
                              onPaste={colIdx === 0 ? handlePaste : undefined}
                              className={`w-full px-2 py-1 bg-transparent text-center focus:bg-slate-700/50 focus:outline-none focus:ring-1 focus:ring-purple-500/50 font-mono text-xs ${
                                colIdx === row.length - 1 ? 'text-green-400' : 'text-gray-100'
                              }`}
                            />
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-sm text-gray-400 mt-2">
                {parseData().X.length} valid rows detected
              </p>
            </div>

            {/* File Upload */}
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <h2 className="text-2xl font-bold mb-4 text-gray-100">Or Upload File</h2>
              <input
                type="file"
                accept=".xlsx,.xls,.csv"
                onChange={handleFileUpload}
                className="block w-full text-sm text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-purple-600 file:text-white hover:file:bg-purple-700 cursor-pointer"
              />
              <p className="text-gray-400 text-sm mt-2">
                First row = headers, last column = target variable
              </p>
            </div>
          </div>
        )}

        {/* Model Tab */}
        {activeTab === 'model' && (
          <div className="space-y-6">
            {/* Method Selection */}
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <h2 className="text-2xl font-bold mb-4 text-gray-100">Select Model</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {methods.map(method => (
                  <button
                    key={method.id}
                    onClick={() => setSelectedMethod(method.id)}
                    className={`p-4 rounded-lg border-2 text-left transition-all ${
                      selectedMethod === method.id
                        ? 'border-purple-500 bg-purple-900/30'
                        : 'border-slate-600 bg-slate-700/30 hover:border-slate-500'
                    }`}
                  >
                    <div className="text-3xl mb-2">{method.icon}</div>
                    <h3 className="font-bold text-gray-100">{method.name}</h3>
                    <p className="text-gray-400 text-xs mt-1">{method.description}</p>
                  </button>
                ))}
              </div>
            </div>

            {/* Model Parameters */}
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <h2 className="text-2xl font-bold mb-4 text-gray-100">Model Parameters</h2>

              {/* Decision Tree Parameters */}
              {selectedMethod === 'decision_tree' && (
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Max Depth</label>
                    <input
                      type="number"
                      min="1"
                      value={treeParams.max_depth}
                      onChange={(e) => setTreeParams({ ...treeParams, max_depth: e.target.value })}
                      placeholder="None (unlimited)"
                      className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Min Samples Split</label>
                    <input
                      type="number"
                      min="2"
                      value={treeParams.min_samples_split}
                      onChange={(e) => setTreeParams({ ...treeParams, min_samples_split: parseInt(e.target.value) })}
                      className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Min Samples Leaf</label>
                    <input
                      type="number"
                      min="1"
                      value={treeParams.min_samples_leaf}
                      onChange={(e) => setTreeParams({ ...treeParams, min_samples_leaf: parseInt(e.target.value) })}
                      className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                </div>
              )}

              {/* Random Forest Parameters */}
              {selectedMethod === 'random_forest' && (
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Number of Trees</label>
                    <input
                      type="number"
                      min="10"
                      value={rfParams.n_estimators}
                      onChange={(e) => setRfParams({ ...rfParams, n_estimators: parseInt(e.target.value) })}
                      className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Max Depth</label>
                    <input
                      type="number"
                      min="1"
                      value={rfParams.max_depth}
                      onChange={(e) => setRfParams({ ...rfParams, max_depth: e.target.value })}
                      placeholder="None (unlimited)"
                      className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Max Features</label>
                    <select
                      value={rfParams.max_features}
                      onChange={(e) => setRfParams({ ...rfParams, max_features: e.target.value })}
                      className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    >
                      <option value="sqrt">sqrt</option>
                      <option value="log2">log2</option>
                      <option value="1.0">All features</option>
                    </select>
                  </div>
                </div>
              )}

              {/* Gradient Boosting Parameters */}
              {selectedMethod === 'gradient_boosting' && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Number of Trees</label>
                    <input
                      type="number"
                      min="10"
                      value={gbParams.n_estimators}
                      onChange={(e) => setGbParams({ ...gbParams, n_estimators: parseInt(e.target.value) })}
                      className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Learning Rate</label>
                    <input
                      type="number"
                      step="0.01"
                      min="0.01"
                      max="1"
                      value={gbParams.learning_rate}
                      onChange={(e) => setGbParams({ ...gbParams, learning_rate: parseFloat(e.target.value) })}
                      className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Max Depth</label>
                    <input
                      type="number"
                      min="1"
                      value={gbParams.max_depth}
                      onChange={(e) => setGbParams({ ...gbParams, max_depth: parseInt(e.target.value) })}
                      className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Subsample</label>
                    <input
                      type="number"
                      step="0.1"
                      min="0.1"
                      max="1"
                      value={gbParams.subsample}
                      onChange={(e) => setGbParams({ ...gbParams, subsample: parseFloat(e.target.value) })}
                      className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                </div>
              )}

              {/* Regularized Regression Parameters */}
              {selectedMethod === 'regularized' && (
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Method</label>
                    <select
                      value={regParams.method}
                      onChange={(e) => setRegParams({ ...regParams, method: e.target.value })}
                      className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    >
                      <option value="ridge">Ridge (L2)</option>
                      <option value="lasso">Lasso (L1)</option>
                      <option value="elasticnet">Elastic Net</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Alpha (Regularization)</label>
                    <input
                      type="number"
                      step="0.1"
                      min="0.01"
                      value={regParams.alpha}
                      onChange={(e) => setRegParams({ ...regParams, alpha: parseFloat(e.target.value) })}
                      className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                  {regParams.method === 'elasticnet' && (
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">L1 Ratio</label>
                      <input
                        type="number"
                        step="0.1"
                        min="0"
                        max="1"
                        value={regParams.l1_ratio}
                        onChange={(e) => setRegParams({ ...regParams, l1_ratio: parseFloat(e.target.value) })}
                        className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                      />
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Train Settings */}
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <h2 className="text-2xl font-bold mb-4 text-gray-100">Training Settings</h2>
              <div className="max-w-xs">
                <label className="block text-sm font-medium text-gray-300 mb-2">Test Set Size</label>
                <input
                  type="range"
                  min="0.1"
                  max="0.4"
                  step="0.05"
                  value={testSize}
                  onChange={(e) => setTestSize(parseFloat(e.target.value))}
                  className="w-full"
                />
                <p className="text-gray-400 text-sm">{(testSize * 100).toFixed(0)}% for testing, {((1 - testSize) * 100).toFixed(0)}% for training</p>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-4">
              <button
                onClick={handleTrainModel}
                disabled={loading || parseData().X.length < 10}
                className="flex-1 px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white rounded-xl font-bold text-lg shadow-lg transition-all disabled:opacity-50"
              >
                {loading ? 'Training...' : 'Train Model'}
              </button>
              <button
                onClick={handleModelScreening}
                disabled={loading || parseData().X.length < 10}
                className="flex-1 px-8 py-4 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white rounded-xl font-bold text-lg shadow-lg transition-all disabled:opacity-50"
              >
                {loading ? 'Comparing...' : 'Compare All Models'}
              </button>
            </div>
          </div>
        )}

        {/* Results Tab */}
        {activeTab === 'results' && result && (
          <div className="space-y-6">
            {/* Model Summary */}
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <h2 className="text-2xl font-bold mb-4 text-gray-100">
                {result.model_type?.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase())} Results
              </h2>
              <div className="flex items-center gap-2 text-gray-400 mb-4">
                <span className="px-3 py-1 bg-slate-700 rounded-full text-sm">{result.problem_type}</span>
                <span className="px-3 py-1 bg-slate-700 rounded-full text-sm">
                  {result.data_info?.n_train} train / {result.data_info?.n_test} test samples
                </span>
              </div>

              {/* Metrics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {result.problem_type === 'regression' ? (
                  <>
                    <div className="bg-slate-700/30 rounded-lg p-4">
                      <p className="text-gray-400 text-sm mb-1">Test R²</p>
                      <p className={`text-3xl font-bold ${getMetricColor(result.test_metrics?.r2, 'r2')}`}>
                        {result.test_metrics?.r2?.toFixed(3)}
                      </p>
                    </div>
                    <div className="bg-slate-700/30 rounded-lg p-4">
                      <p className="text-gray-400 text-sm mb-1">Train R²</p>
                      <p className="text-2xl font-bold text-white">{result.train_metrics?.r2?.toFixed(3)}</p>
                    </div>
                    <div className="bg-slate-700/30 rounded-lg p-4">
                      <p className="text-gray-400 text-sm mb-1">RMSE</p>
                      <p className="text-2xl font-bold text-white">{result.test_metrics?.rmse?.toFixed(3)}</p>
                    </div>
                    <div className="bg-slate-700/30 rounded-lg p-4">
                      <p className="text-gray-400 text-sm mb-1">MAE</p>
                      <p className="text-2xl font-bold text-white">{result.test_metrics?.mae?.toFixed(3)}</p>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="bg-slate-700/30 rounded-lg p-4">
                      <p className="text-gray-400 text-sm mb-1">Test Accuracy</p>
                      <p className={`text-3xl font-bold ${getMetricColor(result.test_metrics?.accuracy, 'accuracy')}`}>
                        {(result.test_metrics?.accuracy * 100)?.toFixed(1)}%
                      </p>
                    </div>
                    <div className="bg-slate-700/30 rounded-lg p-4">
                      <p className="text-gray-400 text-sm mb-1">Precision</p>
                      <p className="text-2xl font-bold text-white">{result.test_metrics?.precision?.toFixed(3)}</p>
                    </div>
                    <div className="bg-slate-700/30 rounded-lg p-4">
                      <p className="text-gray-400 text-sm mb-1">Recall</p>
                      <p className="text-2xl font-bold text-white">{result.test_metrics?.recall?.toFixed(3)}</p>
                    </div>
                    <div className="bg-slate-700/30 rounded-lg p-4">
                      <p className="text-gray-400 text-sm mb-1">F1 Score</p>
                      <p className="text-2xl font-bold text-white">{result.test_metrics?.f1_score?.toFixed(3)}</p>
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* Feature Importance */}
            {result.feature_importance && (
              <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
                <h2 className="text-2xl font-bold mb-4 text-gray-100">Feature Importance</h2>
                <Plot
                  data={[{
                    x: result.feature_importance.importances.slice().reverse(),
                    y: result.feature_importance.features.slice().reverse(),
                    type: 'bar',
                    orientation: 'h',
                    marker: {
                      color: result.feature_importance.importances.slice().reverse().map((_, i, arr) =>
                        `rgba(168, 85, 247, ${0.4 + 0.6 * (i / arr.length)})`
                      )
                    }
                  }]}
                  layout={{
                    xaxis: { title: 'Importance', gridcolor: '#334155' },
                    yaxis: { gridcolor: '#334155' },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(15,23,42,0.5)',
                    font: { color: '#e2e8f0' },
                    margin: { l: 100, r: 20, t: 20, b: 50 },
                    autosize: true
                  }}
                  style={{ width: '100%', height: '300px' }}
                  useResizeHandler={true}
                />
              </div>
            )}

            {/* Coefficients for regularized regression */}
            {result.coefficients && (
              <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
                <h2 className="text-2xl font-bold mb-4 text-gray-100">Coefficients</h2>
                <p className="text-gray-400 mb-4">
                  {result.n_nonzero_coef} non-zero coefficients out of {result.coefficients.features.length}
                </p>
                <Plot
                  data={[{
                    x: result.coefficients.values.slice().reverse(),
                    y: result.coefficients.features.slice().reverse(),
                    type: 'bar',
                    orientation: 'h',
                    marker: {
                      color: result.coefficients.values.slice().reverse().map(v => v >= 0 ? '#22c55e' : '#ef4444')
                    }
                  }]}
                  layout={{
                    xaxis: { title: 'Coefficient Value', gridcolor: '#334155', zeroline: true, zerolinecolor: '#ffffff' },
                    yaxis: { gridcolor: '#334155' },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(15,23,42,0.5)',
                    font: { color: '#e2e8f0' },
                    margin: { l: 100, r: 20, t: 20, b: 50 },
                    autosize: true
                  }}
                  style={{ width: '100%', height: '300px' }}
                  useResizeHandler={true}
                />
              </div>
            )}

            {/* Confusion Matrix for classification */}
            {result.problem_type === 'classification' && result.test_metrics?.confusion_matrix && (
              <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
                <h2 className="text-2xl font-bold mb-4 text-gray-100">Confusion Matrix</h2>
                <Plot
                  data={[{
                    z: result.test_metrics.confusion_matrix,
                    x: result.class_labels || result.test_metrics.classes || result.test_metrics.confusion_matrix.map((_, i) => `Class ${i}`),
                    y: result.class_labels || result.test_metrics.classes || result.test_metrics.confusion_matrix.map((_, i) => `Class ${i}`),
                    type: 'heatmap',
                    colorscale: 'Purples',
                    showscale: true
                  }]}
                  layout={{
                    xaxis: { title: 'Predicted', gridcolor: '#334155' },
                    yaxis: { title: 'Actual', gridcolor: '#334155', autorange: 'reversed' },
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

            {/* ROC Curve for binary classification */}
            {result.test_metrics?.roc_curve && (
              <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
                <h2 className="text-2xl font-bold mb-4 text-gray-100">ROC Curve (AUC = {result.test_metrics.auc_roc?.toFixed(3)})</h2>
                <Plot
                  data={[
                    {
                      x: result.test_metrics.roc_curve.fpr,
                      y: result.test_metrics.roc_curve.tpr,
                      type: 'scatter',
                      mode: 'lines',
                      line: { color: '#a855f7', width: 2 },
                      name: 'ROC Curve'
                    },
                    {
                      x: [0, 1],
                      y: [0, 1],
                      type: 'scatter',
                      mode: 'lines',
                      line: { color: '#64748b', width: 1, dash: 'dash' },
                      name: 'Random'
                    }
                  ]}
                  layout={{
                    xaxis: { title: 'False Positive Rate', gridcolor: '#334155', range: [0, 1] },
                    yaxis: { title: 'True Positive Rate', gridcolor: '#334155', range: [0, 1] },
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

            {/* Tree Structure for decision tree */}
            {result.tree_structure && (
              <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
                <h2 className="text-2xl font-bold mb-4 text-gray-100">Tree Structure</h2>
                <p className="text-gray-400 mb-2">
                  Depth: {result.parameters?.max_depth} | Leaves: {result.parameters?.n_leaves}
                </p>
                <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm overflow-x-auto">
                  <TreeNode node={result.tree_structure} depth={0} />
                </div>
              </div>
            )}
          </div>
        )}

        {/* Compare Tab */}
        {activeTab === 'compare' && screeningResult && (
          <div className="space-y-6">
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <h2 className="text-2xl font-bold mb-4 text-gray-100">Model Comparison</h2>
              <p className="text-gray-400 mb-4">
                Metric: {screeningResult.scoring_metric} | {screeningResult.cv_folds}-fold cross-validation
              </p>

              {/* Best Model */}
              {screeningResult.best_method && (
                <div className="mb-6 p-4 bg-green-900/30 border border-green-700/50 rounded-lg flex items-center gap-3">
                  <CheckCircle className="text-green-400" size={24} />
                  <div>
                    <p className="text-green-400 font-semibold">Best Model: {screeningResult.best_method.replace('_', ' ')}</p>
                    <p className="text-gray-300 text-sm">Based on cross-validation {screeningResult.scoring_metric}</p>
                  </div>
                </div>
              )}

              {/* Comparison Chart */}
              <Plot
                data={[{
                  x: screeningResult.results.filter(r => r.cv_mean !== undefined).map(r => r.method.replace('_', ' ')),
                  y: screeningResult.results.filter(r => r.cv_mean !== undefined).map(r => r.cv_mean),
                  error_y: {
                    type: 'data',
                    array: screeningResult.results.filter(r => r.cv_std !== undefined).map(r => r.cv_std),
                    visible: true
                  },
                  type: 'bar',
                  marker: {
                    color: screeningResult.results.filter(r => r.cv_mean !== undefined).map(r =>
                      r.method === screeningResult.best_method ? '#22c55e' : '#a855f7'
                    )
                  }
                }]}
                layout={{
                  xaxis: { title: 'Model', gridcolor: '#334155' },
                  yaxis: { title: `CV ${screeningResult.scoring_metric}`, gridcolor: '#334155' },
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(15,23,42,0.5)',
                  font: { color: '#e2e8f0' },
                  autosize: true
                }}
                style={{ width: '100%', height: '400px' }}
                useResizeHandler={true}
              />

              {/* Results Table */}
              <div className="mt-6 overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-slate-700">
                      <th className="px-4 py-2 text-left text-gray-300">Rank</th>
                      <th className="px-4 py-2 text-left text-gray-300">Model</th>
                      <th className="px-4 py-2 text-right text-gray-300">CV Mean</th>
                      <th className="px-4 py-2 text-right text-gray-300">CV Std</th>
                      <th className="px-4 py-2 text-right text-gray-300">Test Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {screeningResult.results.filter(r => r.cv_mean !== undefined).map((r, idx) => (
                      <tr key={r.method} className={`border-b border-slate-700 ${r.method === screeningResult.best_method ? 'bg-green-900/20' : ''}`}>
                        <td className="px-4 py-2 text-gray-400">{idx + 1}</td>
                        <td className="px-4 py-2 text-gray-100 font-medium">{r.method.replace('_', ' ')}</td>
                        <td className="px-4 py-2 text-right text-white font-bold">{r.cv_mean?.toFixed(4)}</td>
                        <td className="px-4 py-2 text-right text-gray-400">±{r.cv_std?.toFixed(4)}</td>
                        <td className="px-4 py-2 text-right text-gray-300">{r.test_score?.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Info Tab */}
        {activeTab === 'info' && (
          <div className="space-y-6">
            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <h2 className="text-2xl font-bold mb-4 text-gray-100">About Predictive Modeling</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {methods.map(method => (
                  <div key={method.id} className="bg-slate-700/30 rounded-lg p-4">
                    <div className="flex items-center gap-3 mb-3">
                      <span className="text-3xl">{method.icon}</span>
                      <h3 className="text-xl font-bold text-gray-100">{method.name}</h3>
                    </div>
                    <p className="text-gray-300 mb-3">{method.description}</p>
                    <p className="text-green-400 text-sm"><strong>Pros:</strong> {method.pros}</p>
                    <p className="text-red-400 text-sm"><strong>Cons:</strong> {method.cons}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-slate-800/50 rounded-2xl p-6 backdrop-blur-sm border border-slate-700/50">
              <h2 className="text-2xl font-bold mb-4 text-gray-100">Metrics Guide</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-lg font-bold text-gray-100 mb-3">Regression Metrics</h3>
                  <ul className="space-y-2 text-gray-300">
                    <li><strong className="text-white">R²:</strong> Variance explained (0-1, higher better)</li>
                    <li><strong className="text-white">RMSE:</strong> Root mean squared error (lower better)</li>
                    <li><strong className="text-white">MAE:</strong> Mean absolute error (lower better)</li>
                  </ul>
                </div>
                <div>
                  <h3 className="text-lg font-bold text-gray-100 mb-3">Classification Metrics</h3>
                  <ul className="space-y-2 text-gray-300">
                    <li><strong className="text-white">Accuracy:</strong> Correct predictions / Total</li>
                    <li><strong className="text-white">Precision:</strong> True positives / Predicted positives</li>
                    <li><strong className="text-white">Recall:</strong> True positives / Actual positives</li>
                    <li><strong className="text-white">F1:</strong> Harmonic mean of precision & recall</li>
                    <li><strong className="text-white">AUC-ROC:</strong> Area under ROC curve (0.5-1)</li>
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

// Tree visualization component
const TreeNode = ({ node, depth }) => {
  if (!node || depth > 5) return null

  const indent = '  '.repeat(depth)

  if (node.type === 'leaf') {
    return (
      <div className="text-green-400">
        {indent}└─ Predict: {typeof node.prediction === 'number' ? node.prediction.toFixed(2) : node.prediction} (n={node.samples})
      </div>
    )
  }

  if (node.type === 'split') {
    return (
      <div>
        <div className="text-blue-400">
          {indent}├─ {node.feature} {'<='} {node.threshold?.toFixed(3)} (n={node.samples})
        </div>
        {node.left && <TreeNode node={node.left} depth={depth + 1} />}
        {node.right && (
          <>
            <div className="text-blue-400">{indent}├─ {node.feature} {'>'} {node.threshold?.toFixed(3)}</div>
            <TreeNode node={node.right} depth={depth + 1} />
          </>
        )}
      </div>
    )
  }

  return null
}

export default PredictiveModeling
