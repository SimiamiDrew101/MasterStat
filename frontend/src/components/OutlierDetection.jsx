import { useState, useEffect } from 'react'
import Plot from 'react-plotly.js'
import { AlertTriangle, Eye, EyeOff, Trash2, Check, X } from 'lucide-react'
import { getPlotlyConfig, getPlotlyLayout } from '../utils/plotlyConfig'
import axios from 'axios'

const OutlierDetection = ({
  data,
  columnName = 'Response',
  onApply,
  onCancel
}) => {
  const [selectedMethod, setSelectedMethod] = useState('zscore')
  const [detectionParams, setDetectionParams] = useState({
    threshold: 3.0,
    contamination: 0.1
  })
  const [outlierResults, setOutlierResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [manualSelection, setManualSelection] = useState(new Set())
  const [showOutliers, setShowOutliers] = useState(true)

  // Detection methods
  const methods = [
    {
      value: 'zscore',
      label: 'Z-Score',
      description: 'Flag points beyond threshold standard deviations from mean',
      params: ['threshold']
    },
    {
      value: 'iqr',
      label: 'IQR (Interquartile Range)',
      description: 'Flag points beyond 1.5×IQR from quartiles',
      params: []
    },
    {
      value: 'isolation_forest',
      label: 'Isolation Forest',
      description: 'Machine learning algorithm for anomaly detection',
      params: ['contamination']
    },
    {
      value: 'elliptic_envelope',
      label: 'Elliptic Envelope',
      description: 'Robust covariance estimation for outlier detection',
      params: ['contamination']
    },
    {
      value: 'dbscan',
      label: 'DBSCAN',
      description: 'Density-based clustering to identify outliers',
      params: []
    }
  ]

  // Detect outliers using backend API
  const detectOutliers = async () => {
    if (!data || data.length === 0) {
      console.log('OutlierDetection: No data to process')
      return
    }

    console.log('OutlierDetection: Detecting outliers with method:', selectedMethod, 'data length:', data.length)
    setLoading(true)
    setError(null)

    try {
      const response = await axios.post('/api/preprocessing/detect-outliers', {
        data: data,
        method: selectedMethod,
        threshold: detectionParams.threshold,
        contamination: detectionParams.contamination
      })

      console.log('OutlierDetection: Detection successful', response.data)
      setOutlierResults(response.data)
      // Initialize manual selection with detected outliers
      setManualSelection(new Set(response.data.outlier_indices))
    } catch (err) {
      const errorMsg = err.response?.data?.detail || err.message || 'Failed to detect outliers'
      setError(errorMsg)
      console.error('Outlier detection error:', err)
      console.error('Error details:', errorMsg)
    } finally {
      setLoading(false)
    }
  }

  // Run detection when method or params change
  useEffect(() => {
    console.log('OutlierDetection: useEffect triggered', {
      hasData: !!data,
      dataLength: data?.length,
      method: selectedMethod,
      threshold: detectionParams.threshold,
      contamination: detectionParams.contamination
    })

    if (data && data.length > 0) {
      detectOutliers()
    }
  }, [selectedMethod, detectionParams.threshold, detectionParams.contamination, data])

  // Toggle manual selection
  const togglePoint = (index) => {
    const newSelection = new Set(manualSelection)
    if (newSelection.has(index)) {
      newSelection.delete(index)
    } else {
      newSelection.add(index)
    }
    setManualSelection(newSelection)
  }

  // Select all detected outliers
  const selectAll = () => {
    if (outlierResults) {
      setManualSelection(new Set(outlierResults.outlier_indices))
    }
  }

  // Clear all selections
  const clearAll = () => {
    setManualSelection(new Set())
  }

  // Apply changes (remove outliers)
  const handleApply = () => {
    const selectedIndices = Array.from(manualSelection)
    const cleanedData = data.filter((_, idx) => !selectedIndices.includes(idx))

    if (onApply) {
      onApply(cleanedData, {
        method: selectedMethod,
        removedCount: selectedIndices.length,
        removedIndices: selectedIndices
      })
    }
  }

  // Handle parameter change
  const handleParamChange = (param, value) => {
    setDetectionParams(prev => ({
      ...prev,
      [param]: parseFloat(value) || 0
    }))
  }

  // Create scatter plot with outliers highlighted
  const createScatterPlot = () => {
    if (!data || data.length === 0) return null

    const indices = data.map((_, i) => i)
    const manualIndices = Array.from(manualSelection)

    // Separate inliers and outliers
    const inlierIndices = indices.filter(i => !manualIndices.includes(i))
    const outlierIndices = manualIndices

    const traces = [
      // Inliers
      {
        x: inlierIndices,
        y: inlierIndices.map(i => data[i]),
        type: 'scatter',
        mode: 'markers',
        name: 'Inliers',
        marker: {
          color: '#06b6d4',
          size: 8,
          opacity: 0.7
        }
      }
    ]

    // Add outliers if they should be shown
    if (showOutliers && outlierIndices.length > 0) {
      traces.push({
        x: outlierIndices,
        y: outlierIndices.map(i => data[i]),
        type: 'scatter',
        mode: 'markers',
        name: 'Outliers',
        marker: {
          color: '#f59e0b',
          size: 10,
          symbol: 'x',
          line: { color: '#d97706', width: 2 }
        }
      })
    }

    return traces
  }

  // Create box plot
  const createBoxPlot = () => {
    if (!data || data.length === 0) return null

    const manualIndices = Array.from(manualSelection)
    const inlierData = data.filter((_, i) => !manualIndices.includes(i))

    return [
      {
        y: inlierData,
        type: 'box',
        name: 'Inliers',
        marker: { color: '#06b6d4' },
        boxmean: 'sd'
      },
      {
        y: manualIndices.map(i => data[i]),
        type: 'scatter',
        mode: 'markers',
        name: 'Outliers',
        marker: {
          color: '#f59e0b',
          size: 10,
          symbol: 'x'
        }
      }
    ]
  }

  const currentMethod = methods.find(m => m.value === selectedMethod)

  return (
    <div className="space-y-6">
      {/* Method Selection */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <div className="flex items-center gap-2 mb-4">
          <AlertTriangle className="w-5 h-5 text-orange-400" />
          <h3 className="text-xl font-bold text-gray-100">Outlier Detection</h3>
        </div>

        <div className="space-y-4">
          {/* Method Selector */}
          <div>
            <label className="block text-gray-200 font-medium mb-2">Detection Method</label>
            <select
              value={selectedMethod}
              onChange={(e) => setSelectedMethod(e.target.value)}
              className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-orange-500"
            >
              {methods.map(m => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </select>
            <p className="text-sm text-gray-400 mt-1">{currentMethod?.description}</p>
          </div>

          {/* Parameters */}
          {currentMethod?.params.includes('threshold') && (
            <div>
              <label className="block text-gray-200 font-medium mb-2">
                Z-Score Threshold (standard deviations)
              </label>
              <input
                type="number"
                step="0.1"
                min="1"
                max="5"
                value={detectionParams.threshold}
                onChange={(e) => handleParamChange('threshold', e.target.value)}
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-orange-500"
              />
              <p className="text-sm text-gray-400 mt-1">
                Typical values: 2 (loose), 3 (moderate), 4 (strict)
              </p>
            </div>
          )}

          {currentMethod?.params.includes('contamination') && (
            <div>
              <label className="block text-gray-200 font-medium mb-2">
                Expected Contamination (0.0 - 0.5)
              </label>
              <input
                type="number"
                step="0.01"
                min="0.01"
                max="0.5"
                value={detectionParams.contamination}
                onChange={(e) => handleParamChange('contamination', e.target.value)}
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-orange-500"
              />
              <p className="text-sm text-gray-400 mt-1">
                Proportion of expected outliers (e.g., 0.1 = 10%)
              </p>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="bg-red-900/20 border border-red-700/50 rounded-lg p-4">
              <p className="text-red-200">{error}</p>
            </div>
          )}

          {/* Loading State */}
          {loading && (
            <div className="flex items-center justify-center p-4">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-orange-400"></div>
            </div>
          )}
        </div>
      </div>

      {/* Results */}
      {outlierResults && !loading && (
        <>
          {/* Statistics */}
          <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
            <h3 className="text-xl font-bold text-gray-100 mb-4">Detection Results</h3>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-slate-700/30 rounded-lg p-4">
                <div className="text-sm text-gray-400">Total Points</div>
                <div className="text-2xl font-bold text-gray-100">
                  {outlierResults.statistics.n_total}
                </div>
              </div>

              <div className="bg-red-900/20 border border-red-700/30 rounded-lg p-4">
                <div className="text-sm text-gray-400">Outliers Detected</div>
                <div className="text-2xl font-bold text-red-400">
                  {outlierResults.statistics.n_outliers}
                </div>
                <div className="text-xs text-gray-400 mt-1">
                  {outlierResults.statistics.outlier_percentage.toFixed(1)}%
                </div>
              </div>

              <div className="bg-green-900/20 border border-green-700/30 rounded-lg p-4">
                <div className="text-sm text-gray-400">Inliers</div>
                <div className="text-2xl font-bold text-green-400">
                  {outlierResults.statistics.n_inliers}
                </div>
              </div>

              <div className="bg-slate-700/30 rounded-lg p-4">
                <div className="text-sm text-gray-400">Selected to Remove</div>
                <div className="text-2xl font-bold text-orange-400">
                  {manualSelection.size}
                </div>
              </div>
            </div>

            {/* Method Info */}
            {outlierResults.method_info && (
              <div className="mt-4 p-3 bg-slate-700/30 rounded-lg">
                <div className="text-sm text-gray-300">
                  <strong>Method:</strong> {currentMethod?.label}
                </div>
                {Object.entries(outlierResults.method_info)
                  .filter(([key]) => key !== 'method')
                  .map(([key, value]) => (
                    <div key={key} className="text-sm text-gray-300">
                      <strong>{key.replace(/_/g, ' ')}:</strong>{' '}
                      {typeof value === 'number' ? value.toFixed(4) : String(value)}
                    </div>
                  ))}
              </div>
            )}
          </div>

          {/* Visualization */}
          <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold text-gray-100">Visualization</h3>
              <button
                onClick={() => setShowOutliers(!showOutliers)}
                className="flex items-center gap-2 px-3 py-1 bg-slate-700 hover:bg-slate-600 text-gray-200 rounded-lg transition"
              >
                {showOutliers ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
                {showOutliers ? 'Hide' : 'Show'} Outliers
              </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Scatter Plot */}
              <div>
                <h4 className="text-lg font-semibold text-gray-200 mb-3">Index Plot</h4>
                <Plot
                  data={createScatterPlot()}
                  layout={{
                    ...getPlotlyLayout(''),
                    xaxis: {
                      title: 'Index',
                      color: '#cbd5e1',
                      gridcolor: 'rgba(51, 65, 85, 0.5)',
                      zerolinecolor: 'rgba(71, 85, 105, 0.7)'
                    },
                    yaxis: {
                      title: columnName,
                      color: '#cbd5e1',
                      gridcolor: 'rgba(51, 65, 85, 0.5)',
                      zerolinecolor: 'rgba(71, 85, 105, 0.7)'
                    },
                    showlegend: true,
                    legend: { x: 0.01, y: 0.99, bgcolor: 'rgba(15, 23, 42, 0.8)', font: { color: '#e2e8f0' } },
                    height: 350,
                    margin: { l: 50, r: 20, t: 20, b: 50 }
                  }}
                  config={{
                    ...getPlotlyConfig(),
                    modeBarButtonsToAdd: [],
                    displayModeBar: false,
                    renderer: 'svg'
                  }}
                  style={{ width: '100%' }}
                  onClick={(data) => {
                    if (data.points && data.points[0]) {
                      const pointIndex = data.points[0].x
                      togglePoint(pointIndex)
                    }
                  }}
                />
                <p className="text-xs text-gray-400 mt-2">
                  Click on points to toggle outlier selection
                </p>
              </div>

              {/* Box Plot */}
              <div>
                <h4 className="text-lg font-semibold text-gray-200 mb-3">Box Plot</h4>
                <Plot
                  data={createBoxPlot()}
                  layout={{
                    ...getPlotlyLayout(''),
                    yaxis: {
                      title: columnName,
                      color: '#cbd5e1',
                      gridcolor: 'rgba(51, 65, 85, 0.5)',
                      zerolinecolor: 'rgba(71, 85, 105, 0.7)'
                    },
                    showlegend: true,
                    legend: { x: 0.01, y: 0.99, bgcolor: 'rgba(15, 23, 42, 0.8)', font: { color: '#e2e8f0' } },
                    height: 350,
                    margin: { l: 50, r: 20, t: 20, b: 50 }
                  }}
                  config={{
                    ...getPlotlyConfig(),
                    displayModeBar: false,
                    renderer: 'svg'
                  }}
                  style={{ width: '100%' }}
                />
              </div>
            </div>
          </div>

          {/* Manual Selection Controls */}
          <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
            <h3 className="text-xl font-bold text-gray-100 mb-4">Outlier Selection</h3>

            <div className="flex gap-3 mb-4">
              <button
                onClick={selectAll}
                className="flex items-center gap-2 px-4 py-2 bg-orange-600 hover:bg-orange-700 text-white rounded-lg transition"
              >
                <Check className="w-4 h-4" />
                Select All Detected
              </button>
              <button
                onClick={clearAll}
                className="flex items-center gap-2 px-4 py-2 bg-slate-600 hover:bg-slate-700 text-white rounded-lg transition"
              >
                <X className="w-4 h-4" />
                Clear Selection
              </button>
            </div>

            <div className="bg-amber-900/20 border border-amber-700/50 rounded-lg p-4 mb-4">
              <p className="text-amber-200 text-sm">
                <strong>Note:</strong> {manualSelection.size} point(s) selected for removal.
                Click "Apply" to remove them from the dataset, or "Cancel" to keep all data.
              </p>
            </div>

            <div className="flex gap-3">
              <button
                onClick={handleApply}
                disabled={manualSelection.size === 0}
                className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-slate-600 disabled:cursor-not-allowed text-white rounded-lg transition"
              >
                <Trash2 className="w-4 h-4" />
                Remove Selected ({manualSelection.size})
              </button>
              <button
                onClick={onCancel}
                className="flex items-center gap-2 px-4 py-2 bg-slate-600 hover:bg-slate-700 text-white rounded-lg transition"
              >
                Cancel
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

export default OutlierDetection
