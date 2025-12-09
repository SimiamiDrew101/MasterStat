import { useState, useEffect } from 'react'
import { Database, AlertCircle, Info, Sparkles, Settings } from 'lucide-react'
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

/**
 * ImputationSelector - Component for selecting and configuring missing data imputation methods
 * Supports Mean, Median, KNN, MICE, Linear Interpolation, and LOCF methods
 */
const ImputationSelector = ({ data, columnName = 'Response', onApply, onCancel }) => {
  const [selectedMethod, setSelectedMethod] = useState('mean')
  const [parameters, setParameters] = useState({
    knn_neighbors: 5,
    mice_iterations: 10,
    mice_random_state: 42
  })
  const [imputedData, setImputedData] = useState(null)
  const [previewData, setPreviewData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [missingInfo, setMissingInfo] = useState(null)

  // Imputation method options
  const methods = [
    {
      value: 'mean',
      label: 'Mean Imputation',
      description: 'Replace missing values with the mean of observed values',
      icon: '📊',
      params: []
    },
    {
      value: 'median',
      label: 'Median Imputation',
      description: 'Replace missing values with the median of observed values',
      icon: '📈',
      params: []
    },
    {
      value: 'knn',
      label: 'KNN Imputation',
      description: 'Use K-Nearest Neighbors to estimate missing values',
      icon: '🎯',
      params: ['knn_neighbors']
    },
    {
      value: 'mice',
      label: 'MICE (Multiple Imputation)',
      description: 'Multivariate Imputation by Chained Equations',
      icon: '🔗',
      params: ['mice_iterations', 'mice_random_state']
    },
    {
      value: 'linear',
      label: 'Linear Interpolation',
      description: 'Interpolate missing values linearly between observed values',
      icon: '📉',
      params: []
    },
    {
      value: 'locf',
      label: 'LOCF (Last Observation Carried Forward)',
      description: 'Forward fill missing values with the last observed value',
      icon: '⏭️',
      params: []
    }
  ]

  // Analyze missing data pattern
  useEffect(() => {
    if (data && Array.isArray(data)) {
      const total = data.length
      const missing = data.filter(v => v === null || v === undefined || isNaN(v)).length
      const observed = total - missing
      const missingIndices = data.map((v, i) => (v === null || v === undefined || isNaN(v)) ? i : null).filter(i => i !== null)

      setMissingInfo({
        total,
        missing,
        observed,
        percentMissing: (missing / total) * 100,
        missingIndices
      })
    }
  }, [data])

  // Generate preview when method or parameters change
  useEffect(() => {
    if (data && missingInfo && missingInfo.missing > 0) {
      generatePreview()
    }
  }, [selectedMethod, parameters, data])

  // Generate preview of imputation
  const generatePreview = async () => {
    if (!data || !missingInfo || missingInfo.missing === 0) return

    setLoading(true)
    setError(null)

    try {
      const response = await axios.post(`${API_URL}/api/imputation/preview`, {
        data: data,
        method: selectedMethod,
        parameters: parameters
      })

      setPreviewData(response.data)
      setError(null)
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to generate preview')
      setPreviewData(null)
    } finally {
      setLoading(false)
    }
  }

  // Apply imputation
  const handleApply = async () => {
    if (!data || !missingInfo || missingInfo.missing === 0) return

    setLoading(true)
    setError(null)

    try {
      const response = await axios.post(`${API_URL}/api/imputation/impute`, {
        data: data,
        method: selectedMethod,
        parameters: parameters
      })

      setImputedData(response.data.imputed_data)

      if (onApply) {
        onApply(response.data.imputed_data, {
          method: selectedMethod,
          parameters: parameters,
          statistics: response.data.statistics
        })
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to apply imputation')
    } finally {
      setLoading(false)
    }
  }

  // Update parameter
  const updateParameter = (param, value) => {
    setParameters(prev => ({
      ...prev,
      [param]: value
    }))
  }

  const selectedMethodInfo = methods.find(m => m.value === selectedMethod)

  // If no missing data, don't show the component
  if (!missingInfo || missingInfo.missing === 0) {
    return (
      <div className="bg-green-900/20 border border-green-700/50 rounded-lg p-4">
        <div className="flex items-center gap-2">
          <Info className="w-5 h-5 text-green-400" />
          <p className="text-green-200">No missing data detected. All values are complete.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Missing Data Summary */}
      <div className="bg-orange-900/20 border border-orange-700/50 rounded-lg p-5">
        <div className="flex items-center gap-2 mb-3">
          <AlertCircle className="w-5 h-5 text-orange-400" />
          <h4 className="font-semibold text-gray-100">Missing Data Detected</h4>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-slate-800/50 rounded-lg p-3">
            <div className="text-xs text-gray-400 mb-1">Total Observations</div>
            <div className="text-2xl font-bold text-cyan-400">{missingInfo.total}</div>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-3">
            <div className="text-xs text-gray-400 mb-1">Observed Values</div>
            <div className="text-2xl font-bold text-green-400">{missingInfo.observed}</div>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-3">
            <div className="text-xs text-gray-400 mb-1">Missing Values</div>
            <div className="text-2xl font-bold text-orange-400">{missingInfo.missing}</div>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-3">
            <div className="text-xs text-gray-400 mb-1">% Missing</div>
            <div className="text-2xl font-bold text-red-400">{missingInfo.percentMissing.toFixed(1)}%</div>
          </div>
        </div>

        {missingInfo.percentMissing > 30 && (
          <div className="mt-4 bg-red-900/20 border border-red-700/50 rounded-lg p-3">
            <p className="text-sm text-red-200">
              <strong>Warning:</strong> High proportion of missing data ({missingInfo.percentMissing.toFixed(1)}%).
              Imputation results should be interpreted with caution.
            </p>
          </div>
        )}
      </div>

      {/* Method Selection */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <div className="flex items-center gap-2 mb-4">
          <Database className="w-5 h-5 text-indigo-400" />
          <h3 className="text-xl font-bold text-gray-100">Imputation Method</h3>
        </div>

        <div className="space-y-4">
          {/* Method Selector */}
          <div>
            <label className="block text-gray-200 font-medium mb-2">Select Method</label>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {methods.map(method => (
                <button
                  key={method.value}
                  onClick={() => setSelectedMethod(method.value)}
                  className={`p-4 rounded-lg border-2 transition text-left ${
                    selectedMethod === method.value
                      ? 'border-indigo-500 bg-indigo-900/30'
                      : 'border-slate-600 bg-slate-700/30 hover:border-slate-500'
                  }`}
                >
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-2xl">{method.icon}</span>
                    <span className="font-semibold text-gray-100">{method.label}</span>
                  </div>
                  <p className="text-xs text-gray-400">{method.description}</p>
                </button>
              ))}
            </div>
          </div>

          {/* Method Description */}
          {selectedMethodInfo && (
            <div className="bg-slate-700/30 rounded-lg p-4">
              <div className="flex items-start gap-2">
                <Info className="w-5 h-5 text-blue-400 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-sm font-semibold text-gray-200 mb-1">{selectedMethodInfo.label}</p>
                  <p className="text-sm text-gray-300">{selectedMethodInfo.description}</p>
                </div>
              </div>
            </div>
          )}

          {/* Method Parameters */}
          {selectedMethodInfo && selectedMethodInfo.params.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Settings className="w-4 h-4 text-gray-400" />
                <h4 className="font-medium text-gray-200">Method Parameters</h4>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {selectedMethodInfo.params.includes('knn_neighbors') && (
                  <div>
                    <label className="block text-sm text-gray-300 mb-2">
                      Number of Neighbors (k)
                    </label>
                    <input
                      type="number"
                      min="1"
                      max="20"
                      value={parameters.knn_neighbors}
                      onChange={(e) => updateParameter('knn_neighbors', parseInt(e.target.value))}
                      className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    />
                    <p className="text-xs text-gray-400 mt-1">Number of nearest neighbors to use for imputation</p>
                  </div>
                )}

                {selectedMethodInfo.params.includes('mice_iterations') && (
                  <div>
                    <label className="block text-sm text-gray-300 mb-2">
                      Iterations
                    </label>
                    <input
                      type="number"
                      min="1"
                      max="50"
                      value={parameters.mice_iterations}
                      onChange={(e) => updateParameter('mice_iterations', parseInt(e.target.value))}
                      className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    />
                    <p className="text-xs text-gray-400 mt-1">Number of MICE iterations</p>
                  </div>
                )}

                {selectedMethodInfo.params.includes('mice_random_state') && (
                  <div>
                    <label className="block text-sm text-gray-300 mb-2">
                      Random Seed
                    </label>
                    <input
                      type="number"
                      value={parameters.mice_random_state}
                      onChange={(e) => updateParameter('mice_random_state', parseInt(e.target.value))}
                      className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    />
                    <p className="text-xs text-gray-400 mt-1">Seed for reproducibility</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Preview Section */}
          {previewData && (
            <div className="bg-indigo-900/20 border border-indigo-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-3">
                <Sparkles className="w-5 h-5 text-indigo-400" />
                <h4 className="font-semibold text-indigo-200">Imputation Preview</h4>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
                {previewData.sample_imputed_values && previewData.sample_imputed_values.length > 0 && (
                  <div className="col-span-full">
                    <span className="text-gray-400">Sample Imputed Values:</span>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {previewData.sample_imputed_values.slice(0, 5).map((val, idx) => (
                        <span key={idx} className="px-2 py-1 bg-indigo-900/30 border border-indigo-700/50 rounded text-indigo-300 font-mono">
                          {val.toFixed(3)}
                        </span>
                      ))}
                      {previewData.sample_imputed_values.length > 5 && (
                        <span className="px-2 py-1 text-gray-400">
                          +{previewData.sample_imputed_values.length - 5} more
                        </span>
                      )}
                    </div>
                  </div>
                )}
                {previewData.statistics && (
                  <>
                    <div>
                      <span className="text-gray-400">Mean (imputed):</span>{' '}
                      <span className="text-indigo-300 font-mono">{previewData.statistics.mean?.toFixed(3)}</span>
                    </div>
                    <div>
                      <span className="text-gray-400">Std (imputed):</span>{' '}
                      <span className="text-indigo-300 font-mono">{previewData.statistics.std?.toFixed(3)}</span>
                    </div>
                    <div>
                      <span className="text-gray-400">Values Imputed:</span>{' '}
                      <span className="text-indigo-300 font-mono">{missingInfo.missing}</span>
                    </div>
                  </>
                )}
              </div>
            </div>
          )}

          {/* Loading State */}
          {loading && (
            <div className="bg-slate-700/30 rounded-lg p-4">
              <div className="flex items-center gap-3">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-indigo-400"></div>
                <p className="text-gray-300">Processing imputation...</p>
              </div>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="bg-red-900/20 border border-red-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2">
                <AlertCircle className="w-5 h-5 text-red-400" />
                <p className="text-red-200">{error}</p>
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex gap-3 pt-2">
            <button
              onClick={handleApply}
              disabled={loading || !previewData}
              className="flex items-center gap-2 px-6 py-2.5 bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-600 disabled:cursor-not-allowed text-white rounded-lg transition font-medium"
            >
              <Database className="w-4 h-4" />
              Apply Imputation
            </button>
            {onCancel && (
              <button
                onClick={onCancel}
                className="px-6 py-2.5 bg-slate-600 hover:bg-slate-700 text-white rounded-lg transition"
              >
                Cancel
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default ImputationSelector
