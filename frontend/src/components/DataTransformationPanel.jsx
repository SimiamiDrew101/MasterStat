import { useState, useEffect } from 'react'
import Plot from 'react-plotly.js'
import { RefreshCw, Undo2, Check, Info, Zap } from 'lucide-react'
import { getPlotlyConfig } from '../utils/plotlyConfig'
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const DataTransformationPanel = ({
  data,
  columnName = 'Response',
  onApply,
  onReset
}) => {
  const [selectedTransform, setSelectedTransform] = useState('none')
  const [transformedData, setTransformedData] = useState(null)
  const [originalData, setOriginalData] = useState([])
  const [transformHistory, setTransformHistory] = useState([])
  const [centeringMethod, setCenteringMethod] = useState('none')
  const [scalingMethod, setScalingMethod] = useState('none')
  const [customCenter, setCustomCenter] = useState('')
  const [customScale, setCustomScale] = useState('')
  const [useBackend, setUseBackend] = useState(false)
  const [backendLoading, setBackendLoading] = useState(false)
  const [backendInfo, setBackendInfo] = useState(null)

  // Transformation options
  const transforms = [
    { value: 'none', label: 'None', description: 'Original data' },
    { value: 'log', label: 'Log', description: 'Natural logarithm (data must be > 0)' },
    { value: 'log10', label: 'Log10', description: 'Base-10 logarithm (data must be > 0)' },
    { value: 'sqrt', label: 'Square Root', description: 'Square root (data must be ≥ 0)' },
    { value: 'boxcox', label: 'Box-Cox', description: 'Optimal power transformation (data must be > 0)' },
    { value: 'zscore', label: 'Z-Score', description: 'Standardize to mean=0, std=1' },
    { value: 'minmax', label: 'Min-Max', description: 'Scale to range [0, 1]' },
    { value: 'rank', label: 'Rank', description: 'Convert to ranks (1, 2, 3, ...)' }
  ]

  const centeringOptions = [
    { value: 'none', label: 'None' },
    { value: 'mean', label: 'Mean' },
    { value: 'median', label: 'Median' },
    { value: 'custom', label: 'Custom' }
  ]

  const scalingOptions = [
    { value: 'none', label: 'None' },
    { value: 'std', label: 'Standard Deviation' },
    { value: 'range', label: 'Range [0, 1]' },
    { value: 'custom', label: 'Custom' }
  ]

  // Initialize original data
  useEffect(() => {
    if (data && data.length > 0) {
      setOriginalData(data)
    }
  }, [data])

  // Apply transformation when selection changes
  useEffect(() => {
    if (originalData.length > 0) {
      if (useBackend && selectedTransform !== 'none') {
        applyBackendTransformation()
      } else {
        applyTransformation()
      }
    }
  }, [selectedTransform, centeringMethod, scalingMethod, customCenter, customScale, originalData, useBackend])

  // Calculate statistics
  const calculateStats = (values) => {
    if (!values || values.length === 0) return null

    const sorted = [...values].sort((a, b) => a - b)
    const n = values.length
    const mean = values.reduce((sum, val) => sum + val, 0) / n
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (n - 1)
    const std = Math.sqrt(variance)
    const min = sorted[0]
    const max = sorted[n - 1]
    const q1 = sorted[Math.floor(n * 0.25)]
    const median = n % 2 === 0
      ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2
      : sorted[Math.floor(n / 2)]
    const q3 = sorted[Math.floor(n * 0.75)]

    return { mean, std, variance, min, max, q1, median, q3, n }
  }

  // Call backend API for transformation
  const applyBackendTransformation = async () => {
    if (!originalData || originalData.length === 0) return

    setBackendLoading(true)
    setBackendInfo(null)

    try {
      const response = await axios.post(`${API_URL}/api/preprocessing/transform`, {
        data: originalData,
        transform_type: selectedTransform,
        centering: centeringMethod,
        scaling: scalingMethod,
        custom_center: customCenter ? parseFloat(customCenter) : null,
        custom_scale: customScale ? parseFloat(customScale) : null
      })

      setTransformedData({
        values: response.data.transformed_data,
        isValid: true,
        errorMessage: '',
        transform: selectedTransform,
        centering: centeringMethod,
        scaling: scalingMethod
      })
      setBackendInfo(response.data.parameters)
    } catch (error) {
      setTransformedData({
        values: originalData,
        isValid: false,
        errorMessage: error.response?.data?.detail || 'Backend transformation failed',
        transform: selectedTransform,
        centering: centeringMethod,
        scaling: scalingMethod
      })
      setBackendInfo(null)
    } finally {
      setBackendLoading(false)
    }
  }

  // Apply transformation
  const applyTransformation = () => {
    if (!originalData || originalData.length === 0) return

    let values = [...originalData]
    let transformed = [...values]
    let isValid = true
    let errorMessage = ''

    try {
      // Apply main transformation
      switch (selectedTransform) {
        case 'log':
          if (values.some(v => v <= 0)) {
            errorMessage = 'Log transformation requires all values > 0'
            isValid = false
          } else {
            transformed = values.map(v => Math.log(v))
          }
          break

        case 'log10':
          if (values.some(v => v <= 0)) {
            errorMessage = 'Log10 transformation requires all values > 0'
            isValid = false
          } else {
            transformed = values.map(v => Math.log10(v))
          }
          break

        case 'sqrt':
          if (values.some(v => v < 0)) {
            errorMessage = 'Square root transformation requires all values ≥ 0'
            isValid = false
          } else {
            transformed = values.map(v => Math.sqrt(v))
          }
          break

        case 'boxcox':
          if (values.some(v => v <= 0)) {
            errorMessage = 'Box-Cox transformation requires all values > 0'
            isValid = false
          } else {
            // Simplified Box-Cox with λ ≈ 0 (log transform)
            // In practice, this should calculate optimal λ
            transformed = values.map(v => Math.log(v))
          }
          break

        case 'zscore':
          const mean = values.reduce((sum, v) => sum + v, 0) / values.length
          const variance = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length
          const std = Math.sqrt(variance)
          transformed = values.map(v => (v - mean) / std)
          break

        case 'minmax':
          const min = Math.min(...values)
          const max = Math.max(...values)
          const range = max - min
          transformed = values.map(v => range === 0 ? 0 : (v - min) / range)
          break

        case 'rank':
          const sorted = [...values].map((v, i) => ({ value: v, index: i }))
            .sort((a, b) => a.value - b.value)
          transformed = new Array(values.length)
          sorted.forEach((item, rank) => {
            transformed[item.index] = rank + 1
          })
          break

        case 'none':
        default:
          transformed = [...values]
          break
      }

      // Apply centering
      if (isValid && centeringMethod !== 'none') {
        let centerValue = 0
        switch (centeringMethod) {
          case 'mean':
            centerValue = transformed.reduce((sum, v) => sum + v, 0) / transformed.length
            break
          case 'median':
            const sorted = [...transformed].sort((a, b) => a - b)
            centerValue = sorted.length % 2 === 0
              ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
              : sorted[Math.floor(sorted.length / 2)]
            break
          case 'custom':
            centerValue = parseFloat(customCenter) || 0
            break
        }
        transformed = transformed.map(v => v - centerValue)
      }

      // Apply scaling
      if (isValid && scalingMethod !== 'none') {
        let scaleValue = 1
        switch (scalingMethod) {
          case 'std':
            const mean = transformed.reduce((sum, v) => sum + v, 0) / transformed.length
            const variance = transformed.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / transformed.length
            scaleValue = Math.sqrt(variance)
            break
          case 'range':
            const min = Math.min(...transformed)
            const max = Math.max(...transformed)
            scaleValue = max - min
            break
          case 'custom':
            scaleValue = parseFloat(customScale) || 1
            break
        }
        if (scaleValue !== 0) {
          transformed = transformed.map(v => v / scaleValue)
        }
      }

      setTransformedData({
        values: transformed,
        isValid,
        errorMessage,
        transform: selectedTransform,
        centering: centeringMethod,
        scaling: scalingMethod
      })
    } catch (error) {
      setTransformedData({
        values: values,
        isValid: false,
        errorMessage: `Transformation error: ${error.message}`,
        transform: selectedTransform,
        centering: centeringMethod,
        scaling: scalingMethod
      })
    }
  }

  // Handle apply
  const handleApply = () => {
    if (transformedData && transformedData.isValid && onApply) {
      setTransformHistory([...transformHistory, {
        transform: selectedTransform,
        centering: centeringMethod,
        scaling: scalingMethod,
        timestamp: new Date()
      }])
      onApply(transformedData.values, {
        transform: selectedTransform,
        centering: centeringMethod,
        scaling: scalingMethod
      })
    }
  }

  // Handle undo
  const handleUndo = () => {
    if (transformHistory.length > 0) {
      const newHistory = [...transformHistory]
      newHistory.pop()
      setTransformHistory(newHistory)

      if (newHistory.length > 0) {
        const lastTransform = newHistory[newHistory.length - 1]
        setSelectedTransform(lastTransform.transform)
        setCenteringMethod(lastTransform.centering)
        setScalingMethod(lastTransform.scaling)
      } else {
        setSelectedTransform('none')
        setCenteringMethod('none')
        setScalingMethod('none')
      }
    }
  }

  // Handle reset
  const handleReset = () => {
    setSelectedTransform('none')
    setCenteringMethod('none')
    setScalingMethod('none')
    setCustomCenter('')
    setCustomScale('')
    setTransformHistory([])
    if (onReset) {
      onReset()
    }
  }

  const originalStats = calculateStats(originalData)
  const transformedStats = transformedData?.isValid ? calculateStats(transformedData.values) : null

  // Create histogram data
  const createHistogramTrace = (values, name, color) => {
    return {
      x: values,
      type: 'histogram',
      name: name,
      opacity: 0.7,
      marker: { color: color },
      nbinsx: Math.min(30, Math.ceil(Math.sqrt(values.length)))
    }
  }

  return (
    <div className="space-y-6">
      {/* Transformation Selector */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <div className="flex items-center gap-2 mb-4">
          <RefreshCw className="w-5 h-5 text-indigo-400" />
          <h3 className="text-xl font-bold text-gray-100">Data Transformation</h3>
        </div>

        <div className="space-y-4">
          {/* Main Transform */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="block text-gray-200 font-medium">Transform Type</label>
              <button
                onClick={() => setUseBackend(!useBackend)}
                className={`flex items-center gap-1 px-3 py-1 rounded-lg text-xs transition ${
                  useBackend
                    ? 'bg-indigo-600 hover:bg-indigo-700 text-white'
                    : 'bg-slate-700 hover:bg-slate-600 text-gray-300'
                }`}
                title={useBackend ? 'Using backend API (optimal calculations)' : 'Using client-side (instant preview)'}
              >
                <Zap className="w-3 h-3" />
                {useBackend ? 'Backend (Optimal)' : 'Client (Fast)'}
              </button>
            </div>
            <select
              value={selectedTransform}
              onChange={(e) => setSelectedTransform(e.target.value)}
              className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              {transforms.map(t => (
                <option key={t.value} value={t.value}>{t.label}</option>
              ))}
            </select>
            <p className="text-sm text-gray-400 mt-1">
              {transforms.find(t => t.value === selectedTransform)?.description}
            </p>
          </div>

          {/* Centering Options */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-gray-200 font-medium mb-2">Centering</label>
              <select
                value={centeringMethod}
                onChange={(e) => setCenteringMethod(e.target.value)}
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                {centeringOptions.map(opt => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </select>
              {centeringMethod === 'custom' && (
                <input
                  type="number"
                  step="any"
                  value={customCenter}
                  onChange={(e) => setCustomCenter(e.target.value)}
                  placeholder="Enter center value"
                  className="mt-2 w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              )}
            </div>

            <div>
              <label className="block text-gray-200 font-medium mb-2">Scaling</label>
              <select
                value={scalingMethod}
                onChange={(e) => setScalingMethod(e.target.value)}
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                {scalingOptions.map(opt => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </select>
              {scalingMethod === 'custom' && (
                <input
                  type="number"
                  step="any"
                  value={customScale}
                  onChange={(e) => setCustomScale(e.target.value)}
                  placeholder="Enter scale value"
                  className="mt-2 w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              )}
            </div>
          </div>

          {/* Loading State */}
          {backendLoading && (
            <div className="bg-indigo-900/20 border border-indigo-700/50 rounded-lg p-4">
              <div className="flex items-center gap-3">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-indigo-400"></div>
                <p className="text-indigo-200">Processing transformation on backend...</p>
              </div>
            </div>
          )}

          {/* Backend Info */}
          {useBackend && backendInfo && !backendLoading && (
            <div className="bg-indigo-900/20 border border-indigo-700/50 rounded-lg p-4">
              <div className="flex items-start gap-2">
                <Zap className="w-5 h-5 text-indigo-400 mt-0.5" />
                <div className="flex-1">
                  <p className="text-indigo-200 font-semibold mb-2">Backend Transformation Applied</p>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    {backendInfo.boxcox_lambda !== undefined && (
                      <div className="col-span-2">
                        <span className="text-gray-400">Box-Cox λ (optimal):</span>{' '}
                        <span className="text-indigo-300 font-mono">{backendInfo.boxcox_lambda.toFixed(4)}</span>
                      </div>
                    )}
                    {backendInfo.center_value !== undefined && backendInfo.center_value !== null && (
                      <div>
                        <span className="text-gray-400">Center:</span>{' '}
                        <span className="text-indigo-300 font-mono">{backendInfo.center_value.toFixed(4)}</span>
                      </div>
                    )}
                    {backendInfo.scale_value !== undefined && backendInfo.scale_value !== null && (
                      <div>
                        <span className="text-gray-400">Scale:</span>{' '}
                        <span className="text-indigo-300 font-mono">{backendInfo.scale_value.toFixed(4)}</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Error Message */}
          {transformedData && !transformedData.isValid && (
            <div className="bg-red-900/20 border border-red-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2">
                <Info className="w-5 h-5 text-red-400" />
                <p className="text-red-200">{transformedData.errorMessage}</p>
              </div>
            </div>
          )}

          {/* Control Buttons */}
          <div className="flex gap-3">
            <button
              onClick={handleApply}
              disabled={!transformedData?.isValid || selectedTransform === 'none'}
              className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-600 disabled:cursor-not-allowed text-white rounded-lg transition"
            >
              <Check className="w-4 h-4" />
              Apply Transform
            </button>
            <button
              onClick={handleUndo}
              disabled={transformHistory.length === 0}
              className="flex items-center gap-2 px-4 py-2 bg-slate-600 hover:bg-slate-700 disabled:bg-slate-700 disabled:cursor-not-allowed text-white rounded-lg transition"
            >
              <Undo2 className="w-4 h-4" />
              Undo
            </button>
            <button
              onClick={handleReset}
              className="flex items-center gap-2 px-4 py-2 bg-slate-600 hover:bg-slate-700 text-white rounded-lg transition"
            >
              <RefreshCw className="w-4 h-4" />
              Reset
            </button>
          </div>
        </div>
      </div>

      {/* Before/After Comparison */}
      {originalData.length > 0 && (
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
          <h3 className="text-xl font-bold text-gray-100 mb-4">Distribution Comparison</h3>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Original Distribution */}
            <div>
              <h4 className="text-lg font-semibold text-gray-200 mb-3">Original Data</h4>
              {originalStats && (
                <div className="mb-4 p-3 bg-slate-700/30 rounded-lg">
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div><span className="text-gray-400">Mean:</span> <span className="text-gray-100">{originalStats.mean.toFixed(3)}</span></div>
                    <div><span className="text-gray-400">Std:</span> <span className="text-gray-100">{originalStats.std.toFixed(3)}</span></div>
                    <div><span className="text-gray-400">Min:</span> <span className="text-gray-100">{originalStats.min.toFixed(3)}</span></div>
                    <div><span className="text-gray-400">Max:</span> <span className="text-gray-100">{originalStats.max.toFixed(3)}</span></div>
                    <div><span className="text-gray-400">Median:</span> <span className="text-gray-100">{originalStats.median.toFixed(3)}</span></div>
                    <div><span className="text-gray-400">n:</span> <span className="text-gray-100">{originalStats.n}</span></div>
                  </div>
                </div>
              )}
              <Plot
                data={[createHistogramTrace(originalData, 'Original', '#818cf8')]}
                layout={{
                  ...getPlotlyConfig().layout,
                  xaxis: { title: columnName, color: '#cbd5e1' },
                  yaxis: { title: 'Frequency', color: '#cbd5e1' },
                  showlegend: false,
                  height: 300,
                  margin: { l: 50, r: 20, t: 20, b: 50 }
                }}
                config={{
                  ...getPlotlyConfig().config,
                  displayModeBar: false,
                  renderer: 'svg'
                }}
                style={{ width: '100%' }}
              />
            </div>

            {/* Transformed Distribution */}
            <div>
              <h4 className="text-lg font-semibold text-gray-200 mb-3">
                Transformed Data
                {selectedTransform !== 'none' && (
                  <span className="ml-2 text-sm text-indigo-400">
                    ({transforms.find(t => t.value === selectedTransform)?.label})
                  </span>
                )}
              </h4>
              {transformedData?.isValid && transformedStats ? (
                <>
                  <div className="mb-4 p-3 bg-slate-700/30 rounded-lg">
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div><span className="text-gray-400">Mean:</span> <span className="text-gray-100">{transformedStats.mean.toFixed(3)}</span></div>
                      <div><span className="text-gray-400">Std:</span> <span className="text-gray-100">{transformedStats.std.toFixed(3)}</span></div>
                      <div><span className="text-gray-400">Min:</span> <span className="text-gray-100">{transformedStats.min.toFixed(3)}</span></div>
                      <div><span className="text-gray-400">Max:</span> <span className="text-gray-100">{transformedStats.max.toFixed(3)}</span></div>
                      <div><span className="text-gray-400">Median:</span> <span className="text-gray-100">{transformedStats.median.toFixed(3)}</span></div>
                      <div><span className="text-gray-400">n:</span> <span className="text-gray-100">{transformedStats.n}</span></div>
                    </div>
                  </div>
                  <Plot
                    data={[createHistogramTrace(transformedData.values, 'Transformed', '#22d3ee')]}
                    layout={{
                      ...getPlotlyConfig().layout,
                      xaxis: { title: `${columnName} (Transformed)`, color: '#cbd5e1' },
                      yaxis: { title: 'Frequency', color: '#cbd5e1' },
                      showlegend: false,
                      height: 300,
                      margin: { l: 50, r: 20, t: 20, b: 50 }
                    }}
                    config={{
                      ...getPlotlyConfig().config,
                      displayModeBar: false,
                      renderer: 'svg'
                    }}
                    style={{ width: '100%' }}
                  />
                </>
              ) : (
                <div className="flex items-center justify-center h-full bg-slate-700/20 rounded-lg p-8">
                  <p className="text-gray-400">
                    {transformedData?.errorMessage || 'Select a transformation to preview'}
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Transform History */}
      {transformHistory.length > 0 && (
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
          <h3 className="text-xl font-bold text-gray-100 mb-4">Transform History</h3>
          <div className="space-y-2">
            {transformHistory.map((item, idx) => (
              <div key={idx} className="p-3 bg-slate-700/30 rounded-lg text-sm text-gray-300">
                <span className="font-semibold text-indigo-400">#{idx + 1}</span>
                {' '}Transform: {transforms.find(t => t.value === item.transform)?.label}
                {item.centering !== 'none' && ` | Center: ${item.centering}`}
                {item.scaling !== 'none' && ` | Scale: ${item.scaling}`}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default DataTransformationPanel
