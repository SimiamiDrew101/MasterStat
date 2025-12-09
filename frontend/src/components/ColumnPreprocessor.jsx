import { useState } from 'react'
import { Wand2, AlertTriangle, ChevronDown, ChevronUp } from 'lucide-react'
import DataTransformationPanel from './DataTransformationPanel'
import OutlierDetection from './OutlierDetection'

const ColumnPreprocessor = ({
  columns,
  onDataUpdate,
  className = ''
}) => {
  const [isExpanded, setIsExpanded] = useState(false)
  const [activeMode, setActiveMode] = useState(null) // 'transform' or 'outlier'
  const [selectedColumn, setSelectedColumn] = useState(0)
  const [processedData, setProcessedData] = useState({})

  // Get current column data
  const getCurrentColumnData = () => {
    const column = columns[selectedColumn]
    if (!column || !column.data) return []

    // Filter out empty values
    return column.data.filter(v => v !== null && v !== undefined && v !== '')
  }

  // Handle transformation apply
  const handleTransformApply = (transformedValues, transformInfo) => {
    const column = columns[selectedColumn]

    // Update processed data state
    setProcessedData({
      ...processedData,
      [selectedColumn]: {
        original: column.data,
        transformed: transformedValues,
        info: transformInfo
      }
    })

    // Notify parent component
    if (onDataUpdate) {
      onDataUpdate(selectedColumn, transformedValues, transformInfo)
    }

    setActiveMode(null)
  }

  // Handle transformation reset
  const handleTransformReset = () => {
    if (processedData[selectedColumn]) {
      const original = processedData[selectedColumn].original

      if (onDataUpdate) {
        onDataUpdate(selectedColumn, original, { transform: 'none' })
      }

      const newProcessedData = { ...processedData }
      delete newProcessedData[selectedColumn]
      setProcessedData(newProcessedData)
    }

    setActiveMode(null)
  }

  // Handle outlier removal
  const handleOutlierApply = (cleanedValues, outlierInfo) => {
    const column = columns[selectedColumn]

    // Update processed data state
    setProcessedData({
      ...processedData,
      [selectedColumn]: {
        original: column.data,
        transformed: cleanedValues,
        info: { ...outlierInfo, type: 'outlier_removal' }
      }
    })

    // Notify parent component
    if (onDataUpdate) {
      onDataUpdate(selectedColumn, cleanedValues, outlierInfo)
    }

    setActiveMode(null)
  }

  const columnData = getCurrentColumnData()
  const hasData = columnData.length > 0

  return (
    <div className={`bg-slate-800/30 backdrop-blur-lg rounded-2xl border border-slate-700/50 ${className}`}>
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-4 hover:bg-slate-700/30 transition rounded-t-2xl"
      >
        <div className="flex items-center gap-2">
          <Wand2 className="w-5 h-5 text-purple-400" />
          <h3 className="text-lg font-bold text-gray-100">Data Preprocessing</h3>
          {Object.keys(processedData).length > 0 && (
            <span className="px-2 py-0.5 bg-purple-600/30 text-purple-300 text-xs rounded-full">
              {Object.keys(processedData).length} column(s) preprocessed
            </span>
          )}
        </div>
        {isExpanded ? (
          <ChevronUp className="w-5 h-5 text-gray-400" />
        ) : (
          <ChevronDown className="w-5 h-5 text-gray-400" />
        )}
      </button>

      {/* Content */}
      {isExpanded && (
        <div className="p-4 pt-0 space-y-4">
          {/* Column Selector */}
          <div>
            <label className="block text-gray-200 font-medium mb-2">Select Column to Preprocess</label>
            <select
              value={selectedColumn}
              onChange={(e) => {
                setSelectedColumn(parseInt(e.target.value))
                setActiveMode(null)
              }}
              className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-purple-500"
            >
              {columns.map((column, idx) => (
                <option key={idx} value={idx}>
                  {column.name} ({column.data.filter(v => v !== null && v !== undefined && v !== '').length} values)
                  {processedData[idx] && ' - Preprocessed'}
                </option>
              ))}
            </select>
          </div>

          {/* Action Buttons */}
          {!activeMode && hasData && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <button
                onClick={() => setActiveMode('transform')}
                className="flex flex-col items-center justify-center p-4 bg-gradient-to-br from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 rounded-lg transition"
              >
                <Wand2 className="w-6 h-6 mb-2" />
                <span className="font-semibold">Transform Data</span>
                <span className="text-xs text-gray-200 mt-1">Log, Box-Cox, Z-Score, etc.</span>
              </button>

              <button
                onClick={() => setActiveMode('outlier')}
                className="flex flex-col items-center justify-center p-4 bg-gradient-to-br from-orange-600 to-red-600 hover:from-orange-700 hover:to-red-700 rounded-lg transition"
              >
                <AlertTriangle className="w-6 h-6 mb-2" />
                <span className="font-semibold">Detect Outliers</span>
                <span className="text-xs text-gray-200 mt-1">Z-Score, IQR, ML methods</span>
              </button>
            </div>
          )}

          {/* No Data Message */}
          {!hasData && (
            <div className="bg-amber-900/20 border border-amber-700/50 rounded-lg p-4">
              <p className="text-amber-200 text-sm">
                Please enter data in the selected column first before preprocessing.
              </p>
            </div>
          )}

          {/* Transformation Panel */}
          {activeMode === 'transform' && hasData && (
            <div>
              <div className="flex items-center justify-between mb-3">
                <h4 className="text-lg font-semibold text-gray-100">
                  Transform: {columns[selectedColumn].name}
                </h4>
                <button
                  onClick={() => setActiveMode(null)}
                  className="px-3 py-1 bg-slate-700 hover:bg-slate-600 text-gray-100 rounded-lg transition text-sm"
                >
                  Cancel
                </button>
              </div>
              <DataTransformationPanel
                data={columnData}
                columnName={columns[selectedColumn].name}
                onApply={handleTransformApply}
                onReset={handleTransformReset}
              />
            </div>
          )}

          {/* Outlier Detection Panel */}
          {activeMode === 'outlier' && hasData && (
            <div>
              <div className="flex items-center justify-between mb-3">
                <h4 className="text-lg font-semibold text-gray-100">
                  Outlier Detection: {columns[selectedColumn].name}
                </h4>
                <button
                  onClick={() => setActiveMode(null)}
                  className="px-3 py-1 bg-slate-700 hover:bg-slate-600 text-gray-100 rounded-lg transition text-sm"
                >
                  Cancel
                </button>
              </div>
              <OutlierDetection
                data={columnData}
                columnName={columns[selectedColumn].name}
                onApply={handleOutlierApply}
                onCancel={() => setActiveMode(null)}
              />
            </div>
          )}

          {/* Preprocessing Summary */}
          {processedData[selectedColumn] && !activeMode && (
            <div className="bg-green-900/20 border border-green-700/50 rounded-lg p-4">
              <div className="flex items-start gap-2">
                <Wand2 className="w-5 h-5 text-green-400 mt-0.5" />
                <div className="flex-1">
                  <p className="text-green-200 font-semibold mb-1">Column Preprocessed</p>
                  <p className="text-sm text-gray-300">
                    {processedData[selectedColumn].info.type === 'outlier_removal'
                      ? `${processedData[selectedColumn].info.removedCount} outlier(s) removed using ${processedData[selectedColumn].info.method}`
                      : `Applied ${processedData[selectedColumn].info.transform} transformation`}
                  </p>
                  <button
                    onClick={handleTransformReset}
                    className="mt-2 px-3 py-1 bg-slate-700 hover:bg-slate-600 text-gray-100 rounded text-xs transition"
                  >
                    Reset to Original
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default ColumnPreprocessor
