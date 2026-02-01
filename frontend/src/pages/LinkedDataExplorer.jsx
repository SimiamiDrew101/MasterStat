import { useState, useEffect, useCallback, useMemo } from 'react'
import Plot from 'react-plotly.js'
import {
  Link2,
  Upload,
  Plus,
  Trash2,
  MousePointer2,
  Square,
  Maximize2,
  ZoomIn,
  RotateCcw,
  Filter,
  Table,
  BarChart3,
  ScatterChart,
  Info,
  Download,
  Settings
} from 'lucide-react'
import { SelectionProvider, useSelection } from '../contexts/SelectionContext'

const API_URL = import.meta.env.VITE_API_URL || ''

// Inner component that uses selection context
const LinkedExplorerContent = () => {
  // Data state
  const [data, setData] = useState([])
  const [columns, setColumns] = useState([])
  const [numericColumns, setNumericColumns] = useState([])
  const [categoricalColumns, setCategoricalColumns] = useState([])

  // Plot configuration
  const [xColumn, setXColumn] = useState('')
  const [yColumn, setYColumn] = useState('')
  const [colorColumn, setColorColumn] = useState('')
  const [histColumn, setHistColumn] = useState('')

  // UI state
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [showTable, setShowTable] = useState(true)
  const [tablePageSize, setTablePageSize] = useState(20)
  const [tablePage, setTablePage] = useState(0)

  // Selection context
  const {
    selectedIndices,
    highlightedIndex,
    setHighlightedIndex,
    selectionColor,
    setLinkedData,
    setLinkedColumns,
    clearSelection,
    toggleIndex,
    selectByBrush,
    isSelected,
    hasSelection,
    selectionCount,
    selectionStats
  } = useSelection()

  // Update linked data when data changes
  useEffect(() => {
    setLinkedData(data)
    setLinkedColumns(columns)
  }, [data, columns, setLinkedData, setLinkedColumns])

  // Parse CSV data
  const parseData = (text, fileName) => {
    try {
      let parsedData = []

      if (fileName.endsWith('.json')) {
        parsedData = JSON.parse(text)
      } else {
        const lines = text.trim().split('\n')
        const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''))

        for (let i = 1; i < lines.length; i++) {
          const values = lines[i].split(',').map(v => v.trim().replace(/"/g, ''))
          const row = { _index: i - 1 }
          headers.forEach((h, j) => {
            const val = values[j]
            const num = parseFloat(val)
            row[h] = isNaN(num) ? val : num
          })
          parsedData.push(row)
        }
      }

      if (parsedData.length === 0) {
        throw new Error('No data found in file')
      }

      const cols = Object.keys(parsedData[0]).filter(c => c !== '_index')
      const numeric = []
      const categorical = []

      cols.forEach(col => {
        const values = parsedData.map(r => r[col]).filter(v => v !== null && v !== undefined && v !== '')
        const numericCount = values.filter(v => typeof v === 'number').length
        if (numericCount / values.length > 0.5) {
          numeric.push(col)
        } else {
          categorical.push(col)
        }
      })

      setData(parsedData)
      setColumns(cols)
      setNumericColumns(numeric)
      setCategoricalColumns(categorical)
      setError(null)

      // Auto-select first columns
      if (numeric.length >= 2) {
        setXColumn(numeric[0])
        setYColumn(numeric[1])
        setHistColumn(numeric[0])
      }
      if (categorical.length > 0) {
        setColorColumn(categorical[0])
      }

      clearSelection()
    } catch (err) {
      setError(err.message)
    }
  }

  // Handle file upload
  const handleFileUpload = (event) => {
    const file = event.target.files[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (e) => parseData(e.target.result, file.name)
    reader.readAsText(file)
  }

  // Generate sample data
  const generateSampleData = () => {
    const sampleData = []
    const groups = ['Group A', 'Group B', 'Group C']
    const regions = ['North', 'South', 'East', 'West']

    for (let i = 0; i < 150; i++) {
      const group = groups[Math.floor(Math.random() * 3)]
      const baseX = group === 'Group A' ? 30 : group === 'Group B' ? 50 : 70
      const baseY = group === 'Group A' ? 40 : group === 'Group B' ? 60 : 50

      sampleData.push({
        _index: i,
        x: baseX + (Math.random() - 0.5) * 40,
        y: baseY + (Math.random() - 0.5) * 30,
        z: Math.random() * 100,
        size: Math.random() * 50 + 10,
        group: group,
        region: regions[Math.floor(Math.random() * 4)],
        category: Math.random() > 0.5 ? 'Type 1' : 'Type 2'
      })
    }

    setData(sampleData)
    setColumns(['x', 'y', 'z', 'size', 'group', 'region', 'category'])
    setNumericColumns(['x', 'y', 'z', 'size'])
    setCategoricalColumns(['group', 'region', 'category'])
    setXColumn('x')
    setYColumn('y')
    setHistColumn('x')
    setColorColumn('group')
    clearSelection()
  }

  // Handle scatter plot selection
  const handleScatterSelect = useCallback((eventData) => {
    if (!eventData || !eventData.points) return

    const indices = eventData.points.map(p => p.pointIndex)
    if (eventData.range) {
      // Brush selection
      const { x, y } = eventData.range
      selectByBrush(x, y, xColumn, yColumn, data)
    } else {
      // Point selection - add to existing
      indices.forEach(idx => {
        if (!isSelected(idx)) {
          toggleIndex(idx)
        }
      })
    }
  }, [data, xColumn, yColumn, selectByBrush, isSelected, toggleIndex])

  // Handle point click
  const handlePointClick = useCallback((eventData) => {
    if (!eventData || !eventData.points || !eventData.points[0]) return
    const pointIndex = eventData.points[0].pointIndex
    toggleIndex(pointIndex)
  }, [toggleIndex])

  // Handle hover
  const handleHover = useCallback((eventData) => {
    if (!eventData || !eventData.points || !eventData.points[0]) return
    setHighlightedIndex(eventData.points[0].pointIndex)
  }, [setHighlightedIndex])

  const handleUnhover = useCallback(() => {
    setHighlightedIndex(null)
  }, [setHighlightedIndex])

  // Generate marker colors based on selection
  const getMarkerColors = useCallback(() => {
    if (!data.length) return []

    return data.map((row, i) => {
      if (isSelected(i)) {
        return selectionColor // Selected color
      } else if (highlightedIndex === i) {
        return '#60a5fa' // Highlighted color
      } else if (hasSelection) {
        return 'rgba(148, 163, 184, 0.3)' // Dimmed when something else selected
      } else if (colorColumn && categoricalColumns.includes(colorColumn)) {
        return undefined // Let Plotly handle categorical coloring
      } else {
        return '#3b82f6' // Default blue
      }
    })
  }, [data, isSelected, highlightedIndex, hasSelection, selectionColor, colorColumn, categoricalColumns])

  // Generate scatter plot
  const scatterTrace = useMemo(() => {
    if (!data.length || !xColumn || !yColumn) return null

    const colors = getMarkerColors()
    const hasColorColumn = colorColumn && categoricalColumns.includes(colorColumn)

    // If using categorical colors and no selection, use grouping
    if (hasColorColumn && !hasSelection) {
      const groups = [...new Set(data.map(d => d[colorColumn]))]
      const groupColors = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6', '#ec4899']

      return groups.map((group, gi) => {
        const groupData = data.filter(d => d[colorColumn] === group)
        const groupIndices = data.map((d, i) => d[colorColumn] === group ? i : -1).filter(i => i >= 0)

        return {
          type: 'scatter',
          mode: 'markers',
          name: String(group),
          x: groupData.map(d => d[xColumn]),
          y: groupData.map(d => d[yColumn]),
          customdata: groupIndices,
          marker: {
            size: 10,
            color: groupIndices.map(i =>
              isSelected(i) ? selectionColor :
              highlightedIndex === i ? '#60a5fa' :
              groupColors[gi % groupColors.length]
            ),
            line: {
              color: groupIndices.map(i => isSelected(i) ? '#fff' : 'transparent'),
              width: groupIndices.map(i => isSelected(i) ? 2 : 0)
            }
          },
          hovertemplate: `${xColumn}: %{x}<br>${yColumn}: %{y}<br>${colorColumn}: ${group}<extra></extra>`
        }
      })
    }

    return [{
      type: 'scatter',
      mode: 'markers',
      x: data.map(d => d[xColumn]),
      y: data.map(d => d[yColumn]),
      marker: {
        size: data.map((_, i) => isSelected(i) ? 14 : 10),
        color: colors,
        line: {
          color: data.map((_, i) => isSelected(i) ? '#fff' : 'transparent'),
          width: data.map((_, i) => isSelected(i) ? 2 : 0)
        }
      },
      hovertemplate: `${xColumn}: %{x}<br>${yColumn}: %{y}<extra></extra>`
    }]
  }, [data, xColumn, yColumn, colorColumn, categoricalColumns, getMarkerColors, hasSelection, isSelected, highlightedIndex, selectionColor])

  // Generate histogram
  const histogramTrace = useMemo(() => {
    if (!data.length || !histColumn) return null

    if (hasSelection) {
      // Show both selected and unselected
      const selectedData = data.filter((_, i) => isSelected(i))
      const unselectedData = data.filter((_, i) => !isSelected(i))

      return [
        {
          type: 'histogram',
          x: unselectedData.map(d => d[histColumn]),
          name: 'Unselected',
          marker: { color: 'rgba(148, 163, 184, 0.5)' },
          opacity: 0.7
        },
        {
          type: 'histogram',
          x: selectedData.map(d => d[histColumn]),
          name: 'Selected',
          marker: { color: selectionColor },
          opacity: 0.9
        }
      ]
    }

    return [{
      type: 'histogram',
      x: data.map(d => d[histColumn]),
      marker: { color: '#3b82f6' }
    }]
  }, [data, histColumn, hasSelection, isSelected, selectionColor])

  // Generate box plot
  const boxTrace = useMemo(() => {
    if (!data.length || !yColumn) return null

    if (hasSelection) {
      const selectedData = data.filter((_, i) => isSelected(i))
      const unselectedData = data.filter((_, i) => !isSelected(i))

      return [
        {
          type: 'box',
          y: unselectedData.map(d => d[yColumn]),
          name: 'Unselected',
          marker: { color: 'rgba(148, 163, 184, 0.7)' },
          boxpoints: false
        },
        {
          type: 'box',
          y: selectedData.map(d => d[yColumn]),
          name: 'Selected',
          marker: { color: selectionColor },
          boxpoints: 'all',
          jitter: 0.3
        }
      ]
    }

    if (colorColumn && categoricalColumns.includes(colorColumn)) {
      const groups = [...new Set(data.map(d => d[colorColumn]))]
      return groups.map(group => ({
        type: 'box',
        y: data.filter(d => d[colorColumn] === group).map(d => d[yColumn]),
        name: String(group),
        boxpoints: false
      }))
    }

    return [{
      type: 'box',
      y: data.map(d => d[yColumn]),
      name: yColumn,
      marker: { color: '#3b82f6' },
      boxpoints: 'outliers'
    }]
  }, [data, yColumn, colorColumn, categoricalColumns, hasSelection, isSelected, selectionColor])

  // Layout for plots
  const plotLayout = {
    paper_bgcolor: '#1e293b',
    plot_bgcolor: '#0f172a',
    font: { color: '#e2e8f0' },
    margin: { t: 30, b: 40, l: 50, r: 20 },
    xaxis: { gridcolor: '#475569', zerolinecolor: '#475569', tickfont: { color: '#94a3b8' } },
    yaxis: { gridcolor: '#475569', zerolinecolor: '#475569', tickfont: { color: '#94a3b8' } },
    showlegend: true,
    legend: { font: { color: '#e2e8f0' }, bgcolor: 'rgba(30, 41, 59, 0.8)' },
    dragmode: 'select'
  }

  // Table pagination
  const paginatedData = useMemo(() => {
    const start = tablePage * tablePageSize
    return data.slice(start, start + tablePageSize)
  }, [data, tablePage, tablePageSize])

  const totalPages = Math.ceil(data.length / tablePageSize)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-900/30 to-cyan-900/30 backdrop-blur-lg rounded-2xl p-8 border border-blue-700/50">
        <div className="flex items-center space-x-3 mb-4">
          <Link2 className="w-10 h-10 text-blue-400" />
          <h2 className="text-4xl font-bold text-gray-100">Linked Data Explorer</h2>
        </div>
        <p className="text-gray-300 text-lg">
          Explore data with linked visualizations. Select points in any plot to see them highlighted across all views.
          Brush to select ranges, click points to add/remove from selection.
        </p>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-900/30 rounded-xl p-4 border border-red-700/50">
          <p className="text-red-200">Error: {error}</p>
        </div>
      )}

      {/* Controls */}
      <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700/50">
        <div className="flex flex-wrap items-center gap-4">
          {/* Data Import */}
          <div className="flex items-center gap-2">
            <input
              type="file"
              id="fileInput"
              onChange={handleFileUpload}
              accept=".csv,.json"
              className="hidden"
            />
            <label
              htmlFor="fileInput"
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 cursor-pointer"
            >
              <Upload className="w-4 h-4" />
              Import
            </label>
            <button
              onClick={generateSampleData}
              className="flex items-center gap-2 px-4 py-2 bg-slate-700 text-gray-200 rounded-lg hover:bg-slate-600"
            >
              <Plus className="w-4 h-4" />
              Sample Data
            </button>
          </div>

          {/* Column Selectors */}
          {data.length > 0 && (
            <>
              <select
                value={xColumn}
                onChange={(e) => setXColumn(e.target.value)}
                className="px-3 py-2 bg-slate-700 text-gray-100 rounded-lg border border-slate-600"
              >
                <option value="">X Axis</option>
                {numericColumns.map(col => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>

              <select
                value={yColumn}
                onChange={(e) => setYColumn(e.target.value)}
                className="px-3 py-2 bg-slate-700 text-gray-100 rounded-lg border border-slate-600"
              >
                <option value="">Y Axis</option>
                {numericColumns.map(col => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>

              <select
                value={colorColumn}
                onChange={(e) => setColorColumn(e.target.value)}
                className="px-3 py-2 bg-slate-700 text-gray-100 rounded-lg border border-slate-600"
              >
                <option value="">Color By</option>
                {categoricalColumns.map(col => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>

              <select
                value={histColumn}
                onChange={(e) => setHistColumn(e.target.value)}
                className="px-3 py-2 bg-slate-700 text-gray-100 rounded-lg border border-slate-600"
              >
                <option value="">Histogram</option>
                {numericColumns.map(col => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
            </>
          )}

          {/* Selection Controls */}
          {hasSelection && (
            <button
              onClick={clearSelection}
              className="flex items-center gap-2 px-4 py-2 bg-red-600/50 text-red-200 rounded-lg hover:bg-red-600"
            >
              <RotateCcw className="w-4 h-4" />
              Clear Selection ({selectionCount})
            </button>
          )}

          <button
            onClick={() => setShowTable(!showTable)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg ${
              showTable ? 'bg-green-600 text-white' : 'bg-slate-700 text-gray-200'
            }`}
          >
            <Table className="w-4 h-4" />
            Table
          </button>
        </div>

        {/* Data info */}
        {data.length > 0 && (
          <div className="mt-3 text-gray-400 text-sm">
            {data.length} rows x {columns.length} columns
            {hasSelection && (
              <span className="ml-4 text-amber-400">
                {selectionCount} selected ({selectionStats?.percentage}%)
              </span>
            )}
          </div>
        )}
      </div>

      {/* Selection Statistics */}
      {selectionStats && (
        <div className="bg-amber-900/20 rounded-xl p-4 border border-amber-700/30">
          <h3 className="text-amber-200 font-semibold mb-3 flex items-center gap-2">
            <Filter className="w-5 h-5" />
            Selection Statistics ({selectionStats.count} of {selectionStats.totalCount} points)
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(selectionStats.columnStats).slice(0, 4).map(([col, stats]) => (
              <div key={col} className="bg-slate-800/50 rounded-lg p-3">
                <p className="text-gray-400 text-xs mb-1">{col}</p>
                <p className="text-gray-100 font-bold">{stats.mean.toFixed(2)}</p>
                <p className="text-gray-500 text-xs">
                  Min: {stats.min.toFixed(2)} | Max: {stats.max.toFixed(2)}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Linked Plots */}
      {data.length > 0 && (
        <div className="grid grid-cols-2 gap-4">
          {/* Scatter Plot */}
          <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700/50">
            <h3 className="text-gray-100 font-semibold mb-2 flex items-center gap-2">
              <ScatterChart className="w-5 h-5 text-blue-400" />
              Scatter Plot
              <span className="text-gray-500 text-sm ml-2">Drag to brush select</span>
            </h3>
            {scatterTrace && (
              <Plot
                data={scatterTrace}
                layout={{
                  ...plotLayout,
                  xaxis: { ...plotLayout.xaxis, title: { text: xColumn, font: { color: '#e2e8f0' } } },
                  yaxis: { ...plotLayout.yaxis, title: { text: yColumn, font: { color: '#e2e8f0' } } },
                  height: 350
                }}
                config={{ responsive: true }}
                style={{ width: '100%' }}
                onSelected={handleScatterSelect}
                onClick={handlePointClick}
                onHover={handleHover}
                onUnhover={handleUnhover}
              />
            )}
          </div>

          {/* Histogram */}
          <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700/50">
            <h3 className="text-gray-100 font-semibold mb-2 flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-green-400" />
              Histogram
            </h3>
            {histogramTrace && (
              <Plot
                data={histogramTrace}
                layout={{
                  ...plotLayout,
                  xaxis: { ...plotLayout.xaxis, title: { text: histColumn, font: { color: '#e2e8f0' } } },
                  yaxis: { ...plotLayout.yaxis, title: { text: 'Count', font: { color: '#e2e8f0' } } },
                  barmode: 'overlay',
                  height: 350
                }}
                config={{ responsive: true }}
                style={{ width: '100%' }}
              />
            )}
          </div>

          {/* Box Plot */}
          <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700/50 col-span-2">
            <h3 className="text-gray-100 font-semibold mb-2">Box Plot Comparison</h3>
            {boxTrace && (
              <Plot
                data={boxTrace}
                layout={{
                  ...plotLayout,
                  yaxis: { ...plotLayout.yaxis, title: { text: yColumn, font: { color: '#e2e8f0' } } },
                  height: 250
                }}
                config={{ responsive: true }}
                style={{ width: '100%' }}
              />
            )}
          </div>
        </div>
      )}

      {/* Linked Data Table */}
      {showTable && data.length > 0 && (
        <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700/50">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-gray-100 font-semibold flex items-center gap-2">
              <Table className="w-5 h-5 text-purple-400" />
              Data Table
              <span className="text-gray-500 text-sm ml-2">Click rows to toggle selection</span>
            </h3>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setTablePage(Math.max(0, tablePage - 1))}
                disabled={tablePage === 0}
                className="px-2 py-1 bg-slate-700 text-gray-300 rounded disabled:opacity-50"
              >
                Prev
              </button>
              <span className="text-gray-400 text-sm">
                Page {tablePage + 1} of {totalPages}
              </span>
              <button
                onClick={() => setTablePage(Math.min(totalPages - 1, tablePage + 1))}
                disabled={tablePage >= totalPages - 1}
                className="px-2 py-1 bg-slate-700 text-gray-300 rounded disabled:opacity-50"
              >
                Next
              </button>
            </div>
          </div>

          <div className="overflow-x-auto max-h-[400px] overflow-y-auto">
            <table className="w-full">
              <thead className="sticky top-0 bg-slate-700">
                <tr>
                  <th className="px-3 py-2 text-left text-gray-300 text-sm">#</th>
                  {columns.slice(0, 8).map(col => (
                    <th key={col} className="px-3 py-2 text-left text-gray-300 text-sm">{col}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {paginatedData.map((row, i) => {
                  const actualIndex = tablePage * tablePageSize + i
                  const selected = isSelected(actualIndex)
                  const highlighted = highlightedIndex === actualIndex

                  return (
                    <tr
                      key={actualIndex}
                      onClick={() => toggleIndex(actualIndex)}
                      onMouseEnter={() => setHighlightedIndex(actualIndex)}
                      onMouseLeave={() => setHighlightedIndex(null)}
                      className={`cursor-pointer transition-colors ${
                        selected
                          ? 'bg-amber-900/40 hover:bg-amber-900/60'
                          : highlighted
                            ? 'bg-blue-900/30'
                            : 'hover:bg-slate-700/50'
                      }`}
                    >
                      <td className="px-3 py-2 text-gray-400 text-sm">{actualIndex + 1}</td>
                      {columns.slice(0, 8).map(col => (
                        <td key={col} className="px-3 py-2 text-gray-200 text-sm">
                          {typeof row[col] === 'number' ? row[col].toFixed(2) : String(row[col])}
                        </td>
                      ))}
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Empty State */}
      {data.length === 0 && (
        <div className="bg-slate-800/50 rounded-xl p-12 border border-slate-700/50 text-center">
          <Link2 className="w-16 h-16 text-gray-600 mx-auto mb-4" />
          <p className="text-gray-400 text-lg mb-2">No data loaded</p>
          <p className="text-gray-500">Import a CSV/JSON file or use sample data to get started</p>
        </div>
      )}

      {/* Info Panel */}
      <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700/50">
        <h3 className="text-gray-100 font-semibold mb-3 flex items-center gap-2">
          <Info className="w-5 h-5 text-blue-400" />
          How to Use Linked Data Explorer
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div className="bg-slate-700/30 rounded-lg p-3">
            <h4 className="text-blue-300 font-medium mb-1">Brush Selection</h4>
            <p className="text-gray-400">
              Click and drag on the scatter plot to select a rectangular region. All points inside will be highlighted.
            </p>
          </div>
          <div className="bg-slate-700/30 rounded-lg p-3">
            <h4 className="text-green-300 font-medium mb-1">Point Selection</h4>
            <p className="text-gray-400">
              Click individual points or table rows to add/remove them from the selection.
            </p>
          </div>
          <div className="bg-slate-700/30 rounded-lg p-3">
            <h4 className="text-purple-300 font-medium mb-1">Linked Views</h4>
            <p className="text-gray-400">
              Selections sync across all plots and the data table. Statistics update automatically.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

// Main component with provider
const LinkedDataExplorer = () => {
  return (
    <SelectionProvider>
      <LinkedExplorerContent />
    </SelectionProvider>
  )
}

export default LinkedDataExplorer
