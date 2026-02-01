import { useState, useCallback, useRef, useEffect } from 'react'
import Plot from 'react-plotly.js'
import {
  LayoutGrid,
  Columns,
  Rows,
  Palette,
  Circle,
  Tag,
  BarChart,
  LineChart,
  ScatterChart,
  BoxSelect,
  Activity,
  Grid3x3,
  Trash2,
  Save,
  Upload,
  Download,
  Plus,
  Settings,
  Eye,
  EyeOff,
  Layers,
  GripVertical,
  X,
  Target
} from 'lucide-react'

const API_URL = import.meta.env.VITE_API_URL || ''

// Chart type configurations
const CHART_TYPES = {
  scatter: { name: 'Scatter', icon: ScatterChart, requiresXY: true },
  line: { name: 'Line', icon: LineChart, requiresXY: true },
  bar: { name: 'Bar', icon: BarChart, requiresXY: true },
  box: { name: 'Box', icon: BoxSelect, requiresX: true },
  histogram: { name: 'Histogram', icon: Activity, requiresX: true },
  heatmap: { name: 'Heatmap', icon: Grid3x3, requiresXYZ: true },
  contour: { name: 'Contour', icon: Layers, requiresXYZ: true },
}

// Drop zone configurations
const DROP_ZONES = {
  x: { label: 'X Axis', icon: Columns, color: 'blue', required: ['scatter', 'line', 'bar', 'box', 'histogram', 'heatmap', 'contour'] },
  y: { label: 'Y Axis', icon: Rows, color: 'green', required: ['scatter', 'line', 'bar', 'heatmap', 'contour'] },
  z: { label: 'Value (Z)', icon: Target, color: 'purple', required: ['heatmap', 'contour'] },
  group: { label: 'Group By', icon: LayoutGrid, color: 'orange', required: [] },
  color: { label: 'Color By', icon: Palette, color: 'pink', required: [] },
  size: { label: 'Size By', icon: Circle, color: 'cyan', required: [] },
  facet: { label: 'Facet By', icon: Grid3x3, color: 'amber', required: [] },
}

// Statistics overlays
const STATISTICS = {
  mean: { label: 'Mean Line', color: '#ef4444' },
  median: { label: 'Median Line', color: '#22c55e' },
  ci: { label: '95% CI', color: '#3b82f6' },
  trendline: { label: 'Trendline', color: '#8b5cf6' },
}

const GraphBuilder = () => {
  // Data state
  const [data, setData] = useState([])
  const [columns, setColumns] = useState([])
  const [numericColumns, setNumericColumns] = useState([])
  const [categoricalColumns, setCategoricalColumns] = useState([])

  // Chart configuration
  const [chartType, setChartType] = useState('scatter')
  const [assignments, setAssignments] = useState({
    x: null,
    y: null,
    z: null,
    group: null,
    color: null,
    size: null,
    facet: null,
  })
  const [statistics, setStatistics] = useState({
    mean: false,
    median: false,
    ci: false,
    trendline: false,
  })
  const [chartTitle, setChartTitle] = useState('')
  const [showLegend, setShowLegend] = useState(true)

  // UI state
  const [draggedColumn, setDraggedColumn] = useState(null)
  const [selectedColumn, setSelectedColumn] = useState(null)  // For click-to-assign
  const [previewData, setPreviewData] = useState(null)
  const [error, setError] = useState(null)
  const [savedCharts, setSavedCharts] = useState([])
  const [showSettings, setShowSettings] = useState(false)

  // File input ref
  const fileInputRef = useRef(null)

  // Parse CSV/JSON data
  const parseData = (text, fileName) => {
    try {
      let parsedData = []

      if (fileName.endsWith('.json')) {
        parsedData = JSON.parse(text)
      } else {
        // Parse CSV
        const lines = text.trim().split('\n')
        const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''))

        for (let i = 1; i < lines.length; i++) {
          const values = lines[i].split(',').map(v => v.trim().replace(/"/g, ''))
          const row = {}
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

      // Detect column types
      const cols = Object.keys(parsedData[0])
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

      // Reset assignments
      setAssignments({
        x: null, y: null, z: null, group: null, color: null, size: null, facet: null
      })
    } catch (err) {
      setError(err.message)
    }
  }

  // Handle file upload
  const handleFileUpload = (event) => {
    const file = event.target.files[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (e) => {
      parseData(e.target.result, file.name)
    }
    reader.readAsText(file)
  }

  // Generate sample data
  const generateSampleData = () => {
    const sampleData = []
    const groups = ['A', 'B', 'C']
    const categories = ['Low', 'Medium', 'High']

    for (let i = 0; i < 100; i++) {
      sampleData.push({
        x: Math.random() * 100,
        y: Math.random() * 50 + (Math.random() * 100) * 0.5,
        value: Math.random() * 100,
        group: groups[Math.floor(Math.random() * 3)],
        category: categories[Math.floor(Math.random() * 3)],
        size_var: Math.random() * 30 + 5,
      })
    }

    setData(sampleData)
    setColumns(['x', 'y', 'value', 'group', 'category', 'size_var'])
    setNumericColumns(['x', 'y', 'value', 'size_var'])
    setCategoricalColumns(['group', 'category'])
    setError(null)
  }

  // Drag handlers
  const handleDragStart = (e, column) => {
    setDraggedColumn(column)
    e.dataTransfer.setData('text/plain', column)
    e.dataTransfer.effectAllowed = 'move'
    // Set a drag image (optional but helps with visual feedback)
    if (e.target) {
      e.dataTransfer.setDragImage(e.target, 0, 0)
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    e.stopPropagation()
    e.dataTransfer.dropEffect = 'move'
  }

  const handleDragEnter = (e) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const handleDrop = (e, zone) => {
    e.preventDefault()
    e.stopPropagation()
    const column = e.dataTransfer.getData('text/plain') || draggedColumn
    if (column) {
      setAssignments(prev => ({
        ...prev,
        [zone]: column
      }))
    }
    setDraggedColumn(null)
  }

  const handleDragEnd = () => {
    setDraggedColumn(null)
  }

  // Click-to-assign handlers (alternative to drag-and-drop)
  const handleColumnClick = (column) => {
    if (selectedColumn === column) {
      setSelectedColumn(null)  // Deselect if clicking same column
    } else {
      setSelectedColumn(column)
    }
  }

  const handleZoneClick = (zone) => {
    if (selectedColumn) {
      setAssignments(prev => ({
        ...prev,
        [zone]: selectedColumn
      }))
      setSelectedColumn(null)
    }
  }

  const removeAssignment = (zone) => {
    setAssignments(prev => ({
      ...prev,
      [zone]: null
    }))
  }

  // Check if chart can be rendered
  const canRenderChart = () => {
    const type = CHART_TYPES[chartType]
    if (type.requiresXY && (!assignments.x || !assignments.y)) return false
    if (type.requiresX && !assignments.x) return false
    if (type.requiresXYZ && (!assignments.x || !assignments.y || !assignments.z)) return false
    return data.length > 0
  }

  // Generate Plotly traces
  const generateTraces = () => {
    if (!canRenderChart()) return []

    const traces = []
    const xData = data.map(d => d[assignments.x])
    const yData = assignments.y ? data.map(d => d[assignments.y]) : null
    const zData = assignments.z ? data.map(d => d[assignments.z]) : null
    const colorData = assignments.color ? data.map(d => d[assignments.color]) : null
    const sizeData = assignments.size ? data.map(d => d[assignments.size]) : null
    const groupData = assignments.group ? data.map(d => d[assignments.group]) : null

    // Get unique groups
    const groups = groupData ? [...new Set(groupData)] : [null]
    const colors = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4']

    groups.forEach((group, groupIndex) => {
      const mask = group === null ? data.map(() => true) : groupData.map(g => g === group)
      const filteredX = xData.filter((_, i) => mask[i])
      const filteredY = yData ? yData.filter((_, i) => mask[i]) : null
      const filteredZ = zData ? zData.filter((_, i) => mask[i]) : null
      const filteredColor = colorData ? colorData.filter((_, i) => mask[i]) : null
      const filteredSize = sizeData ? sizeData.filter((_, i) => mask[i]) : null

      const trace = {
        name: group || 'All',
      }

      switch (chartType) {
        case 'scatter':
          trace.type = 'scatter'
          trace.mode = 'markers'
          trace.x = filteredX
          trace.y = filteredY
          trace.marker = {
            color: filteredColor || colors[groupIndex % colors.length],
            size: filteredSize || 8,
            colorscale: filteredColor && typeof filteredColor[0] === 'number' ? 'Viridis' : undefined,
            showscale: filteredColor && typeof filteredColor[0] === 'number',
          }
          break

        case 'line':
          trace.type = 'scatter'
          trace.mode = 'lines+markers'
          // Sort by x for proper line rendering
          const sortedIndices = filteredX.map((_, i) => i).sort((a, b) => filteredX[a] - filteredX[b])
          trace.x = sortedIndices.map(i => filteredX[i])
          trace.y = sortedIndices.map(i => filteredY[i])
          trace.line = { color: colors[groupIndex % colors.length] }
          break

        case 'bar':
          trace.type = 'bar'
          trace.x = filteredX
          trace.y = filteredY
          trace.marker = { color: colors[groupIndex % colors.length] }
          break

        case 'box':
          trace.type = 'box'
          trace.y = filteredX
          trace.name = group || assignments.x
          trace.marker = { color: colors[groupIndex % colors.length] }
          break

        case 'histogram':
          trace.type = 'histogram'
          trace.x = filteredX
          trace.marker = { color: colors[groupIndex % colors.length] }
          break

        case 'heatmap':
          if (groupIndex === 0) {
            // Create 2D aggregation
            const uniqueX = [...new Set(xData)].sort((a, b) => a - b)
            const uniqueY = [...new Set(yData)].sort((a, b) => a - b)
            const zMatrix = uniqueY.map(() => uniqueX.map(() => 0))
            const counts = uniqueY.map(() => uniqueX.map(() => 0))

            data.forEach((d, i) => {
              const xi = uniqueX.indexOf(d[assignments.x])
              const yi = uniqueY.indexOf(d[assignments.y])
              if (xi >= 0 && yi >= 0) {
                zMatrix[yi][xi] += d[assignments.z] || 0
                counts[yi][xi]++
              }
            })

            // Average
            for (let i = 0; i < zMatrix.length; i++) {
              for (let j = 0; j < zMatrix[i].length; j++) {
                if (counts[i][j] > 0) {
                  zMatrix[i][j] /= counts[i][j]
                }
              }
            }

            trace.type = 'heatmap'
            trace.x = uniqueX
            trace.y = uniqueY
            trace.z = zMatrix
            trace.colorscale = 'Viridis'
          } else {
            return // Only one heatmap
          }
          break

        case 'contour':
          if (groupIndex === 0) {
            trace.type = 'contour'
            trace.x = filteredX
            trace.y = filteredY
            trace.z = filteredZ
            trace.colorscale = 'Viridis'
            trace.contours = { coloring: 'heatmap' }
          } else {
            return
          }
          break
      }

      traces.push(trace)
    })

    // Add statistics overlays for scatter/line
    if ((chartType === 'scatter' || chartType === 'line') && assignments.y) {
      const numericY = data.map(d => d[assignments.y]).filter(v => typeof v === 'number')

      if (statistics.mean) {
        const mean = numericY.reduce((a, b) => a + b, 0) / numericY.length
        traces.push({
          type: 'scatter',
          mode: 'lines',
          x: [Math.min(...xData.filter(v => typeof v === 'number')), Math.max(...xData.filter(v => typeof v === 'number'))],
          y: [mean, mean],
          line: { color: STATISTICS.mean.color, dash: 'dash', width: 2 },
          name: `Mean: ${mean.toFixed(2)}`,
          showlegend: true,
        })
      }

      if (statistics.median) {
        const sorted = [...numericY].sort((a, b) => a - b)
        const median = sorted.length % 2 === 0
          ? (sorted[sorted.length/2 - 1] + sorted[sorted.length/2]) / 2
          : sorted[Math.floor(sorted.length/2)]
        traces.push({
          type: 'scatter',
          mode: 'lines',
          x: [Math.min(...xData.filter(v => typeof v === 'number')), Math.max(...xData.filter(v => typeof v === 'number'))],
          y: [median, median],
          line: { color: STATISTICS.median.color, dash: 'dot', width: 2 },
          name: `Median: ${median.toFixed(2)}`,
          showlegend: true,
        })
      }

      if (statistics.trendline && chartType === 'scatter') {
        // Simple linear regression
        const n = Math.min(xData.length, yData.length)
        const xNum = xData.slice(0, n).filter((_, i) => typeof yData[i] === 'number')
        const yNum = yData.slice(0, n).filter((v, i) => typeof v === 'number' && typeof xData[i] === 'number')

        if (xNum.length > 1) {
          const xMean = xNum.reduce((a, b) => a + b, 0) / xNum.length
          const yMean = yNum.reduce((a, b) => a + b, 0) / yNum.length
          let num = 0, den = 0
          for (let i = 0; i < xNum.length; i++) {
            num += (xNum[i] - xMean) * (yNum[i] - yMean)
            den += (xNum[i] - xMean) ** 2
          }
          const slope = den !== 0 ? num / den : 0
          const intercept = yMean - slope * xMean
          const xMin = Math.min(...xNum)
          const xMax = Math.max(...xNum)

          traces.push({
            type: 'scatter',
            mode: 'lines',
            x: [xMin, xMax],
            y: [slope * xMin + intercept, slope * xMax + intercept],
            line: { color: STATISTICS.trendline.color, width: 2 },
            name: `y = ${slope.toFixed(2)}x + ${intercept.toFixed(2)}`,
            showlegend: true,
          })
        }
      }
    }

    return traces
  }

  // Generate layout
  const generateLayout = () => {
    return {
      title: {
        text: chartTitle || `${CHART_TYPES[chartType].name} Chart`,
        font: { color: '#e2e8f0', size: 18 }
      },
      paper_bgcolor: '#1e293b',
      plot_bgcolor: '#0f172a',
      font: { color: '#e2e8f0' },
      xaxis: {
        title: { text: assignments.x || 'X', font: { color: '#e2e8f0' } },
        gridcolor: '#475569',
        zerolinecolor: '#475569',
        tickfont: { color: '#94a3b8' }
      },
      yaxis: {
        title: { text: assignments.y || 'Y', font: { color: '#e2e8f0' } },
        gridcolor: '#475569',
        zerolinecolor: '#475569',
        tickfont: { color: '#94a3b8' }
      },
      showlegend: showLegend,
      legend: { font: { color: '#e2e8f0' }, bgcolor: 'rgba(30, 41, 59, 0.8)' },
      margin: { t: 50, b: 50, l: 60, r: 40 },
      coloraxis: {
        colorbar: {
          tickfont: { color: '#e2e8f0' },
          title: { font: { color: '#e2e8f0' } }
        }
      }
    }
  }

  // Save current chart configuration
  const saveChart = () => {
    const chartConfig = {
      id: Date.now(),
      name: chartTitle || `Chart ${savedCharts.length + 1}`,
      chartType,
      assignments: { ...assignments },
      statistics: { ...statistics },
      chartTitle,
      showLegend,
      savedAt: new Date().toISOString(),
    }
    setSavedCharts([...savedCharts, chartConfig])
  }

  // Load saved chart
  const loadChart = (config) => {
    setChartType(config.chartType)
    setAssignments(config.assignments)
    setStatistics(config.statistics)
    setChartTitle(config.chartTitle)
    setShowLegend(config.showLegend)
  }

  // Delete saved chart
  const deleteChart = (id) => {
    setSavedCharts(savedCharts.filter(c => c.id !== id))
  }

  // Export chart as image
  const exportChart = () => {
    const plotElement = document.querySelector('.js-plotly-plot')
    if (plotElement) {
      import('plotly.js-dist-min').then(Plotly => {
        Plotly.downloadImage(plotElement, {
          format: 'png',
          width: 1200,
          height: 800,
          filename: chartTitle || 'graph-builder-chart'
        })
      })
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-900/30 to-purple-900/30 backdrop-blur-lg rounded-2xl p-8 border border-indigo-700/50">
        <div className="flex items-center space-x-3 mb-4">
          <LayoutGrid className="w-10 h-10 text-indigo-400" />
          <h2 className="text-4xl font-bold text-gray-100">Graph Builder</h2>
        </div>
        <p className="text-gray-300 text-lg">
          Create visualizations by dragging columns to chart zones. Supports scatter, line, bar, box, histogram, heatmap, and contour charts.
        </p>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-900/30 backdrop-blur-lg rounded-xl p-4 border border-red-700/50">
          <p className="text-red-200 font-medium">Error: {error}</p>
        </div>
      )}

      <div className="grid grid-cols-12 gap-6">
        {/* Left Panel - Data & Columns */}
        <div className="col-span-3 space-y-4">
          {/* Data Source */}
          <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl p-4 border border-slate-700/50">
            <h3 className="text-lg font-semibold text-gray-100 mb-3">Data Source</h3>

            <div className="space-y-2">
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileUpload}
                accept=".csv,.json"
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                <Upload className="w-4 h-4" />
                Import CSV/JSON
              </button>
              <button
                onClick={generateSampleData}
                className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-slate-700 text-gray-200 rounded-lg hover:bg-slate-600 transition-colors"
              >
                <Plus className="w-4 h-4" />
                Use Sample Data
              </button>
            </div>

            {data.length > 0 && (
              <div className="mt-3 text-gray-400 text-sm">
                {data.length} rows x {columns.length} columns
              </div>
            )}
          </div>

          {/* Column List */}
          <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl p-4 border border-slate-700/50">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-lg font-semibold text-gray-100">Columns</h3>
              {selectedColumn && (
                <button
                  onClick={() => setSelectedColumn(null)}
                  className="text-xs px-2 py-1 bg-red-600/50 text-red-200 rounded hover:bg-red-600"
                >
                  Clear
                </button>
              )}
            </div>

            {selectedColumn && (
              <p className="text-green-400 text-xs mb-2">
                Click a zone on the right to assign "{selectedColumn}"
              </p>
            )}

            {!selectedColumn && columns.length > 0 && (
              <p className="text-gray-500 text-xs mb-2">
                Click a column, then click a zone to assign
              </p>
            )}

            {columns.length === 0 ? (
              <p className="text-gray-400 text-sm">Load data to see columns</p>
            ) : (
              <div className="space-y-1 max-h-[400px] overflow-y-auto">
                {columns.map(col => (
                  <div
                    key={col}
                    draggable={true}
                    onDragStart={(e) => handleDragStart(e, col)}
                    onDragEnd={handleDragEnd}
                    onClick={() => handleColumnClick(col)}
                    className={`flex items-center gap-2 px-3 py-2 rounded-lg cursor-grab active:cursor-grabbing transition-all select-none ${
                      draggedColumn === col
                        ? 'bg-blue-600/50 border border-blue-500 opacity-50'
                        : selectedColumn === col
                          ? 'bg-green-600/50 border-2 border-green-500 ring-2 ring-green-500/30'
                          : 'bg-slate-700/50 hover:bg-slate-600/50 border border-transparent hover:border-blue-500/50'
                    }`}
                  >
                    <GripVertical className="w-4 h-4 text-gray-500" />
                    <span className="text-gray-200 text-sm flex-1">{col}</span>
                    <span className={`text-xs px-1.5 py-0.5 rounded ${
                      numericColumns.includes(col)
                        ? 'bg-blue-900/50 text-blue-300'
                        : 'bg-green-900/50 text-green-300'
                    }`}>
                      {numericColumns.includes(col) ? 'Num' : 'Cat'}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Saved Charts */}
          <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl p-4 border border-slate-700/50">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-lg font-semibold text-gray-100">Saved Charts</h3>
              <button
                onClick={saveChart}
                disabled={!canRenderChart()}
                className="p-1.5 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
                title="Save current chart"
              >
                <Save className="w-4 h-4" />
              </button>
            </div>

            {savedCharts.length === 0 ? (
              <p className="text-gray-400 text-sm">No saved charts</p>
            ) : (
              <div className="space-y-2 max-h-[200px] overflow-y-auto">
                {savedCharts.map(chart => (
                  <div
                    key={chart.id}
                    className="flex items-center gap-2 px-3 py-2 bg-slate-700/50 rounded-lg"
                  >
                    <span className="text-gray-200 text-sm flex-1 truncate">{chart.name}</span>
                    <button
                      onClick={() => loadChart(chart)}
                      className="p-1 text-blue-400 hover:text-blue-300"
                      title="Load"
                    >
                      <Eye className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => deleteChart(chart.id)}
                      className="p-1 text-red-400 hover:text-red-300"
                      title="Delete"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Center Panel - Chart */}
        <div className="col-span-6 space-y-4">
          {/* Chart Type Selector */}
          <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl p-4 border border-slate-700/50">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-lg font-semibold text-gray-100">Chart Type</h3>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setShowSettings(!showSettings)}
                  className={`p-2 rounded-lg transition-colors ${
                    showSettings ? 'bg-blue-600 text-white' : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
                  }`}
                >
                  <Settings className="w-4 h-4" />
                </button>
                <button
                  onClick={exportChart}
                  disabled={!canRenderChart()}
                  className="p-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50"
                  title="Export as PNG"
                >
                  <Download className="w-4 h-4" />
                </button>
              </div>
            </div>

            <div className="flex flex-wrap gap-2">
              {Object.entries(CHART_TYPES).map(([type, config]) => {
                const Icon = config.icon
                return (
                  <button
                    key={type}
                    onClick={() => setChartType(type)}
                    className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-colors ${
                      chartType === type
                        ? 'bg-blue-600 text-white'
                        : 'bg-slate-700/50 text-gray-300 hover:bg-slate-600/50'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    {config.name}
                  </button>
                )
              })}
            </div>

            {/* Settings Panel */}
            {showSettings && (
              <div className="mt-4 pt-4 border-t border-slate-700 space-y-3">
                <div>
                  <label className="block text-gray-300 text-sm mb-1">Chart Title</label>
                  <input
                    type="text"
                    value={chartTitle}
                    onChange={(e) => setChartTitle(e.target.value)}
                    placeholder="Enter chart title..."
                    className="w-full px-3 py-2 bg-slate-700 text-gray-100 rounded-lg border border-slate-600 focus:border-blue-500 focus:outline-none"
                  />
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="showLegend"
                    checked={showLegend}
                    onChange={(e) => setShowLegend(e.target.checked)}
                    className="rounded bg-slate-700 border-slate-600"
                  />
                  <label htmlFor="showLegend" className="text-gray-300 text-sm">Show Legend</label>
                </div>
              </div>
            )}
          </div>

          {/* Chart Display */}
          <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl p-4 border border-slate-700/50 min-h-[500px]">
            {canRenderChart() ? (
              <Plot
                data={generateTraces()}
                layout={generateLayout()}
                config={{ responsive: true }}
                style={{ width: '100%', height: '480px' }}
              />
            ) : (
              <div className="h-[480px] flex items-center justify-center">
                <div className="text-center">
                  <LayoutGrid className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                  <p className="text-gray-400 text-lg mb-2">Drag columns to zones to create a chart</p>
                  <p className="text-gray-500 text-sm">
                    {data.length === 0
                      ? 'First, import data or use sample data'
                      : `Required: ${DROP_ZONES.x.required.includes(chartType) ? 'X' : ''}${DROP_ZONES.y.required.includes(chartType) ? ', Y' : ''}${DROP_ZONES.z.required.includes(chartType) ? ', Z' : ''}`
                    }
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Panel - Drop Zones & Statistics */}
        <div className="col-span-3 space-y-4">
          {/* Drop Zones */}
          <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl p-4 border border-slate-700/50">
            <h3 className="text-lg font-semibold text-gray-100 mb-3">Chart Mapping</h3>

            <div className="space-y-2">
              {Object.entries(DROP_ZONES).map(([zone, config]) => {
                const Icon = config.icon
                const isRequired = config.required.includes(chartType)
                const isAssigned = assignments[zone] !== null

                return (
                  <div
                    key={zone}
                    onDragOver={handleDragOver}
                    onDragEnter={handleDragEnter}
                    onDrop={(e) => handleDrop(e, zone)}
                    onClick={() => handleZoneClick(zone)}
                    className={`flex items-center gap-2 px-3 py-2 rounded-lg border-2 border-dashed transition-all cursor-pointer ${
                      draggedColumn || selectedColumn
                        ? 'border-blue-500 bg-blue-900/20 hover:bg-blue-900/40'
                        : isAssigned
                          ? 'border-green-600 bg-green-900/20'
                          : isRequired
                            ? 'border-amber-600/50 bg-amber-900/10 hover:border-amber-500'
                            : 'border-slate-600 bg-slate-700/30 hover:border-slate-500'
                    }`}
                  >
                    <Icon className={`w-4 h-4 ${
                      isAssigned ? 'text-green-400' : isRequired ? 'text-amber-400' : 'text-gray-400'
                    }`} />
                    <span className={`text-sm flex-1 ${
                      isAssigned ? 'text-green-300' : 'text-gray-400'
                    }`}>
                      {isAssigned ? assignments[zone] : config.label}
                    </span>
                    {isRequired && !isAssigned && (
                      <span className="text-xs text-amber-400">Required</span>
                    )}
                    {isAssigned && (
                      <button
                        onClick={() => removeAssignment(zone)}
                        className="p-1 text-red-400 hover:text-red-300"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    )}
                  </div>
                )
              })}
            </div>
          </div>

          {/* Statistics Overlays */}
          {(chartType === 'scatter' || chartType === 'line') && (
            <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl p-4 border border-slate-700/50">
              <h3 className="text-lg font-semibold text-gray-100 mb-3">Statistics</h3>

              <div className="space-y-2">
                {Object.entries(STATISTICS).map(([stat, config]) => (
                  <label
                    key={stat}
                    className="flex items-center gap-2 cursor-pointer"
                  >
                    <input
                      type="checkbox"
                      checked={statistics[stat]}
                      onChange={(e) => setStatistics(prev => ({ ...prev, [stat]: e.target.checked }))}
                      className="rounded bg-slate-700 border-slate-600"
                    />
                    <span
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: config.color }}
                    />
                    <span className="text-gray-300 text-sm">{config.label}</span>
                  </label>
                ))}
              </div>
            </div>
          )}

          {/* Quick Tips */}
          <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl p-4 border border-slate-700/50">
            <h3 className="text-lg font-semibold text-gray-100 mb-3">Tips</h3>
            <ul className="text-gray-400 text-sm space-y-2">
              <li> Drag columns from the left panel to mapping zones</li>
              <li> Use Group By to split data by category</li>
              <li> Color By adds a color scale to markers</li>
              <li> Size By scales marker size by values</li>
              <li> Add statistics overlays for analysis</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

export default GraphBuilder
