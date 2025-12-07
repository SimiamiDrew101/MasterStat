import { useState, useMemo } from 'react'
import Plot from 'react-plotly.js'
import { BarChart3 } from 'lucide-react'

const Histogram = ({
  data,
  variableName = 'Variable',
  title = null,
  showNormalCurve = true,
  showKDE = false,
  binMethod = 'auto'
}) => {
  const [binRule, setBinRule] = useState(binMethod)
  const [showNormal, setShowNormal] = useState(showNormalCurve)
  const [showDensity, setShowDensity] = useState(showKDE)

  // Calculate statistics
  const stats = useMemo(() => {
    if (!data || data.length === 0) return null

    const validData = data.filter(d => d !== null && d !== undefined && !isNaN(d))
    if (validData.length === 0) return null

    const n = validData.length
    const mean = validData.reduce((sum, val) => sum + val, 0) / n
    const variance = validData.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (n - 1)
    const stdDev = Math.sqrt(variance)
    const min = Math.min(...validData)
    const max = Math.max(...validData)

    // Calculate optimal bin width using different rules
    const range = max - min
    const sturges = Math.ceil(Math.log2(n) + 1)
    const scott = Math.ceil(range / (3.5 * stdDev * Math.pow(n, -1/3)))
    const fd = Math.ceil(range / (2 * (validData.sort((a, b) => a - b)[Math.floor(n * 0.75)] -
                                         validData[Math.floor(n * 0.25)]) * Math.pow(n, -1/3)))

    return {
      n,
      mean,
      stdDev,
      min,
      max,
      range,
      bins: {
        auto: sturges,
        sturges,
        scott: Math.max(scott, 5),
        fd: Math.max(fd, 5)
      }
    }
  }, [data])

  // Generate normal distribution curve
  const normalCurve = useMemo(() => {
    if (!stats || !showNormal) return null

    const { mean, stdDev, min, max, n } = stats
    const points = 100
    const step = (max - min) / points
    const x = Array.from({ length: points }, (_, i) => min + i * step)

    // Calculate bin width to scale the normal curve properly
    const binWidth = (max - min) / stats.bins[binRule]
    const scaleFactor = n * binWidth

    const y = x.map(xi => {
      const z = (xi - mean) / stdDev
      const pdf = (1 / (stdDev * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * z * z)
      return pdf * scaleFactor
    })

    return { x, y }
  }, [stats, showNormal, binRule])

  // Generate KDE curve
  const kdeCurve = useMemo(() => {
    if (!stats || !showDensity || !data) return null

    const validData = data.filter(d => d !== null && d !== undefined && !isNaN(d))
    const { min, max, stdDev, n } = stats

    // Bandwidth using Silverman's rule of thumb
    const bandwidth = 1.06 * stdDev * Math.pow(n, -0.2)

    const points = 200
    const step = (max - min) / points
    const x = Array.from({ length: points }, (_, i) => min + i * step)

    // Calculate bin width to scale KDE
    const binWidth = (max - min) / stats.bins[binRule]
    const scaleFactor = n * binWidth

    const y = x.map(xi => {
      // Gaussian kernel density estimation
      const kde = validData.reduce((sum, dataPoint) => {
        const u = (xi - dataPoint) / bandwidth
        const kernel = (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * u * u)
        return sum + kernel
      }, 0) / (n * bandwidth)

      return kde * scaleFactor
    })

    return { x, y }
  }, [stats, showDensity, data, binRule])

  if (!data || data.length === 0) {
    return (
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <div className="flex items-center gap-3 mb-4">
          <BarChart3 className="w-6 h-6 text-blue-400" />
          <h3 className="text-xl font-bold text-gray-100">Histogram</h3>
        </div>
        <p className="text-gray-400">No data available for histogram</p>
      </div>
    )
  }

  if (!stats) {
    return (
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <p className="text-gray-400">Invalid data for histogram</p>
      </div>
    )
  }

  const plotData = [
    {
      x: data.filter(d => d !== null && d !== undefined && !isNaN(d)),
      type: 'histogram',
      name: variableName,
      nbinsx: stats.bins[binRule],
      marker: {
        color: 'rgba(59, 130, 246, 0.6)',
        line: {
          color: 'rgba(59, 130, 246, 1)',
          width: 1
        }
      },
      hovertemplate: '<b>Range:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
    }
  ]

  if (showNormal && normalCurve) {
    plotData.push({
      x: normalCurve.x,
      y: normalCurve.y,
      type: 'scatter',
      mode: 'lines',
      name: 'Normal Distribution',
      line: {
        color: 'rgba(239, 68, 68, 0.8)',
        width: 2,
        dash: 'dash'
      },
      hovertemplate: '<b>Value:</b> %{x:.2f}<br><b>Density:</b> %{y:.2f}<extra></extra>'
    })
  }

  if (showDensity && kdeCurve) {
    plotData.push({
      x: kdeCurve.x,
      y: kdeCurve.y,
      type: 'scatter',
      mode: 'lines',
      name: 'KDE',
      line: {
        color: 'rgba(34, 197, 94, 0.8)',
        width: 2
      },
      hovertemplate: '<b>Value:</b> %{x:.2f}<br><b>Density:</b> %{y:.2f}<extra></extra>'
    })
  }

  return (
    <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <BarChart3 className="w-6 h-6 text-blue-400" />
          <h3 className="text-xl font-bold text-gray-100">
            {title || `Histogram - ${variableName}`}
          </h3>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-4">
          {/* Bin Method Selector */}
          <div className="flex items-center gap-2">
            <label className="text-gray-300 text-sm">Bins:</label>
            <select
              value={binRule}
              onChange={(e) => setBinRule(e.target.value)}
              className="bg-slate-700 text-gray-100 px-3 py-1 rounded-lg text-sm border border-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="auto">Auto (Sturges)</option>
              <option value="sturges">Sturges ({stats.bins.sturges})</option>
              <option value="scott">Scott ({stats.bins.scott})</option>
              <option value="fd">Freedman-Diaconis ({stats.bins.fd})</option>
            </select>
          </div>

          {/* Toggle Normal Curve */}
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={showNormal}
              onChange={(e) => setShowNormal(e.target.checked)}
              className="w-4 h-4 text-blue-500 bg-slate-700 border-slate-600 rounded focus:ring-blue-500"
            />
            <span className="text-gray-300 text-sm">Normal Curve</span>
          </label>

          {/* Toggle KDE */}
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={showDensity}
              onChange={(e) => setShowDensity(e.target.checked)}
              className="w-4 h-4 text-green-500 bg-slate-700 border-slate-600 rounded focus:ring-green-500"
            />
            <span className="text-gray-300 text-sm">KDE</span>
          </label>
        </div>
      </div>

      {/* Plot */}
      <Plot
        data={plotData}
        layout={{
          autosize: true,
          height: 400,
          plot_bgcolor: 'rgba(15, 23, 42, 0.5)',
          paper_bgcolor: 'rgba(15, 23, 42, 0)',
          font: { color: '#e2e8f0', family: 'Inter, system-ui, sans-serif' },
          xaxis: {
            title: variableName,
            gridcolor: 'rgba(148, 163, 184, 0.1)',
            zerolinecolor: 'rgba(148, 163, 184, 0.2)'
          },
          yaxis: {
            title: 'Frequency',
            gridcolor: 'rgba(148, 163, 184, 0.1)',
            zerolinecolor: 'rgba(148, 163, 184, 0.2)'
          },
          margin: { l: 60, r: 40, t: 40, b: 60 },
          showlegend: true,
          legend: {
            x: 1,
            xanchor: 'right',
            y: 1,
            bgcolor: 'rgba(30, 41, 59, 0.8)',
            bordercolor: 'rgba(148, 163, 184, 0.2)',
            borderwidth: 1
          },
          hovermode: 'closest'
        }}
        config={{
          displayModeBar: true,
          displaylogo: false,
          modeBarButtonsToRemove: ['lasso2d', 'select2d'],
          toImageButtonOptions: {
            format: 'png',
            filename: `histogram_${variableName}_${new Date().toISOString().slice(0, 10)}`,
            height: 600,
            width: 800,
            scale: 2
          }
        }}
        style={{ width: '100%' }}
        useResizeHandler={true}
      />

      {/* Statistics Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4">
        <div className="bg-slate-700/30 rounded-lg p-3">
          <p className="text-gray-400 text-xs">N</p>
          <p className="text-gray-100 font-semibold">{stats.n}</p>
        </div>
        <div className="bg-slate-700/30 rounded-lg p-3">
          <p className="text-gray-400 text-xs">Mean</p>
          <p className="text-gray-100 font-semibold">{stats.mean.toFixed(3)}</p>
        </div>
        <div className="bg-slate-700/30 rounded-lg p-3">
          <p className="text-gray-400 text-xs">Std Dev</p>
          <p className="text-gray-100 font-semibold">{stats.stdDev.toFixed(3)}</p>
        </div>
        <div className="bg-slate-700/30 rounded-lg p-3">
          <p className="text-gray-400 text-xs">Range</p>
          <p className="text-gray-100 font-semibold">{stats.range.toFixed(3)}</p>
        </div>
      </div>
    </div>
  )
}

export default Histogram
