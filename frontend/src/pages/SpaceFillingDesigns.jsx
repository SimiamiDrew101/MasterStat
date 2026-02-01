import { useState } from 'react'
import axios from 'axios'
import { Plus, Trash2, Play, Download, BarChart3, Grid3X3, Shuffle, Target, Info, RefreshCw, GitCompare } from 'lucide-react'
import Plot from 'react-plotly.js'

const API_URL = import.meta.env.VITE_API_URL || ''

const SpaceFillingDesigns = () => {
  const [factors, setFactors] = useState([
    { name: 'X1', low: 0, high: 1, type: 'continuous' },
    { name: 'X2', low: 0, high: 1, type: 'continuous' }
  ])
  const [nPoints, setNPoints] = useState(20)
  const [method, setMethod] = useState('lhs')
  const [optimization, setOptimization] = useState('')
  const [scramble, setScramble] = useState(true)
  const [seed, setSeed] = useState('')
  const [results, setResults] = useState(null)
  const [comparison, setComparison] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('design')

  const methods = [
    { id: 'lhs', name: 'Latin Hypercube (LHS)', description: 'Stratified sampling with one point per stratum' },
    { id: 'sobol', name: 'Sobol Sequence', description: 'Quasi-random low-discrepancy sequence' },
    { id: 'halton', name: 'Halton Sequence', description: 'Prime-based quasi-random sequence' },
    { id: 'maximin', name: 'Maximin LHS', description: 'LHS optimized for maximum minimum distance' },
    { id: 'uniform', name: 'Uniform Random', description: 'Independent random sampling' }
  ]

  const optimizations = [
    { id: '', name: 'None', description: 'No optimization' },
    { id: 'maximin', name: 'Maximin', description: 'Maximize minimum distance' },
    { id: 'correlation', name: 'Correlation', description: 'Minimize column correlations' },
    { id: 'centermaximin', name: 'Center Maximin', description: 'Maximin with centered points' }
  ]

  const addFactor = () => {
    const newName = `X${factors.length + 1}`
    setFactors([...factors, { name: newName, low: 0, high: 1, type: 'continuous' }])
  }

  const removeFactor = (index) => {
    if (factors.length > 2) {
      setFactors(factors.filter((_, i) => i !== index))
    }
  }

  const updateFactor = (index, field, value) => {
    const updated = [...factors]
    updated[index][field] = field === 'name' || field === 'type' ? value : parseFloat(value) || 0
    setFactors(updated)
  }

  const generateDesign = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await axios.post(`${API_URL}/api/space-filling/generate`, {
        factors,
        n_points: nPoints,
        method,
        optimization: method === 'lhs' ? optimization : null,
        scramble,
        seed: seed ? parseInt(seed) : null
      })
      setResults(response.data)
      setActiveTab('results')
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  const compareAllMethods = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await axios.post(`${API_URL}/api/space-filling/compare`, {
        factors,
        n_points: nPoints,
        seed: seed ? parseInt(seed) : null
      })
      setComparison(response.data)
      setActiveTab('comparison')
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  const downloadDesign = () => {
    if (!results) return
    const headers = ['Run', ...factors.map(f => f.name)]
    const rows = results.design_matrix.map(row =>
      [row.Run, ...factors.map(f => row[f.name].toFixed(6))].join(',')
    )
    const csv = [headers.join(','), ...rows].join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `space_filling_${method}_${nPoints}pts.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  const renderScatterMatrix = () => {
    if (!results || factors.length < 2) return null

    const data = results.design_matrix
    const factorNames = factors.map(f => f.name)
    const n = Math.min(factorNames.length, 4) // Limit to 4 factors for readability

    const traces = []
    const annotations = []

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const xaxis = j === 0 ? 'x' : `x${j + 1}`
        const yaxis = i === 0 ? 'y' : `y${i + 1}`

        if (i === j) {
          // Diagonal: histogram
          traces.push({
            type: 'histogram',
            x: data.map(d => d[factorNames[j]]),
            xaxis,
            yaxis,
            marker: { color: '#3b82f6' },
            showlegend: false
          })
        } else {
          // Off-diagonal: scatter
          traces.push({
            type: 'scatter',
            mode: 'markers',
            x: data.map(d => d[factorNames[j]]),
            y: data.map(d => d[factorNames[i]]),
            xaxis,
            yaxis,
            marker: {
              color: '#3b82f6',
              size: 6,
              opacity: 0.7
            },
            showlegend: false
          })
        }
      }
    }

    // Create axis layout
    const layout = {
      height: 500,
      paper_bgcolor: '#1e293b',
      plot_bgcolor: '#0f172a',
      font: { color: '#e2e8f0' },
      margin: { l: 60, r: 20, t: 40, b: 60 },
      showlegend: false
    }

    // Add grid layout
    const gap = 0.02
    const size = (1 - gap * (n + 1)) / n

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const xKey = j === 0 ? 'xaxis' : `xaxis${j + 1}`
        const yKey = i === 0 ? 'yaxis' : `yaxis${i + 1}`

        layout[xKey] = {
          domain: [gap + j * (size + gap), gap + j * (size + gap) + size],
          showgrid: true,
          gridcolor: '#475569',
          title: i === n - 1 ? factorNames[j] : '',
          tickfont: { size: 10 }
        }
        layout[yKey] = {
          domain: [1 - gap - (i + 1) * (size + gap) + gap, 1 - gap - i * (size + gap)],
          showgrid: true,
          gridcolor: '#475569',
          title: j === 0 ? factorNames[i] : '',
          tickfont: { size: 10 }
        }
      }
    }

    return (
      <Plot
        data={traces}
        layout={layout}
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: '100%' }}
      />
    )
  }

  const render2DProjection = () => {
    if (!results || factors.length < 2) return null

    const data = results.design_matrix
    const x = data.map(d => d[factors[0].name])
    const y = data.map(d => d[factors[1].name])

    return (
      <Plot
        data={[{
          type: 'scatter',
          mode: 'markers',
          x,
          y,
          marker: {
            color: '#3b82f6',
            size: 10,
            line: { color: '#1e40af', width: 1 }
          },
          text: data.map((_, i) => `Run ${i + 1}`),
          hovertemplate: `${factors[0].name}: %{x:.4f}<br>${factors[1].name}: %{y:.4f}<extra>%{text}</extra>`
        }]}
        layout={{
          title: `${factors[0].name} vs ${factors[1].name}`,
          xaxis: {
            title: factors[0].name,
            gridcolor: '#475569',
            range: [factors[0].low - 0.05 * (factors[0].high - factors[0].low),
                    factors[0].high + 0.05 * (factors[0].high - factors[0].low)]
          },
          yaxis: {
            title: factors[1].name,
            gridcolor: '#475569',
            range: [factors[1].low - 0.05 * (factors[1].high - factors[1].low),
                    factors[1].high + 0.05 * (factors[1].high - factors[1].low)]
          },
          paper_bgcolor: '#1e293b',
          plot_bgcolor: '#0f172a',
          font: { color: '#e2e8f0' },
          height: 400,
          margin: { l: 60, r: 20, t: 40, b: 60 }
        }}
        config={{ responsive: true }}
        style={{ width: '100%' }}
      />
    )
  }

  const renderComparisonChart = () => {
    if (!comparison) return null

    const methods = Object.keys(comparison.comparison)
    const metrics = ['discrepancy', 'min_distance', 'projection_uniformity', 'space_filling_score']
    const metricLabels = ['Discrepancy (lower=better)', 'Min Distance', 'Projection Uniformity', 'Overall Score']

    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {metrics.map((metric, idx) => (
          <Plot
            key={metric}
            data={[{
              type: 'bar',
              x: methods.map(m => m.toUpperCase()),
              y: methods.map(m => comparison.comparison[m].metrics[metric]),
              marker: {
                color: methods.map(m =>
                  m === comparison.recommendations.best_overall ? '#22c55e' : '#3b82f6'
                )
              }
            }]}
            layout={{
              title: metricLabels[idx],
              xaxis: { title: '' },
              yaxis: { title: '', gridcolor: '#475569' },
              paper_bgcolor: '#1e293b',
              plot_bgcolor: '#0f172a',
              font: { color: '#e2e8f0', size: 11 },
              height: 250,
              margin: { l: 50, r: 20, t: 40, b: 40 }
            }}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: '100%' }}
          />
        ))}
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-cyan-900/50 to-blue-900/50 rounded-xl p-6 border border-cyan-700/50">
        <h1 className="text-3xl font-bold text-gray-100 mb-2 flex items-center gap-3">
          <Grid3X3 className="w-8 h-8 text-cyan-400" />
          Space-Filling Designs
        </h1>
        <p className="text-gray-300">
          Generate space-filling experimental designs for computer experiments, simulation studies, and surrogate modeling.
          Methods include Latin Hypercube, Sobol sequences, and Halton sequences.
        </p>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-slate-700 pb-2">
        {['design', 'results', 'comparison'].map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 rounded-t-lg font-medium transition-colors ${
              activeTab === tab
                ? 'bg-slate-700 text-white'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {error && (
        <div className="bg-red-900/50 border border-red-500 rounded-lg p-4 text-red-200">
          {error}
        </div>
      )}

      {/* Design Tab */}
      {activeTab === 'design' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Factors */}
          <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700">
            <h2 className="text-xl font-semibold text-gray-100 mb-4 flex items-center gap-2">
              <Target className="w-5 h-5 text-cyan-400" />
              Factors
            </h2>

            <div className="space-y-3">
              {factors.map((factor, idx) => (
                <div key={idx} className="flex items-center gap-2">
                  <input
                    type="text"
                    value={factor.name}
                    onChange={(e) => updateFactor(idx, 'name', e.target.value)}
                    className="w-24 bg-slate-900 text-gray-100 px-3 py-2 rounded border border-slate-600"
                    placeholder="Name"
                  />
                  <input
                    type="number"
                    value={factor.low}
                    onChange={(e) => updateFactor(idx, 'low', e.target.value)}
                    className="w-24 bg-slate-900 text-gray-100 px-3 py-2 rounded border border-slate-600"
                    placeholder="Low"
                  />
                  <span className="text-gray-400">to</span>
                  <input
                    type="number"
                    value={factor.high}
                    onChange={(e) => updateFactor(idx, 'high', e.target.value)}
                    className="w-24 bg-slate-900 text-gray-100 px-3 py-2 rounded border border-slate-600"
                    placeholder="High"
                  />
                  <button
                    onClick={() => removeFactor(idx)}
                    disabled={factors.length <= 2}
                    className="p-2 text-red-400 hover:bg-red-900/30 rounded disabled:opacity-30"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>

            <button
              onClick={addFactor}
              className="mt-4 flex items-center gap-2 px-4 py-2 bg-cyan-600 hover:bg-cyan-700 text-white rounded-lg"
            >
              <Plus className="w-4 h-4" /> Add Factor
            </button>
          </div>

          {/* Settings */}
          <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700">
            <h2 className="text-xl font-semibold text-gray-100 mb-4 flex items-center gap-2">
              <Shuffle className="w-5 h-5 text-cyan-400" />
              Design Settings
            </h2>

            <div className="space-y-4">
              <div>
                <label className="block text-gray-300 mb-2">Number of Points</label>
                <input
                  type="number"
                  value={nPoints}
                  onChange={(e) => setNPoints(Math.max(2, parseInt(e.target.value) || 2))}
                  className="w-full bg-slate-900 text-gray-100 px-3 py-2 rounded border border-slate-600"
                  min={2}
                />
              </div>

              <div>
                <label className="block text-gray-300 mb-2">Method</label>
                <select
                  value={method}
                  onChange={(e) => setMethod(e.target.value)}
                  className="w-full bg-slate-900 text-gray-100 px-3 py-2 rounded border border-slate-600"
                >
                  {methods.map(m => (
                    <option key={m.id} value={m.id}>{m.name}</option>
                  ))}
                </select>
                <p className="text-gray-500 text-sm mt-1">
                  {methods.find(m => m.id === method)?.description}
                </p>
              </div>

              {method === 'lhs' && (
                <div>
                  <label className="block text-gray-300 mb-2">LHS Optimization</label>
                  <select
                    value={optimization}
                    onChange={(e) => setOptimization(e.target.value)}
                    className="w-full bg-slate-900 text-gray-100 px-3 py-2 rounded border border-slate-600"
                  >
                    {optimizations.map(o => (
                      <option key={o.id} value={o.id}>{o.name}</option>
                    ))}
                  </select>
                </div>
              )}

              {(method === 'sobol' || method === 'halton') && (
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={scramble}
                    onChange={(e) => setScramble(e.target.checked)}
                    className="w-4 h-4"
                    id="scramble"
                  />
                  <label htmlFor="scramble" className="text-gray-300">Scramble sequence</label>
                </div>
              )}

              <div>
                <label className="block text-gray-300 mb-2">Random Seed (optional)</label>
                <input
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(e.target.value)}
                  className="w-full bg-slate-900 text-gray-100 px-3 py-2 rounded border border-slate-600"
                  placeholder="Leave empty for random"
                />
              </div>
            </div>

            <div className="flex gap-3 mt-6">
              <button
                onClick={generateDesign}
                disabled={loading}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-cyan-600 hover:bg-cyan-700 text-white rounded-lg font-semibold disabled:opacity-50"
              >
                {loading ? <RefreshCw className="w-5 h-5 animate-spin" /> : <Play className="w-5 h-5" />}
                Generate Design
              </button>
              <button
                onClick={compareAllMethods}
                disabled={loading}
                className="flex items-center justify-center gap-2 px-4 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-semibold disabled:opacity-50"
              >
                <GitCompare className="w-5 h-5" />
                Compare
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Results Tab */}
      {activeTab === 'results' && results && (
        <div className="space-y-6">
          {/* Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
              <div className="text-gray-400 text-sm">Points</div>
              <div className="text-2xl font-bold text-cyan-400">{results.n_points}</div>
            </div>
            <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
              <div className="text-gray-400 text-sm">Factors</div>
              <div className="text-2xl font-bold text-cyan-400">{results.n_factors}</div>
            </div>
            <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
              <div className="text-gray-400 text-sm">Discrepancy</div>
              <div className="text-2xl font-bold text-cyan-400">{results.metrics.discrepancy.toFixed(4)}</div>
              <div className="text-xs text-gray-500">Lower is better</div>
            </div>
            <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
              <div className="text-gray-400 text-sm">Quality Score</div>
              <div className="text-2xl font-bold text-green-400">{results.metrics.space_filling_score.toFixed(1)}/100</div>
            </div>
          </div>

          {/* Method Info */}
          <div className="bg-blue-900/30 border border-blue-700/50 rounded-lg p-4 flex items-start gap-3">
            <Info className="w-5 h-5 text-blue-400 mt-0.5" />
            <div>
              <div className="font-semibold text-blue-300">{methods.find(m => m.id === results.method)?.name}</div>
              <div className="text-blue-200 text-sm">{results.method_description}</div>
            </div>
          </div>

          {/* Visualizations */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
              <h3 className="text-lg font-semibold text-gray-100 mb-4">2D Projection</h3>
              {render2DProjection()}
            </div>
            <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
              <h3 className="text-lg font-semibold text-gray-100 mb-4">Scatter Matrix</h3>
              {renderScatterMatrix()}
            </div>
          </div>

          {/* Detailed Metrics */}
          <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700">
            <h3 className="text-lg font-semibold text-gray-100 mb-4">Quality Metrics</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <div className="text-gray-400 text-sm">Min Distance</div>
                <div className="text-xl font-semibold text-gray-100">{results.metrics.min_distance.toFixed(4)}</div>
              </div>
              <div>
                <div className="text-gray-400 text-sm">Mean Distance</div>
                <div className="text-xl font-semibold text-gray-100">{results.metrics.mean_distance.toFixed(4)}</div>
              </div>
              <div>
                <div className="text-gray-400 text-sm">Coverage</div>
                <div className="text-xl font-semibold text-gray-100">{results.metrics.coverage.toFixed(4)}</div>
              </div>
              <div>
                <div className="text-gray-400 text-sm">Projection Uniformity</div>
                <div className="text-xl font-semibold text-gray-100">{results.metrics.projection_uniformity.toFixed(4)}</div>
              </div>
            </div>
          </div>

          {/* Design Matrix */}
          <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-100">Design Matrix</h3>
              <button
                onClick={downloadDesign}
                className="flex items-center gap-2 px-3 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg text-sm"
              >
                <Download className="w-4 h-4" /> Export CSV
              </button>
            </div>
            <div className="overflow-x-auto max-h-96">
              <table className="w-full text-sm">
                <thead className="bg-slate-900 sticky top-0">
                  <tr>
                    <th className="px-4 py-2 text-left text-gray-300">Run</th>
                    {factors.map(f => (
                      <th key={f.name} className="px-4 py-2 text-left text-gray-300">{f.name}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {results.design_matrix.map((row, idx) => (
                    <tr key={idx} className="border-t border-slate-700 hover:bg-slate-700/30">
                      <td className="px-4 py-2 text-gray-300">{row.Run}</td>
                      {factors.map(f => (
                        <td key={f.name} className="px-4 py-2 text-gray-100">{row[f.name].toFixed(4)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Comparison Tab */}
      {activeTab === 'comparison' && comparison && (
        <div className="space-y-6">
          {/* Recommendation */}
          <div className="bg-green-900/30 border border-green-700/50 rounded-xl p-6">
            <h3 className="text-xl font-semibold text-green-300 mb-2">Recommendation</h3>
            <p className="text-green-200">{comparison.recommendations.recommendation}</p>
            <div className="mt-4 flex gap-4">
              <div>
                <span className="text-gray-400 text-sm">Best Overall:</span>
                <span className="ml-2 px-2 py-1 bg-green-600 text-white rounded text-sm font-semibold">
                  {comparison.recommendations.best_overall.toUpperCase()}
                </span>
              </div>
              <div>
                <span className="text-gray-400 text-sm">Best for Uniformity:</span>
                <span className="ml-2 px-2 py-1 bg-blue-600 text-white rounded text-sm font-semibold">
                  {comparison.recommendations.best_for_uniformity.toUpperCase()}
                </span>
              </div>
              <div>
                <span className="text-gray-400 text-sm">Best for Coverage:</span>
                <span className="ml-2 px-2 py-1 bg-purple-600 text-white rounded text-sm font-semibold">
                  {comparison.recommendations.best_for_coverage.toUpperCase()}
                </span>
              </div>
            </div>
          </div>

          {/* Comparison Charts */}
          {renderComparisonChart()}

          {/* Detailed Comparison Table */}
          <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700">
            <h3 className="text-lg font-semibold text-gray-100 mb-4">Detailed Comparison</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-slate-900">
                  <tr>
                    <th className="px-4 py-2 text-left text-gray-300">Method</th>
                    <th className="px-4 py-2 text-right text-gray-300">Discrepancy</th>
                    <th className="px-4 py-2 text-right text-gray-300">Min Distance</th>
                    <th className="px-4 py-2 text-right text-gray-300">Mean Distance</th>
                    <th className="px-4 py-2 text-right text-gray-300">Coverage</th>
                    <th className="px-4 py-2 text-right text-gray-300">Proj. Uniformity</th>
                    <th className="px-4 py-2 text-right text-gray-300">Score</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(comparison.comparison).map(([method, data]) => (
                    <tr
                      key={method}
                      className={`border-t border-slate-700 ${
                        method === comparison.recommendations.best_overall ? 'bg-green-900/20' : ''
                      }`}
                    >
                      <td className="px-4 py-2 text-gray-100 font-medium">
                        {method.toUpperCase()}
                        {method === comparison.recommendations.best_overall && (
                          <span className="ml-2 text-green-400 text-xs">BEST</span>
                        )}
                      </td>
                      <td className="px-4 py-2 text-right text-gray-300">{data.metrics.discrepancy.toFixed(4)}</td>
                      <td className="px-4 py-2 text-right text-gray-300">{data.metrics.min_distance.toFixed(4)}</td>
                      <td className="px-4 py-2 text-right text-gray-300">{data.metrics.mean_distance.toFixed(4)}</td>
                      <td className="px-4 py-2 text-right text-gray-300">{data.metrics.coverage.toFixed(4)}</td>
                      <td className="px-4 py-2 text-right text-gray-300">{data.metrics.projection_uniformity.toFixed(4)}</td>
                      <td className="px-4 py-2 text-right font-semibold text-cyan-400">{data.metrics.space_filling_score.toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Rankings */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(comparison.rankings).map(([criterion, ranked]) => (
              <div key={criterion} className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
                <div className="text-gray-400 text-sm mb-2 capitalize">{criterion.replace('_', ' ')}</div>
                <ol className="space-y-1">
                  {ranked.map((method, idx) => (
                    <li key={method} className="flex items-center gap-2 text-sm">
                      <span className={`w-5 h-5 flex items-center justify-center rounded-full text-xs ${
                        idx === 0 ? 'bg-yellow-500 text-black' : 'bg-slate-600 text-gray-300'
                      }`}>
                        {idx + 1}
                      </span>
                      <span className="text-gray-200">{method.toUpperCase()}</span>
                    </li>
                  ))}
                </ol>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* No Results Message */}
      {activeTab === 'results' && !results && (
        <div className="bg-slate-800/50 rounded-xl p-12 border border-slate-700 text-center">
          <BarChart3 className="w-16 h-16 text-slate-600 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-gray-400 mb-2">No Design Generated</h3>
          <p className="text-gray-500">Configure your factors and settings, then click "Generate Design"</p>
        </div>
      )}

      {activeTab === 'comparison' && !comparison && (
        <div className="bg-slate-800/50 rounded-xl p-12 border border-slate-700 text-center">
          <GitCompare className="w-16 h-16 text-slate-600 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-gray-400 mb-2">No Comparison Available</h3>
          <p className="text-gray-500">Click "Compare" to compare all methods for your configuration</p>
        </div>
      )}
    </div>
  )
}

export default SpaceFillingDesigns
