import { useState } from 'react'
import axios from 'axios'
import { TrendingUp, Play, Download, BarChart3, Activity, RefreshCw, Calendar, Layers, Target, Info, AlertCircle } from 'lucide-react'
import Plot from 'react-plotly.js'
import ExcelTable from '../components/ExcelTable'

const API_URL = import.meta.env.VITE_API_URL || ''

const TimeSeriesAnalysis = () => {
  const [tableData, setTableData] = useState(Array(20).fill(null).map(() => ['']))
  const [columns] = useState([{ key: 'value', label: 'Value' }])
  const [data, setData] = useState(null)
  const [activeTab, setActiveTab] = useState('data')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Analysis results
  const [summary, setSummary] = useState(null)
  const [decomposition, setDecomposition] = useState(null)
  const [acfPacf, setAcfPacf] = useState(null)
  const [arimaResult, setArimaResult] = useState(null)
  const [forecast, setForecast] = useState(null)
  const [stationarity, setStationarity] = useState(null)

  // Settings
  const [decompModel, setDecompModel] = useState('additive')
  const [decompPeriod, setDecompPeriod] = useState('')
  const [arimaOrder, setArimaOrder] = useState({ p: '', d: '', q: '' })
  const [autoArima, setAutoArima] = useState(true)
  const [forecastHorizon, setForecastHorizon] = useState(12)
  const [confidenceLevel, setConfidenceLevel] = useState(0.95)

  const parseData = () => {
    try {
      const values = tableData
        .map(row => row[0])
        .filter(val => val !== '' && val !== null && val !== undefined)
        .map(val => {
          const num = parseFloat(String(val).trim().replace(',', ''))
          if (isNaN(num)) throw new Error(`Invalid number: ${val}`)
          return num
        })

      if (values.length < 4) {
        setError('Need at least 4 data points')
        return
      }

      setData({ values })
      setError(null)
      setActiveTab('summary')
      fetchSummary({ values })
    } catch (err) {
      setError(err.message)
    }
  }

  const fetchSummary = async (tsData) => {
    setLoading(true)
    try {
      const response = await axios.post(`${API_URL}/api/time-series/summary`, tsData)
      setSummary(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  const runDecomposition = async () => {
    if (!data) return
    setLoading(true)
    setError(null)
    try {
      const response = await axios.post(`${API_URL}/api/time-series/decompose`, {
        data,
        model: decompModel,
        period: decompPeriod ? parseInt(decompPeriod) : null
      })
      setDecomposition(response.data)
      setActiveTab('decomposition')
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  const runAcfPacf = async () => {
    if (!data) return
    setLoading(true)
    setError(null)
    try {
      const response = await axios.post(`${API_URL}/api/time-series/acf-pacf`, {
        data,
        nlags: Math.min(40, Math.floor(data.values.length / 2))
      })
      setAcfPacf(response.data)
      setActiveTab('acf')
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  const runStationarity = async () => {
    if (!data) return
    setLoading(true)
    setError(null)
    try {
      const response = await axios.post(`${API_URL}/api/time-series/stationarity`, {
        data,
        test: 'both'
      })
      setStationarity(response.data)
      setActiveTab('stationarity')
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  const runArima = async () => {
    if (!data) return
    setLoading(true)
    setError(null)
    try {
      const order = autoArima ? null : [
        parseInt(arimaOrder.p) || 0,
        parseInt(arimaOrder.d) || 0,
        parseInt(arimaOrder.q) || 0
      ]
      const response = await axios.post(`${API_URL}/api/time-series/fit-arima`, {
        data,
        order,
        auto_order: autoArima
      })
      setArimaResult(response.data)
      setActiveTab('arima')
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  const runForecast = async () => {
    if (!data) return
    setLoading(true)
    setError(null)
    try {
      const order = autoArima ? null : [
        parseInt(arimaOrder.p) || 0,
        parseInt(arimaOrder.d) || 0,
        parseInt(arimaOrder.q) || 0
      ]
      const response = await axios.post(`${API_URL}/api/time-series/forecast`, {
        data,
        horizon: forecastHorizon,
        order,
        confidence_level: confidenceLevel
      })
      setForecast(response.data)
      setActiveTab('forecast')
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  const loadSampleData = (type) => {
    const samples = {
      trend: Array.from({ length: 50 }, (_, i) => (10 + i * 0.5 + Math.random() * 2).toFixed(2)),
      seasonal: Array.from({ length: 48 }, (_, i) => (50 + 20 * Math.sin(2 * Math.PI * i / 12) + Math.random() * 5).toFixed(2)),
      random: Array.from({ length: 40 }, () => (100 + Math.random() * 20 - 10).toFixed(2))
    }
    const newTableData = samples[type].map(val => [val])
    // Add empty rows to reach minimum
    while (newTableData.length < 20) {
      newTableData.push([''])
    }
    setTableData(newTableData)
  }

  const renderTimeSeriesPlot = (values, title, color = '#3b82f6') => (
    <Plot
      data={[{
        type: 'scatter',
        mode: 'lines+markers',
        y: values,
        x: values.map((_, i) => i + 1),
        line: { color, width: 2 },
        marker: { size: 4 }
      }]}
      layout={{
        title,
        xaxis: { title: 'Time', gridcolor: '#475569' },
        yaxis: { title: 'Value', gridcolor: '#475569' },
        paper_bgcolor: '#1e293b',
        plot_bgcolor: '#0f172a',
        font: { color: '#e2e8f0' },
        height: 300,
        margin: { l: 60, r: 20, t: 40, b: 50 }
      }}
      config={{ responsive: true }}
      style={{ width: '100%' }}
    />
  )

  const renderAcfPacfPlot = () => {
    if (!acfPacf) return null

    const createBarTrace = (values, ci, name, color) => ({
      type: 'bar',
      y: values,
      x: values.map((_, i) => i),
      marker: { color },
      name
    })

    const createCILines = (n, ci) => [
      {
        type: 'scatter',
        mode: 'lines',
        y: Array(n).fill(ci),
        x: Array.from({ length: n }, (_, i) => i),
        line: { color: '#ef4444', dash: 'dash', width: 1 },
        showlegend: false
      },
      {
        type: 'scatter',
        mode: 'lines',
        y: Array(n).fill(-ci),
        x: Array.from({ length: n }, (_, i) => i),
        line: { color: '#ef4444', dash: 'dash', width: 1 },
        showlegend: false
      }
    ]

    return (
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Plot
          data={[
            createBarTrace(acfPacf.acf.values, acfPacf.acf.confidence_interval, 'ACF', '#3b82f6'),
            ...createCILines(acfPacf.acf.values.length, acfPacf.acf.confidence_interval)
          ]}
          layout={{
            title: 'Autocorrelation Function (ACF)',
            xaxis: { title: 'Lag', gridcolor: '#475569' },
            yaxis: { title: 'ACF', gridcolor: '#475569', range: [-1, 1] },
            paper_bgcolor: '#1e293b',
            plot_bgcolor: '#0f172a',
            font: { color: '#e2e8f0' },
            height: 300,
            margin: { l: 60, r: 20, t: 40, b: 50 },
            showlegend: false
          }}
          config={{ responsive: true }}
          style={{ width: '100%' }}
        />
        <Plot
          data={[
            createBarTrace(acfPacf.pacf.values, acfPacf.pacf.confidence_interval, 'PACF', '#22c55e'),
            ...createCILines(acfPacf.pacf.values.length, acfPacf.pacf.confidence_interval)
          ]}
          layout={{
            title: 'Partial Autocorrelation Function (PACF)',
            xaxis: { title: 'Lag', gridcolor: '#475569' },
            yaxis: { title: 'PACF', gridcolor: '#475569', range: [-1, 1] },
            paper_bgcolor: '#1e293b',
            plot_bgcolor: '#0f172a',
            font: { color: '#e2e8f0' },
            height: 300,
            margin: { l: 60, r: 20, t: 40, b: 50 },
            showlegend: false
          }}
          config={{ responsive: true }}
          style={{ width: '100%' }}
        />
      </div>
    )
  }

  const renderDecompositionPlot = () => {
    if (!decomposition) return null

    const plots = [
      { data: decomposition.original, title: 'Original', color: '#3b82f6' },
      { data: decomposition.trend, title: 'Trend', color: '#22c55e' },
      { data: decomposition.seasonal, title: 'Seasonal', color: '#f59e0b' },
      { data: decomposition.residual, title: 'Residual', color: '#ef4444' }
    ]

    return (
      <div className="grid grid-cols-1 gap-2">
        {plots.map(({ data, title, color }) => (
          <Plot
            key={title}
            data={[{
              type: 'scatter',
              mode: 'lines',
              y: data,
              x: data.map((_, i) => i + 1),
              line: { color, width: 1.5 }
            }]}
            layout={{
              title: { text: title, font: { size: 12 } },
              xaxis: { showticklabels: title === 'Residual', gridcolor: '#475569' },
              yaxis: { gridcolor: '#475569' },
              paper_bgcolor: '#1e293b',
              plot_bgcolor: '#0f172a',
              font: { color: '#e2e8f0' },
              height: 120,
              margin: { l: 60, r: 20, t: 30, b: title === 'Residual' ? 30 : 10 }
            }}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: '100%' }}
          />
        ))}
      </div>
    )
  }

  const renderForecastPlot = () => {
    if (!forecast) return null

    return (
      <Plot
        data={[
          {
            type: 'scatter',
            mode: 'lines+markers',
            y: forecast.historical.values,
            x: forecast.historical.index,
            name: 'Historical',
            line: { color: '#3b82f6', width: 2 },
            marker: { size: 4 }
          },
          {
            type: 'scatter',
            mode: 'lines+markers',
            y: forecast.forecast.values,
            x: forecast.forecast.index,
            name: 'Forecast',
            line: { color: '#22c55e', width: 2 },
            marker: { size: 6 }
          },
          {
            type: 'scatter',
            mode: 'lines',
            y: [...forecast.forecast.upper_ci, ...forecast.forecast.lower_ci.slice().reverse()],
            x: [...forecast.forecast.index, ...forecast.forecast.index.slice().reverse()],
            fill: 'toself',
            fillcolor: 'rgba(34, 197, 94, 0.2)',
            line: { color: 'transparent' },
            name: `${(forecast.forecast.confidence_level * 100).toFixed(0)}% CI`
          }
        ]}
        layout={{
          title: 'Time Series Forecast',
          xaxis: { title: 'Time', gridcolor: '#475569' },
          yaxis: { title: 'Value', gridcolor: '#475569' },
          paper_bgcolor: '#1e293b',
          plot_bgcolor: '#0f172a',
          font: { color: '#e2e8f0' },
          height: 400,
          margin: { l: 60, r: 20, t: 40, b: 50 },
          legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,0)' }
        }}
        config={{ responsive: true }}
        style={{ width: '100%' }}
      />
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-900/50 to-purple-900/50 rounded-xl p-6 border border-indigo-700/50">
        <h1 className="text-3xl font-bold text-gray-100 mb-2 flex items-center gap-3">
          <TrendingUp className="w-8 h-8 text-indigo-400" />
          Time Series Analysis
        </h1>
        <p className="text-gray-300">
          Analyze time series data with decomposition, ARIMA modeling, forecasting, and stationarity tests.
        </p>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-slate-700 pb-2 overflow-x-auto">
        {['data', 'summary', 'decomposition', 'acf', 'stationarity', 'arima', 'forecast'].map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 rounded-t-lg font-medium transition-colors whitespace-nowrap ${
              activeTab === tab
                ? 'bg-slate-700 text-white'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            {tab === 'acf' ? 'ACF/PACF' : tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {error && (
        <div className="bg-red-900/50 border border-red-500 rounded-lg p-4 text-red-200 flex items-center gap-2">
          <AlertCircle className="w-5 h-5" />
          {error}
        </div>
      )}

      {/* Data Tab */}
      {activeTab === 'data' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700">
            <h2 className="text-xl font-semibold text-gray-100 mb-4">Enter Time Series Data</h2>
            <p className="text-gray-400 text-sm mb-4">Enter values in chronological order. Use arrow keys to navigate.</p>

            <div className="max-h-80 overflow-y-auto">
              <ExcelTable
                data={tableData}
                columns={columns}
                onChange={setTableData}
                minRows={20}
                maxRows={500}
                allowAddRows={true}
                allowDeleteRows={true}
              />
            </div>

            <div className="flex gap-3 mt-4">
              <button
                onClick={parseData}
                disabled={!tableData.some(row => row[0] !== '' && row[0] !== null)}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg font-semibold disabled:opacity-50"
              >
                <Play className="w-5 h-5" /> Load Data
              </button>
            </div>

            <div className="mt-4">
              <p className="text-gray-400 text-sm mb-2">Or load sample data:</p>
              <div className="flex gap-2">
                <button onClick={() => loadSampleData('trend')} className="px-3 py-1 bg-slate-700 hover:bg-slate-600 text-gray-200 rounded text-sm">
                  Trend
                </button>
                <button onClick={() => loadSampleData('seasonal')} className="px-3 py-1 bg-slate-700 hover:bg-slate-600 text-gray-200 rounded text-sm">
                  Seasonal
                </button>
                <button onClick={() => loadSampleData('random')} className="px-3 py-1 bg-slate-700 hover:bg-slate-600 text-gray-200 rounded text-sm">
                  Random
                </button>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700">
            <h2 className="text-xl font-semibold text-gray-100 mb-4">Preview</h2>
            {data ? (
              <>
                {renderTimeSeriesPlot(data.values, `Time Series (n=${data.values.length})`)}
                <p className="text-gray-400 text-sm mt-2">
                  {data.values.length} observations loaded
                </p>
              </>
            ) : (
              <div className="h-64 flex items-center justify-center text-gray-500">
                <div className="text-center">
                  <BarChart3 className="w-12 h-12 mx-auto mb-2 opacity-50" />
                  <p>Enter data to see preview</p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Summary Tab */}
      {activeTab === 'summary' && summary && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
              <div className="text-gray-400 text-sm">Observations</div>
              <div className="text-2xl font-bold text-indigo-400">{summary.basic_statistics.n}</div>
            </div>
            <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
              <div className="text-gray-400 text-sm">Mean</div>
              <div className="text-2xl font-bold text-indigo-400">{summary.basic_statistics.mean.toFixed(2)}</div>
            </div>
            <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
              <div className="text-gray-400 text-sm">Std Dev</div>
              <div className="text-2xl font-bold text-indigo-400">{summary.basic_statistics.std.toFixed(2)}</div>
            </div>
            <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
              <div className="text-gray-400 text-sm">Range</div>
              <div className="text-2xl font-bold text-indigo-400">
                {(summary.basic_statistics.max - summary.basic_statistics.min).toFixed(2)}
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className={`rounded-xl p-4 border ${summary.trend_analysis.has_trend ? 'bg-yellow-900/30 border-yellow-700/50' : 'bg-slate-800/50 border-slate-700'}`}>
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp className="w-5 h-5 text-yellow-400" />
                <span className="font-semibold text-gray-100">Trend</span>
              </div>
              <div className={`text-lg ${summary.trend_analysis.has_trend ? 'text-yellow-300' : 'text-gray-400'}`}>
                {summary.trend_analysis.has_trend ? 'Significant trend detected' : 'No significant trend'}
              </div>
              <div className="text-sm text-gray-500 mt-1">
                Slope: {summary.trend_analysis.slope.toFixed(4)}, p={summary.trend_analysis.p_value.toFixed(4)}
              </div>
            </div>

            <div className={`rounded-xl p-4 border ${summary.stationarity.is_stationary ? 'bg-green-900/30 border-green-700/50' : 'bg-red-900/30 border-red-700/50'}`}>
              <div className="flex items-center gap-2 mb-2">
                <Activity className="w-5 h-5 text-green-400" />
                <span className="font-semibold text-gray-100">Stationarity</span>
              </div>
              <div className={`text-lg ${summary.stationarity.is_stationary ? 'text-green-300' : 'text-red-300'}`}>
                {summary.stationarity.is_stationary ? 'Stationary' : 'Non-stationary'}
              </div>
              <div className="text-sm text-gray-500 mt-1">
                ADF p-value: {summary.stationarity.adf_pvalue?.toFixed(4) || 'N/A'}
              </div>
            </div>

            <div className={`rounded-xl p-4 border ${summary.seasonality.has_seasonality ? 'bg-purple-900/30 border-purple-700/50' : 'bg-slate-800/50 border-slate-700'}`}>
              <div className="flex items-center gap-2 mb-2">
                <Calendar className="w-5 h-5 text-purple-400" />
                <span className="font-semibold text-gray-100">Seasonality</span>
              </div>
              <div className={`text-lg ${summary.seasonality.has_seasonality ? 'text-purple-300' : 'text-gray-400'}`}>
                {summary.seasonality.has_seasonality ? `Period: ${summary.seasonality.detected_period}` : 'No seasonality'}
              </div>
            </div>
          </div>

          <div className="bg-blue-900/30 border border-blue-700/50 rounded-lg p-4">
            <h3 className="font-semibold text-blue-300 mb-2">Recommendations</h3>
            <ul className="text-blue-200 text-sm space-y-1">
              <li>• Differencing: {summary.recommendations.differencing}</li>
              <li>• Model: {summary.recommendations.seasonal_model}</li>
            </ul>
          </div>
        </div>
      )}

      {/* Decomposition Tab */}
      {activeTab === 'decomposition' && (
        <div className="space-y-6">
          <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700">
            <h3 className="text-lg font-semibold text-gray-100 mb-4">Decomposition Settings</h3>
            <div className="flex flex-wrap gap-4 items-end">
              <div>
                <label className="block text-gray-400 text-sm mb-1">Model</label>
                <select
                  value={decompModel}
                  onChange={(e) => setDecompModel(e.target.value)}
                  className="bg-slate-900 text-gray-100 px-3 py-2 rounded border border-slate-600"
                >
                  <option value="additive">Additive</option>
                  <option value="multiplicative">Multiplicative</option>
                </select>
              </div>
              <div>
                <label className="block text-gray-400 text-sm mb-1">Period (optional)</label>
                <input
                  type="number"
                  value={decompPeriod}
                  onChange={(e) => setDecompPeriod(e.target.value)}
                  className="w-24 bg-slate-900 text-gray-100 px-3 py-2 rounded border border-slate-600"
                  placeholder="Auto"
                />
              </div>
              <button
                onClick={runDecomposition}
                disabled={!data || loading}
                className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg disabled:opacity-50"
              >
                {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Layers className="w-4 h-4" />}
                Decompose
              </button>
            </div>
          </div>

          {decomposition && (
            <>
              <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
                {renderDecompositionPlot()}
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
                  <div className="text-gray-400 text-sm">Period</div>
                  <div className="text-xl font-bold text-purple-400">{decomposition.period}</div>
                </div>
                <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
                  <div className="text-gray-400 text-sm">Trend Strength</div>
                  <div className="text-xl font-bold text-green-400">{(decomposition.statistics.trend_strength * 100).toFixed(1)}%</div>
                </div>
                <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
                  <div className="text-gray-400 text-sm">Seasonal Strength</div>
                  <div className="text-xl font-bold text-yellow-400">{(decomposition.statistics.seasonal_strength * 100).toFixed(1)}%</div>
                </div>
                <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
                  <div className="text-gray-400 text-sm">Model</div>
                  <div className="text-xl font-bold text-indigo-400 capitalize">{decomposition.model}</div>
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* ACF/PACF Tab */}
      {activeTab === 'acf' && (
        <div className="space-y-6">
          <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700">
            <button
              onClick={runAcfPacf}
              disabled={!data || loading}
              className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg disabled:opacity-50"
            >
              {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <BarChart3 className="w-4 h-4" />}
              Compute ACF/PACF
            </button>
          </div>

          {acfPacf && (
            <>
              {renderAcfPacfPlot()}
              <div className="bg-blue-900/30 border border-blue-700/50 rounded-lg p-4">
                <h3 className="font-semibold text-blue-300 mb-2">Suggested ARIMA Orders</h3>
                <div className="flex gap-6 text-blue-200">
                  <div>
                    <span className="text-blue-400">AR (p):</span> {acfPacf.suggestions.ar_order_p}
                  </div>
                  <div>
                    <span className="text-blue-400">MA (q):</span> {acfPacf.suggestions.ma_order_q}
                  </div>
                </div>
                <div className="text-sm text-blue-300 mt-2">
                  ACF pattern: {acfPacf.suggestions.interpretation.acf_pattern} |
                  PACF pattern: {acfPacf.suggestions.interpretation.pacf_pattern}
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* Stationarity Tab */}
      {activeTab === 'stationarity' && (
        <div className="space-y-6">
          <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700">
            <button
              onClick={runStationarity}
              disabled={!data || loading}
              className="flex items-center gap-2 px-4 py-2 bg-yellow-600 hover:bg-yellow-700 text-white rounded-lg disabled:opacity-50"
            >
              {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Target className="w-4 h-4" />}
              Test Stationarity
            </button>
          </div>

          {stationarity && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {stationarity.adf && (
                <div className={`rounded-xl p-6 border ${stationarity.adf.is_stationary ? 'bg-green-900/30 border-green-700/50' : 'bg-red-900/30 border-red-700/50'}`}>
                  <h3 className="text-lg font-semibold text-gray-100 mb-4">ADF Test</h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Test Statistic:</span>
                      <span className="text-gray-100">{stationarity.adf.test_statistic.toFixed(4)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">P-Value:</span>
                      <span className="text-gray-100">{stationarity.adf.p_value.toFixed(4)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Lags Used:</span>
                      <span className="text-gray-100">{stationarity.adf.lags_used}</span>
                    </div>
                    <div className="mt-4 pt-4 border-t border-slate-600">
                      <div className={`font-semibold ${stationarity.adf.is_stationary ? 'text-green-400' : 'text-red-400'}`}>
                        {stationarity.adf.interpretation}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {stationarity.kpss && !stationarity.kpss.error && (
                <div className={`rounded-xl p-6 border ${stationarity.kpss.is_stationary ? 'bg-green-900/30 border-green-700/50' : 'bg-red-900/30 border-red-700/50'}`}>
                  <h3 className="text-lg font-semibold text-gray-100 mb-4">KPSS Test</h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Test Statistic:</span>
                      <span className="text-gray-100">{stationarity.kpss.test_statistic.toFixed(4)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">P-Value:</span>
                      <span className="text-gray-100">{stationarity.kpss.p_value.toFixed(4)}</span>
                    </div>
                    <div className="mt-4 pt-4 border-t border-slate-600">
                      <div className={`font-semibold ${stationarity.kpss.is_stationary ? 'text-green-400' : 'text-red-400'}`}>
                        {stationarity.kpss.interpretation}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {stationarity.conclusion && (
                <div className="md:col-span-2 bg-indigo-900/30 border border-indigo-700/50 rounded-xl p-6">
                  <h3 className="text-lg font-semibold text-indigo-300 mb-2">Conclusion</h3>
                  <p className="text-indigo-200">{stationarity.conclusion}</p>
                  <p className="text-indigo-300 mt-2 text-sm">{stationarity.recommendation}</p>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ARIMA Tab */}
      {activeTab === 'arima' && (
        <div className="space-y-6">
          <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700">
            <h3 className="text-lg font-semibold text-gray-100 mb-4">ARIMA Settings</h3>
            <div className="flex flex-wrap gap-4 items-end">
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={autoArima}
                  onChange={(e) => setAutoArima(e.target.checked)}
                  id="autoArima"
                />
                <label htmlFor="autoArima" className="text-gray-300">Auto-select order</label>
              </div>
              {!autoArima && (
                <>
                  <div>
                    <label className="block text-gray-400 text-sm mb-1">p (AR)</label>
                    <input
                      type="number"
                      value={arimaOrder.p}
                      onChange={(e) => setArimaOrder({ ...arimaOrder, p: e.target.value })}
                      className="w-16 bg-slate-900 text-gray-100 px-3 py-2 rounded border border-slate-600"
                      min="0"
                    />
                  </div>
                  <div>
                    <label className="block text-gray-400 text-sm mb-1">d (Diff)</label>
                    <input
                      type="number"
                      value={arimaOrder.d}
                      onChange={(e) => setArimaOrder({ ...arimaOrder, d: e.target.value })}
                      className="w-16 bg-slate-900 text-gray-100 px-3 py-2 rounded border border-slate-600"
                      min="0"
                    />
                  </div>
                  <div>
                    <label className="block text-gray-400 text-sm mb-1">q (MA)</label>
                    <input
                      type="number"
                      value={arimaOrder.q}
                      onChange={(e) => setArimaOrder({ ...arimaOrder, q: e.target.value })}
                      className="w-16 bg-slate-900 text-gray-100 px-3 py-2 rounded border border-slate-600"
                      min="0"
                    />
                  </div>
                </>
              )}
              <button
                onClick={runArima}
                disabled={!data || loading}
                className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg disabled:opacity-50"
              >
                {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Activity className="w-4 h-4" />}
                Fit ARIMA
              </button>
            </div>
          </div>

          {arimaResult && (
            <>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
                  <div className="text-gray-400 text-sm">Order</div>
                  <div className="text-xl font-bold text-indigo-400">
                    ({arimaResult.order.join(', ')})
                  </div>
                </div>
                <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
                  <div className="text-gray-400 text-sm">AIC</div>
                  <div className="text-xl font-bold text-indigo-400">{arimaResult.model_summary.aic.toFixed(2)}</div>
                </div>
                <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
                  <div className="text-gray-400 text-sm">BIC</div>
                  <div className="text-xl font-bold text-indigo-400">{arimaResult.model_summary.bic.toFixed(2)}</div>
                </div>
                <div className={`rounded-xl p-4 border ${arimaResult.diagnostics.residuals_uncorrelated ? 'bg-green-900/30 border-green-700/50' : 'bg-yellow-900/30 border-yellow-700/50'}`}>
                  <div className="text-gray-400 text-sm">Residuals</div>
                  <div className={`text-lg font-bold ${arimaResult.diagnostics.residuals_uncorrelated ? 'text-green-400' : 'text-yellow-400'}`}>
                    {arimaResult.diagnostics.residuals_uncorrelated ? 'OK' : 'Correlated'}
                  </div>
                </div>
              </div>

              <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
                <h4 className="text-gray-300 mb-2">Fitted vs Original</h4>
                <Plot
                  data={[
                    {
                      type: 'scatter',
                      mode: 'lines',
                      y: arimaResult.original,
                      name: 'Original',
                      line: { color: '#3b82f6', width: 2 }
                    },
                    {
                      type: 'scatter',
                      mode: 'lines',
                      y: arimaResult.fitted_values,
                      name: 'Fitted',
                      line: { color: '#22c55e', width: 2 }
                    }
                  ]}
                  layout={{
                    xaxis: { title: 'Time', gridcolor: '#475569' },
                    yaxis: { title: 'Value', gridcolor: '#475569' },
                    paper_bgcolor: '#1e293b',
                    plot_bgcolor: '#0f172a',
                    font: { color: '#e2e8f0' },
                    height: 300,
                    margin: { l: 60, r: 20, t: 20, b: 50 },
                    legend: { x: 0, y: 1 }
                  }}
                  config={{ responsive: true }}
                  style={{ width: '100%' }}
                />
              </div>
            </>
          )}
        </div>
      )}

      {/* Forecast Tab */}
      {activeTab === 'forecast' && (
        <div className="space-y-6">
          <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700">
            <h3 className="text-lg font-semibold text-gray-100 mb-4">Forecast Settings</h3>
            <div className="flex flex-wrap gap-4 items-end">
              <div>
                <label className="block text-gray-400 text-sm mb-1">Horizon</label>
                <input
                  type="number"
                  value={forecastHorizon}
                  onChange={(e) => setForecastHorizon(parseInt(e.target.value) || 1)}
                  className="w-24 bg-slate-900 text-gray-100 px-3 py-2 rounded border border-slate-600"
                  min="1"
                />
              </div>
              <div>
                <label className="block text-gray-400 text-sm mb-1">Confidence</label>
                <select
                  value={confidenceLevel}
                  onChange={(e) => setConfidenceLevel(parseFloat(e.target.value))}
                  className="bg-slate-900 text-gray-100 px-3 py-2 rounded border border-slate-600"
                >
                  <option value="0.90">90%</option>
                  <option value="0.95">95%</option>
                  <option value="0.99">99%</option>
                </select>
              </div>
              <button
                onClick={runForecast}
                disabled={!data || loading}
                className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg disabled:opacity-50"
              >
                {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <TrendingUp className="w-4 h-4" />}
                Generate Forecast
              </button>
            </div>
          </div>

          {forecast && (
            <>
              <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
                {renderForecastPlot()}
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
                  <div className="text-gray-400 text-sm">Horizon</div>
                  <div className="text-xl font-bold text-green-400">{forecast.summary.horizon} periods</div>
                </div>
                <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
                  <div className="text-gray-400 text-sm">Forecast Mean</div>
                  <div className="text-xl font-bold text-green-400">{forecast.summary.forecast_mean.toFixed(2)}</div>
                </div>
                <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
                  <div className="text-gray-400 text-sm">Model Order</div>
                  <div className="text-xl font-bold text-green-400">({forecast.model.order.join(', ')})</div>
                </div>
                <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
                  <div className="text-gray-400 text-sm">Trend</div>
                  <div className="text-xl font-bold text-green-400 capitalize">{forecast.summary.trend}</div>
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* No Data Message */}
      {!data && activeTab !== 'data' && (
        <div className="bg-slate-800/50 rounded-xl p-12 border border-slate-700 text-center">
          <BarChart3 className="w-16 h-16 text-slate-600 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-gray-400 mb-2">No Data Loaded</h3>
          <p className="text-gray-500">Go to the Data tab to enter your time series</p>
        </div>
      )}
    </div>
  )
}

export default TimeSeriesAnalysis
