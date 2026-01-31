import { useState, useEffect } from 'react'
import axios from 'axios'
import {
  Activity,
  Clock,
  TrendingUp,
  Target,
  Calculator,
  FileDown,
  Plus,
  Trash2,
  Play,
  AlertCircle,
  CheckCircle,
  Info,
  BarChart3,
  Zap
} from 'lucide-react'
import SurvivalCurvePlot from '../components/SurvivalCurvePlot'
import WeibullPlot from '../components/WeibullPlot'
import HazardRatioForest from '../components/HazardRatioForest'
import LifeDistributionResults from '../components/LifeDistributionResults'
import FileUploadZone from '../components/FileUploadZone'
import { useSession } from '../contexts/SessionContext'
import { getPlotlyConfig } from '../utils/plotlyConfig'

const API_URL = import.meta.env.VITE_API_URL || ''

const ReliabilityAnalysis = () => {
  // Tab state
  const [activeTab, setActiveTab] = useState('data')

  // Session management
  const { saveCurrentSession, currentSession } = useSession()

  // Data input state
  const [tableData, setTableData] = useState(
    Array(15).fill(null).map(() => ['', '', '']) // Time, Event, Group
  )
  const [hasGroups, setHasGroups] = useState(false)
  const [timeColumn, setTimeColumn] = useState('Time')
  const [eventColumn, setEventColumn] = useState('Event')
  const [groupColumn, setGroupColumn] = useState('Group')

  // Covariate data for Cox regression
  const [covariates, setCovariates] = useState([])
  const [covariateData, setCovariateData] = useState({})

  // Life Distribution state
  const [lifeDistResult, setLifeDistResult] = useState(null)
  const [lifeDistLoading, setLifeDistLoading] = useState(false)
  const [lifeDistError, setLifeDistError] = useState(null)
  const [selectedDistributions, setSelectedDistributions] = useState(['weibull', 'lognormal', 'exponential', 'loglogistic'])
  const [confidenceLevel, setConfidenceLevel] = useState(0.95)
  const [reliabilityTimePoints, setReliabilityTimePoints] = useState('')

  // Kaplan-Meier state
  const [kmResult, setKmResult] = useState(null)
  const [kmLoading, setKmLoading] = useState(false)
  const [kmError, setKmError] = useState(null)
  const [showKMConfidence, setShowKMConfidence] = useState(true)

  // Cox PH state
  const [coxResult, setCoxResult] = useState(null)
  const [coxLoading, setCoxLoading] = useState(false)
  const [coxError, setCoxError] = useState(null)
  const [coxPenalizer, setCoxPenalizer] = useState(0.0)

  // ALT state
  const [altResult, setAltResult] = useState(null)
  const [altLoading, setAltLoading] = useState(false)
  const [altError, setAltError] = useState(null)
  const [stressVariable, setStressVariable] = useState('Stress')
  const [stressValues, setStressValues] = useState([])
  const [altModelType, setAltModelType] = useState('weibull')
  const [useStress, setUseStress] = useState('')

  // Test Planning state
  const [testPlanResult, setTestPlanResult] = useState(null)
  const [testPlanLoading, setTestPlanLoading] = useState(false)
  const [testPlanError, setTestPlanError] = useState(null)
  const [testPlanParams, setTestPlanParams] = useState({
    target_reliability: 0.95,
    test_time: 1000,
    confidence_level: 0.95,
    test_type: 'demonstration',
    allowable_failures: 0,
    distribution: 'exponential',
    shape_parameter: 2.0,
    comparison_reliability: 0.90,
    power: 0.8
  })

  // Tab configuration
  const tabs = [
    { id: 'data', label: 'Data', icon: BarChart3 },
    { id: 'life-dist', label: 'Life Distributions', icon: TrendingUp },
    { id: 'survival', label: 'Survival Analysis', icon: Activity },
    { id: 'cox', label: 'Cox Regression', icon: Target },
    { id: 'alt', label: 'Accelerated Life', icon: Zap },
    { id: 'planning', label: 'Test Planning', icon: Calculator }
  ]

  // Extract data from table
  const extractData = () => {
    const times = []
    const events = []
    const groups = hasGroups ? [] : null

    tableData.forEach(row => {
      const time = parseFloat(row[0])
      const event = parseInt(row[1])

      if (!isNaN(time) && !isNaN(event)) {
        times.push(time)
        events.push(event)
        if (hasGroups && row[2]) {
          groups.push(row[2])
        }
      }
    })

    return { times, events, groups }
  }

  // Handle table cell change
  const handleCellChange = (rowIndex, colIndex, value) => {
    const newData = [...tableData]
    newData[rowIndex][colIndex] = value
    setTableData(newData)
  }

  // Add row to table
  const addRow = () => {
    setTableData([...tableData, ['', '', '']])
  }

  // Remove row from table
  const removeRow = (index) => {
    if (tableData.length > 1) {
      setTableData(tableData.filter((_, i) => i !== index))
    }
  }

  // Handle file upload
  const handleFileData = (data) => {
    if (data && data.length > 0) {
      const headers = Object.keys(data[0])
      const newTableData = data.map(row => {
        const timeVal = row[headers.find(h => h.toLowerCase().includes('time'))] || row[headers[0]] || ''
        const eventVal = row[headers.find(h => h.toLowerCase().includes('event') || h.toLowerCase().includes('status'))] || row[headers[1]] || ''
        const groupVal = row[headers.find(h => h.toLowerCase().includes('group'))] || row[headers[2]] || ''
        return [String(timeVal), String(eventVal), groupVal]
      })
      setTableData(newTableData)

      // Check if there are group values
      const hasGroupData = newTableData.some(row => row[2] && row[2].trim() !== '')
      setHasGroups(hasGroupData)
    }
  }

  // Handle paste from clipboard
  const handlePaste = (e) => {
    e.preventDefault()
    const text = e.clipboardData.getData('text')
    const rows = text.split('\n').filter(r => r.trim())
    const parsedData = rows.map(row => {
      const cols = row.split(/[\t,]/)
      return [cols[0] || '', cols[1] || '', cols[2] || '']
    })

    if (parsedData.length > 0) {
      setTableData(parsedData)
      const hasGroupData = parsedData.some(row => row[2] && row[2].trim() !== '')
      setHasGroups(hasGroupData)
    }
  }

  // Life Distribution Analysis
  const runLifeDistribution = async () => {
    const { times, events } = extractData()

    if (times.length < 2) {
      setLifeDistError('At least 2 observations required')
      return
    }

    setLifeDistLoading(true)
    setLifeDistError(null)

    try {
      const timePoints = reliabilityTimePoints
        .split(',')
        .map(t => parseFloat(t.trim()))
        .filter(t => !isNaN(t) && t > 0)

      const response = await axios.post(`${API_URL}/api/reliability/life-distribution`, {
        times,
        events,
        distributions: selectedDistributions,
        confidence_level: confidenceLevel,
        time_points: timePoints.length > 0 ? timePoints : null
      })
      setLifeDistResult(response.data)
    } catch (err) {
      setLifeDistError(err.response?.data?.detail || err.message)
    } finally {
      setLifeDistLoading(false)
    }
  }

  // Kaplan-Meier Analysis
  const runKaplanMeier = async () => {
    const { times, events, groups } = extractData()

    if (times.length < 2) {
      setKmError('At least 2 observations required')
      return
    }

    setKmLoading(true)
    setKmError(null)

    try {
      const timePoints = reliabilityTimePoints
        .split(',')
        .map(t => parseFloat(t.trim()))
        .filter(t => !isNaN(t) && t > 0)

      const response = await axios.post(`${API_URL}/api/reliability/kaplan-meier`, {
        times,
        events,
        groups: hasGroups ? groups : null,
        confidence_level: confidenceLevel,
        time_points: timePoints.length > 0 ? timePoints : null
      })
      setKmResult(response.data)
    } catch (err) {
      setKmError(err.response?.data?.detail || err.message)
    } finally {
      setKmLoading(false)
    }
  }

  // Cox Proportional Hazards Analysis
  const runCoxPH = async () => {
    const { times, events } = extractData()

    if (times.length < 2) {
      setCoxError('At least 2 observations required')
      return
    }

    if (Object.keys(covariateData).length === 0) {
      setCoxError('At least one covariate is required for Cox regression')
      return
    }

    setCoxLoading(true)
    setCoxError(null)

    try {
      const response = await axios.post(`${API_URL}/api/reliability/cox-ph`, {
        times,
        events,
        covariates: covariateData,
        confidence_level: confidenceLevel,
        penalizer: coxPenalizer
      })
      setCoxResult(response.data)
    } catch (err) {
      setCoxError(err.response?.data?.detail || err.message)
    } finally {
      setCoxLoading(false)
    }
  }

  // Accelerated Life Testing Analysis
  const runALT = async () => {
    const { times, events } = extractData()

    if (times.length < 2) {
      setAltError('At least 2 observations required')
      return
    }

    if (stressValues.length !== times.length) {
      setAltError('Stress values must match number of observations')
      return
    }

    setAltLoading(true)
    setAltError(null)

    try {
      const response = await axios.post(`${API_URL}/api/reliability/alt`, {
        times,
        events,
        stress_variable: stressVariable,
        stress_values: stressValues,
        model_type: altModelType,
        use_stress: useStress ? parseFloat(useStress) : null,
        confidence_level: confidenceLevel
      })
      setAltResult(response.data)
    } catch (err) {
      setAltError(err.response?.data?.detail || err.message)
    } finally {
      setAltLoading(false)
    }
  }

  // Test Planning
  const runTestPlanning = async () => {
    setTestPlanLoading(true)
    setTestPlanError(null)

    try {
      const response = await axios.post(`${API_URL}/api/reliability/test-planning`, testPlanParams)
      setTestPlanResult(response.data)
    } catch (err) {
      setTestPlanError(err.response?.data?.detail || err.message)
    } finally {
      setTestPlanLoading(false)
    }
  }

  // Add covariate column
  const addCovariate = () => {
    const newName = `Covariate_${covariates.length + 1}`
    setCovariates([...covariates, newName])
    setCovariateData({
      ...covariateData,
      [newName]: Array(tableData.length).fill('')
    })
  }

  // Remove covariate
  const removeCovariate = (name) => {
    setCovariates(covariates.filter(c => c !== name))
    const newData = { ...covariateData }
    delete newData[name]
    setCovariateData(newData)
  }

  // Update covariate value
  const updateCovariateValue = (name, index, value) => {
    const newData = { ...covariateData }
    if (!newData[name]) {
      newData[name] = Array(tableData.length).fill('')
    }
    newData[name][index] = value
    setCovariateData(newData)
  }

  // Save session
  const handleSaveSession = async () => {
    await saveCurrentSession('Reliability Analysis', {
      analysis_type: 'reliability',
      data: {
        tableData,
        hasGroups,
        covariates,
        covariateData,
        stressValues
      },
      results: {
        lifeDistResult,
        kmResult,
        coxResult,
        altResult,
        testPlanResult
      },
      settings: {
        confidenceLevel,
        selectedDistributions,
        altModelType,
        testPlanParams
      }
    })
  }

  // Load sample data
  const loadSampleData = () => {
    const sampleData = [
      ['45', '1', 'A'],
      ['68', '1', 'A'],
      ['92', '0', 'A'],
      ['115', '1', 'A'],
      ['135', '1', 'A'],
      ['156', '0', 'A'],
      ['38', '1', 'B'],
      ['52', '1', 'B'],
      ['78', '1', 'B'],
      ['95', '0', 'B'],
      ['112', '1', 'B'],
      ['142', '1', 'B']
    ]
    setTableData(sampleData)
    setHasGroups(true)
  }

  return (
    <div className="min-h-screen bg-slate-900 text-gray-100">
      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-gray-100 flex items-center gap-3">
              <Activity className="w-8 h-8 text-blue-400" />
              Reliability & Survival Analysis
            </h1>
            <p className="text-gray-400 mt-2">
              Life distribution fitting, Kaplan-Meier survival curves, Cox regression, and accelerated life testing
            </p>
          </div>
          <button
            onClick={handleSaveSession}
            className="flex items-center gap-2 bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg transition-colors"
          >
            <FileDown className="w-4 h-4" />
            Save Session
          </button>
        </div>

        {/* Tab Navigation */}
        <div className="flex gap-2 mb-6 border-b border-slate-700 pb-2 overflow-x-auto">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-t-lg transition-colors whitespace-nowrap ${
                activeTab === tab.id
                  ? 'bg-blue-600 text-white'
                  : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <div className="space-y-6">
          {/* Data Tab */}
          {activeTab === 'data' && (
            <div className="space-y-6">
              <div className="bg-slate-800 rounded-lg p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-semibold text-gray-100">Time-to-Event Data</h2>
                  <div className="flex gap-2">
                    <button
                      onClick={loadSampleData}
                      className="text-sm bg-slate-700 hover:bg-slate-600 text-gray-300 px-3 py-1.5 rounded"
                    >
                      Load Sample Data
                    </button>
                    <button
                      onClick={addRow}
                      className="flex items-center gap-1 bg-blue-600 hover:bg-blue-700 text-white px-3 py-1.5 rounded text-sm"
                    >
                      <Plus className="w-4 h-4" />
                      Add Row
                    </button>
                  </div>
                </div>

                <div className="mb-4">
                  <FileUploadZone onDataLoaded={handleFileData} />
                </div>

                <div className="flex items-center gap-4 mb-4">
                  <label className="flex items-center gap-2 text-sm text-gray-300">
                    <input
                      type="checkbox"
                      checked={hasGroups}
                      onChange={(e) => setHasGroups(e.target.checked)}
                      className="rounded bg-slate-700 border-slate-600"
                    />
                    Include Groups (for stratified analysis)
                  </label>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-gray-400 border-b border-slate-600">
                        <th className="text-left py-2 px-3 w-16">#</th>
                        <th className="text-left py-2 px-3">Time</th>
                        <th className="text-left py-2 px-3">Event (1=failure, 0=censored)</th>
                        {hasGroups && <th className="text-left py-2 px-3">Group</th>}
                        <th className="w-12"></th>
                      </tr>
                    </thead>
                    <tbody>
                      {tableData.map((row, rowIndex) => (
                        <tr key={rowIndex} className="border-b border-slate-700/50">
                          <td className="py-1 px-3 text-gray-500">{rowIndex + 1}</td>
                          <td className="py-1 px-3">
                            <input
                              type="text"
                              value={row[0]}
                              onChange={(e) => handleCellChange(rowIndex, 0, e.target.value)}
                              onPaste={rowIndex === 0 ? handlePaste : undefined}
                              className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-gray-100"
                              placeholder="Time value"
                            />
                          </td>
                          <td className="py-1 px-3">
                            <input
                              type="text"
                              value={row[1]}
                              onChange={(e) => handleCellChange(rowIndex, 1, e.target.value)}
                              className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-gray-100"
                              placeholder="0 or 1"
                            />
                          </td>
                          {hasGroups && (
                            <td className="py-1 px-3">
                              <input
                                type="text"
                                value={row[2]}
                                onChange={(e) => handleCellChange(rowIndex, 2, e.target.value)}
                                className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-gray-100"
                                placeholder="Group name"
                              />
                            </td>
                          )}
                          <td className="py-1 px-3">
                            <button
                              onClick={() => removeRow(rowIndex)}
                              className="text-red-400 hover:text-red-300"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="mt-4 text-sm text-gray-400">
                  <p>Tip: You can paste data directly from Excel or CSV (Tab or comma separated)</p>
                </div>
              </div>

              {/* Data Summary */}
              {tableData.some(row => row[0] && row[1]) && (
                <div className="bg-slate-800 rounded-lg p-6">
                  <h3 className="text-lg font-semibold text-gray-100 mb-4">Data Summary</h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {(() => {
                      const { times, events, groups } = extractData()
                      const nEvents = events.filter(e => e === 1).length
                      const nCensored = events.filter(e => e === 0).length
                      return (
                        <>
                          <div className="bg-slate-700/50 rounded p-4">
                            <div className="text-2xl font-bold text-blue-400">{times.length}</div>
                            <div className="text-sm text-gray-400">Total Observations</div>
                          </div>
                          <div className="bg-slate-700/50 rounded p-4">
                            <div className="text-2xl font-bold text-red-400">{nEvents}</div>
                            <div className="text-sm text-gray-400">Events (Failures)</div>
                          </div>
                          <div className="bg-slate-700/50 rounded p-4">
                            <div className="text-2xl font-bold text-green-400">{nCensored}</div>
                            <div className="text-sm text-gray-400">Censored</div>
                          </div>
                          <div className="bg-slate-700/50 rounded p-4">
                            <div className="text-2xl font-bold text-yellow-400">
                              {((nCensored / times.length) * 100).toFixed(1)}%
                            </div>
                            <div className="text-sm text-gray-400">Censoring Rate</div>
                          </div>
                        </>
                      )
                    })()}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Life Distributions Tab */}
          {activeTab === 'life-dist' && (
            <div className="space-y-6">
              <div className="bg-slate-800 rounded-lg p-6">
                <h2 className="text-xl font-semibold text-gray-100 mb-4">Life Distribution Fitting</h2>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                  <div>
                    <label className="block text-sm text-gray-400 mb-2">Distributions to Fit</label>
                    <div className="space-y-2">
                      {['weibull', 'lognormal', 'exponential', 'loglogistic'].map(dist => (
                        <label key={dist} className="flex items-center gap-2 text-gray-300">
                          <input
                            type="checkbox"
                            checked={selectedDistributions.includes(dist)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setSelectedDistributions([...selectedDistributions, dist])
                              } else {
                                setSelectedDistributions(selectedDistributions.filter(d => d !== dist))
                              }
                            }}
                            className="rounded bg-slate-700 border-slate-600"
                          />
                          {dist.charAt(0).toUpperCase() + dist.slice(1)}
                        </label>
                      ))}
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm text-gray-400 mb-2">Confidence Level</label>
                      <select
                        value={confidenceLevel}
                        onChange={(e) => setConfidenceLevel(parseFloat(e.target.value))}
                        className="w-full bg-slate-900 border border-slate-600 rounded px-3 py-2 text-gray-100"
                      >
                        <option value={0.90}>90%</option>
                        <option value={0.95}>95%</option>
                        <option value={0.99}>99%</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-sm text-gray-400 mb-2">
                        Reliability Time Points (comma separated)
                      </label>
                      <input
                        type="text"
                        value={reliabilityTimePoints}
                        onChange={(e) => setReliabilityTimePoints(e.target.value)}
                        className="w-full bg-slate-900 border border-slate-600 rounded px-3 py-2 text-gray-100"
                        placeholder="e.g., 100, 500, 1000"
                      />
                    </div>
                  </div>
                </div>

                <button
                  onClick={runLifeDistribution}
                  disabled={lifeDistLoading}
                  className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 text-white px-4 py-2 rounded-lg transition-colors"
                >
                  {lifeDistLoading ? (
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  ) : (
                    <Play className="w-4 h-4" />
                  )}
                  Fit Distributions
                </button>

                {lifeDistError && (
                  <div className="mt-4 p-4 bg-red-900/50 border border-red-700 rounded-lg flex items-center gap-2 text-red-300">
                    <AlertCircle className="w-5 h-5" />
                    {lifeDistError}
                  </div>
                )}
              </div>

              {/* Life Distribution Results */}
              {lifeDistResult && (
                <div className="space-y-6">
                  {/* Basic Metrics */}
                  <div className="bg-slate-800 rounded-lg p-6">
                    <h3 className="text-lg font-semibold text-gray-100 mb-4">Data Metrics</h3>
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                      <div className="bg-slate-700/50 rounded p-3">
                        <div className="text-lg font-mono text-gray-100">
                          {lifeDistResult.basic_metrics?.n_observations}
                        </div>
                        <div className="text-xs text-gray-400">Observations</div>
                      </div>
                      <div className="bg-slate-700/50 rounded p-3">
                        <div className="text-lg font-mono text-gray-100">
                          {lifeDistResult.basic_metrics?.n_events}
                        </div>
                        <div className="text-xs text-gray-400">Events</div>
                      </div>
                      <div className="bg-slate-700/50 rounded p-3">
                        <div className="text-lg font-mono text-gray-100">
                          {lifeDistResult.basic_metrics?.n_censored}
                        </div>
                        <div className="text-xs text-gray-400">Censored</div>
                      </div>
                      <div className="bg-slate-700/50 rounded p-3">
                        <div className="text-lg font-mono text-gray-100">
                          {lifeDistResult.basic_metrics?.censoring_rate?.toFixed(1)}%
                        </div>
                        <div className="text-xs text-gray-400">Censoring Rate</div>
                      </div>
                      <div className="bg-slate-700/50 rounded p-3">
                        <div className="text-lg font-mono text-gray-100">
                          {lifeDistResult.basic_metrics?.time_range?.median?.toFixed(2)}
                        </div>
                        <div className="text-xs text-gray-400">Median Time</div>
                      </div>
                    </div>
                  </div>

                  {/* Distribution Comparison and Curves */}
                  <LifeDistributionResults
                    results={lifeDistResult.distributions}
                    comparison={lifeDistResult.comparison}
                    bestDistribution={lifeDistResult.best_distribution}
                  />

                  {/* Weibull Plot */}
                  {lifeDistResult.distributions?.weibull?.probability_plot && (
                    <div className="bg-slate-800 rounded-lg p-6">
                      <WeibullPlot
                        data={lifeDistResult.distributions.weibull}
                        title="Weibull Probability Plot"
                      />
                    </div>
                  )}

                  {/* Recommendation */}
                  {lifeDistResult.recommendation && (
                    <div className="bg-green-900/30 border border-green-700 rounded-lg p-4 flex items-start gap-3">
                      <CheckCircle className="w-5 h-5 text-green-400 mt-0.5" />
                      <div>
                        <div className="font-medium text-green-300">Recommendation</div>
                        <div className="text-green-200">{lifeDistResult.recommendation}</div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Survival Analysis (Kaplan-Meier) Tab */}
          {activeTab === 'survival' && (
            <div className="space-y-6">
              <div className="bg-slate-800 rounded-lg p-6">
                <h2 className="text-xl font-semibold text-gray-100 mb-4">Kaplan-Meier Survival Analysis</h2>

                <div className="flex items-center gap-6 mb-6">
                  <label className="flex items-center gap-2 text-gray-300">
                    <input
                      type="checkbox"
                      checked={showKMConfidence}
                      onChange={(e) => setShowKMConfidence(e.target.checked)}
                      className="rounded bg-slate-700 border-slate-600"
                    />
                    Show Confidence Intervals
                  </label>

                  <div className="flex items-center gap-2">
                    <label className="text-sm text-gray-400">Confidence Level:</label>
                    <select
                      value={confidenceLevel}
                      onChange={(e) => setConfidenceLevel(parseFloat(e.target.value))}
                      className="bg-slate-900 border border-slate-600 rounded px-2 py-1 text-gray-100"
                    >
                      <option value={0.90}>90%</option>
                      <option value={0.95}>95%</option>
                      <option value={0.99}>99%</option>
                    </select>
                  </div>
                </div>

                <button
                  onClick={runKaplanMeier}
                  disabled={kmLoading}
                  className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 text-white px-4 py-2 rounded-lg transition-colors"
                >
                  {kmLoading ? (
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  ) : (
                    <Play className="w-4 h-4" />
                  )}
                  Run Kaplan-Meier Analysis
                </button>

                {kmError && (
                  <div className="mt-4 p-4 bg-red-900/50 border border-red-700 rounded-lg flex items-center gap-2 text-red-300">
                    <AlertCircle className="w-5 h-5" />
                    {kmError}
                  </div>
                )}
              </div>

              {/* Kaplan-Meier Results */}
              {kmResult && (
                <div className="space-y-6">
                  {/* Survival Curve */}
                  <div className="bg-slate-800 rounded-lg p-6">
                    <SurvivalCurvePlot
                      data={kmResult.results}
                      title="Kaplan-Meier Survival Curve"
                      showConfidenceIntervals={showKMConfidence}
                    />
                  </div>

                  {/* Log-Rank Test Results */}
                  {kmResult.results?.log_rank_test && (
                    <div className="bg-slate-800 rounded-lg p-6">
                      <h3 className="text-lg font-semibold text-gray-100 mb-4">Log-Rank Test</h3>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="bg-slate-700/50 rounded p-3">
                          <div className="text-lg font-mono text-gray-100">
                            {kmResult.results.log_rank_test.test_statistic?.toFixed(3)}
                          </div>
                          <div className="text-xs text-gray-400">Test Statistic</div>
                        </div>
                        <div className="bg-slate-700/50 rounded p-3">
                          <div className="text-lg font-mono text-gray-100">
                            {kmResult.results.log_rank_test.p_value?.toFixed(4)}
                          </div>
                          <div className="text-xs text-gray-400">P-Value</div>
                        </div>
                        <div className="bg-slate-700/50 rounded p-3">
                          <div className="text-lg font-mono text-gray-100">
                            {kmResult.results.log_rank_test.degrees_of_freedom}
                          </div>
                          <div className="text-xs text-gray-400">Degrees of Freedom</div>
                        </div>
                        <div className={`rounded p-3 ${
                          kmResult.results.log_rank_test.significant
                            ? 'bg-green-900/50 border border-green-700'
                            : 'bg-slate-700/50'
                        }`}>
                          <div className="text-lg font-mono text-gray-100">
                            {kmResult.results.log_rank_test.significant ? 'Significant' : 'Not Significant'}
                          </div>
                          <div className="text-xs text-gray-400">Result</div>
                        </div>
                      </div>
                      <div className="mt-4 p-3 bg-slate-700/50 rounded text-gray-300">
                        {kmResult.results.log_rank_test.interpretation}
                      </div>
                    </div>
                  )}

                  {/* Group Statistics */}
                  {kmResult.results?.groups && (
                    <div className="bg-slate-800 rounded-lg p-6">
                      <h3 className="text-lg font-semibold text-gray-100 mb-4">Group Statistics</h3>
                      <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                          <thead>
                            <tr className="text-gray-400 border-b border-slate-600">
                              <th className="text-left py-2 px-3">Group</th>
                              <th className="text-right py-2 px-3">N</th>
                              <th className="text-right py-2 px-3">Events</th>
                              <th className="text-right py-2 px-3">Censored</th>
                              <th className="text-right py-2 px-3">Median Survival</th>
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(kmResult.results.groups).map(([name, data]) => (
                              <tr key={name} className="border-b border-slate-700/50">
                                <td className="py-2 px-3 font-medium text-gray-200">{name}</td>
                                <td className="py-2 px-3 text-right font-mono">{data.n_observations}</td>
                                <td className="py-2 px-3 text-right font-mono">{data.n_events}</td>
                                <td className="py-2 px-3 text-right font-mono">{data.n_censored}</td>
                                <td className="py-2 px-3 text-right font-mono">
                                  {data.median_survival_time?.toFixed(2) ?? '-'}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Cox Regression Tab */}
          {activeTab === 'cox' && (
            <div className="space-y-6">
              <div className="bg-slate-800 rounded-lg p-6">
                <h2 className="text-xl font-semibold text-gray-100 mb-4">Cox Proportional Hazards Regression</h2>

                <div className="mb-6">
                  <div className="flex items-center justify-between mb-3">
                    <label className="text-sm text-gray-400">Covariates</label>
                    <button
                      onClick={addCovariate}
                      className="flex items-center gap-1 bg-slate-700 hover:bg-slate-600 text-gray-300 px-3 py-1.5 rounded text-sm"
                    >
                      <Plus className="w-4 h-4" />
                      Add Covariate
                    </button>
                  </div>

                  {covariates.length === 0 ? (
                    <div className="text-center py-8 text-gray-400 bg-slate-700/50 rounded-lg">
                      <Info className="w-8 h-8 mx-auto mb-2" />
                      <p>Add covariates to analyze their effect on survival</p>
                    </div>
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="text-gray-400 border-b border-slate-600">
                            <th className="text-left py-2 px-3 w-16">#</th>
                            {covariates.map((cov, idx) => (
                              <th key={cov} className="text-left py-2 px-3">
                                <div className="flex items-center gap-2">
                                  <input
                                    type="text"
                                    value={cov}
                                    onChange={(e) => {
                                      const newCovariates = [...covariates]
                                      const oldName = newCovariates[idx]
                                      newCovariates[idx] = e.target.value
                                      setCovariates(newCovariates)

                                      // Rename in data
                                      const newData = { ...covariateData }
                                      newData[e.target.value] = newData[oldName]
                                      delete newData[oldName]
                                      setCovariateData(newData)
                                    }}
                                    className="bg-transparent border-b border-slate-500 text-gray-300 px-1 focus:outline-none focus:border-blue-400"
                                  />
                                  <button
                                    onClick={() => removeCovariate(cov)}
                                    className="text-red-400 hover:text-red-300"
                                  >
                                    <Trash2 className="w-3 h-3" />
                                  </button>
                                </div>
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {tableData.slice(0, Math.max(tableData.filter(r => r[0]).length, 5)).map((_, rowIndex) => (
                            <tr key={rowIndex} className="border-b border-slate-700/50">
                              <td className="py-1 px-3 text-gray-500">{rowIndex + 1}</td>
                              {covariates.map(cov => (
                                <td key={cov} className="py-1 px-3">
                                  <input
                                    type="text"
                                    value={covariateData[cov]?.[rowIndex] || ''}
                                    onChange={(e) => updateCovariateValue(cov, rowIndex, e.target.value)}
                                    className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-gray-100"
                                    placeholder="Value"
                                  />
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>

                <div className="flex items-center gap-4 mb-6">
                  <div>
                    <label className="block text-sm text-gray-400 mb-1">Penalizer (L2 regularization)</label>
                    <input
                      type="number"
                      value={coxPenalizer}
                      onChange={(e) => setCoxPenalizer(parseFloat(e.target.value) || 0)}
                      min="0"
                      step="0.01"
                      className="w-32 bg-slate-900 border border-slate-600 rounded px-3 py-2 text-gray-100"
                    />
                  </div>
                </div>

                <button
                  onClick={runCoxPH}
                  disabled={coxLoading || covariates.length === 0}
                  className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 text-white px-4 py-2 rounded-lg transition-colors"
                >
                  {coxLoading ? (
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  ) : (
                    <Play className="w-4 h-4" />
                  )}
                  Run Cox Regression
                </button>

                {coxError && (
                  <div className="mt-4 p-4 bg-red-900/50 border border-red-700 rounded-lg flex items-center gap-2 text-red-300">
                    <AlertCircle className="w-5 h-5" />
                    {coxError}
                  </div>
                )}
              </div>

              {/* Cox PH Results */}
              {coxResult && (
                <div className="space-y-6">
                  {/* Model Fit Statistics */}
                  <div className="bg-slate-800 rounded-lg p-6">
                    <h3 className="text-lg font-semibold text-gray-100 mb-4">Model Fit Statistics</h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="bg-slate-700/50 rounded p-3">
                        <div className="text-lg font-mono text-gray-100">
                          {coxResult.model_fit?.concordance_index?.toFixed(3)}
                        </div>
                        <div className="text-xs text-gray-400">Concordance Index</div>
                        <div className="text-xs text-blue-400 mt-1">
                          {coxResult.model_fit?.concordance_interpretation}
                        </div>
                      </div>
                      <div className="bg-slate-700/50 rounded p-3">
                        <div className="text-lg font-mono text-gray-100">
                          {coxResult.model_fit?.aic?.toFixed(2)}
                        </div>
                        <div className="text-xs text-gray-400">AIC</div>
                      </div>
                      <div className="bg-slate-700/50 rounded p-3">
                        <div className="text-lg font-mono text-gray-100">
                          {coxResult.model_fit?.log_likelihood?.toFixed(2)}
                        </div>
                        <div className="text-xs text-gray-400">Log-Likelihood</div>
                      </div>
                      <div className={`rounded p-3 ${
                        coxResult.model_fit?.likelihood_ratio_test?.significant
                          ? 'bg-green-900/50 border border-green-700'
                          : 'bg-slate-700/50'
                      }`}>
                        <div className="text-lg font-mono text-gray-100">
                          {coxResult.model_fit?.likelihood_ratio_test?.p_value?.toFixed(4)}
                        </div>
                        <div className="text-xs text-gray-400">LR Test P-Value</div>
                      </div>
                    </div>
                  </div>

                  {/* Forest Plot */}
                  {coxResult.forest_plot_data && coxResult.forest_plot_data.length > 0 && (
                    <div className="bg-slate-800 rounded-lg p-6">
                      <HazardRatioForest
                        data={coxResult.forest_plot_data}
                        title="Hazard Ratios with Confidence Intervals"
                      />
                    </div>
                  )}

                  {/* Hazard Ratios Table */}
                  <div className="bg-slate-800 rounded-lg p-6">
                    <h3 className="text-lg font-semibold text-gray-100 mb-4">Hazard Ratios</h3>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="text-gray-400 border-b border-slate-600">
                            <th className="text-left py-2 px-3">Covariate</th>
                            <th className="text-right py-2 px-3">HR</th>
                            <th className="text-right py-2 px-3">95% CI</th>
                            <th className="text-right py-2 px-3">P-Value</th>
                            <th className="text-left py-2 px-3">Interpretation</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(coxResult.hazard_ratios || {}).map(([cov, data]) => (
                            <tr key={cov} className="border-b border-slate-700/50">
                              <td className="py-2 px-3 font-medium text-gray-200">{cov}</td>
                              <td className="py-2 px-3 text-right font-mono">
                                {data.hazard_ratio?.toFixed(3)}
                              </td>
                              <td className="py-2 px-3 text-right font-mono text-gray-400">
                                [{data.hr_ci_lower?.toFixed(3)}, {data.hr_ci_upper?.toFixed(3)}]
                              </td>
                              <td className={`py-2 px-3 text-right font-mono ${
                                data.significant ? 'text-green-400' : 'text-gray-400'
                              }`}>
                                {data.p_value?.toFixed(4)}
                              </td>
                              <td className="py-2 px-3 text-gray-300 text-xs">
                                {data.interpretation}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Accelerated Life Testing Tab */}
          {activeTab === 'alt' && (
            <div className="space-y-6">
              <div className="bg-slate-800 rounded-lg p-6">
                <h2 className="text-xl font-semibold text-gray-100 mb-4">Accelerated Life Testing (ALT)</h2>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                  <div>
                    <label className="block text-sm text-gray-400 mb-2">Stress Variable Name</label>
                    <input
                      type="text"
                      value={stressVariable}
                      onChange={(e) => setStressVariable(e.target.value)}
                      className="w-full bg-slate-900 border border-slate-600 rounded px-3 py-2 text-gray-100"
                      placeholder="e.g., Temperature, Voltage"
                    />
                  </div>

                  <div>
                    <label className="block text-sm text-gray-400 mb-2">Life Distribution Model</label>
                    <select
                      value={altModelType}
                      onChange={(e) => setAltModelType(e.target.value)}
                      className="w-full bg-slate-900 border border-slate-600 rounded px-3 py-2 text-gray-100"
                    >
                      <option value="weibull">Weibull</option>
                      <option value="lognormal">Lognormal</option>
                      <option value="loglogistic">Log-Logistic</option>
                    </select>
                  </div>
                </div>

                <div className="mb-6">
                  <label className="block text-sm text-gray-400 mb-2">
                    Stress Values (one per observation, comma separated)
                  </label>
                  <textarea
                    value={stressValues.join(', ')}
                    onChange={(e) => {
                      const values = e.target.value.split(',').map(v => {
                        const num = parseFloat(v.trim())
                        return isNaN(num) ? 0 : num
                      })
                      setStressValues(values)
                    }}
                    className="w-full bg-slate-900 border border-slate-600 rounded px-3 py-2 text-gray-100 h-24"
                    placeholder="e.g., 85, 85, 85, 100, 100, 100, 125, 125, 125"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Enter stress levels corresponding to each observation in the Data tab
                  </p>
                </div>

                <div className="mb-6">
                  <label className="block text-sm text-gray-400 mb-2">Use Condition Stress Level (for extrapolation)</label>
                  <input
                    type="number"
                    value={useStress}
                    onChange={(e) => setUseStress(e.target.value)}
                    className="w-48 bg-slate-900 border border-slate-600 rounded px-3 py-2 text-gray-100"
                    placeholder="e.g., 25"
                  />
                </div>

                <button
                  onClick={runALT}
                  disabled={altLoading}
                  className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 text-white px-4 py-2 rounded-lg transition-colors"
                >
                  {altLoading ? (
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  ) : (
                    <Play className="w-4 h-4" />
                  )}
                  Run ALT Analysis
                </button>

                {altError && (
                  <div className="mt-4 p-4 bg-red-900/50 border border-red-700 rounded-lg flex items-center gap-2 text-red-300">
                    <AlertCircle className="w-5 h-5" />
                    {altError}
                  </div>
                )}
              </div>

              {/* ALT Results */}
              {altResult && (
                <div className="space-y-6">
                  {/* Model Fit */}
                  <div className="bg-slate-800 rounded-lg p-6">
                    <h3 className="text-lg font-semibold text-gray-100 mb-4">
                      {altResult.model_type?.charAt(0).toUpperCase() + altResult.model_type?.slice(1)} AFT Model
                    </h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="bg-slate-700/50 rounded p-3">
                        <div className="text-lg font-mono text-gray-100">
                          {altResult.model_fit?.log_likelihood?.toFixed(2)}
                        </div>
                        <div className="text-xs text-gray-400">Log-Likelihood</div>
                      </div>
                      <div className="bg-slate-700/50 rounded p-3">
                        <div className="text-lg font-mono text-gray-100">
                          {altResult.model_fit?.aic?.toFixed(2)}
                        </div>
                        <div className="text-xs text-gray-400">AIC</div>
                      </div>
                      <div className="bg-slate-700/50 rounded p-3">
                        <div className="text-lg font-mono text-gray-100">
                          {altResult.model_fit?.concordance_index?.toFixed(3)}
                        </div>
                        <div className="text-xs text-gray-400">Concordance</div>
                      </div>
                      <div className="bg-slate-700/50 rounded p-3">
                        <div className="text-lg font-mono text-gray-100">
                          {altResult.model_fit?.n_events}
                        </div>
                        <div className="text-xs text-gray-400">Events</div>
                      </div>
                    </div>
                  </div>

                  {/* Acceleration Factors */}
                  {altResult.acceleration_factors && (
                    <div className="bg-slate-800 rounded-lg p-6">
                      <h3 className="text-lg font-semibold text-gray-100 mb-4">Acceleration Factors</h3>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        {Object.entries(altResult.acceleration_factors).map(([stress, af]) => (
                          <div key={stress} className="bg-slate-700/50 rounded p-3">
                            <div className="text-lg font-mono text-gray-100">{af?.toFixed(2)}</div>
                            <div className="text-xs text-gray-400">At Stress = {stress}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Use Condition Predictions */}
                  {altResult.use_condition_predictions && (
                    <div className="bg-green-900/30 border border-green-700 rounded-lg p-6">
                      <h3 className="text-lg font-semibold text-green-300 mb-4">
                        Predictions at Use Conditions (Stress = {altResult.use_condition_predictions.stress_level})
                      </h3>
                      <div className="text-2xl font-mono text-green-200">
                        Median Life: {altResult.use_condition_predictions.median_life?.toFixed(2)}
                      </div>
                    </div>
                  )}

                  {/* Interpretation */}
                  {altResult.interpretation && (
                    <div className="bg-slate-800 rounded-lg p-4 flex items-start gap-3">
                      <Info className="w-5 h-5 text-blue-400 mt-0.5" />
                      <div className="text-gray-300">{altResult.interpretation}</div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Test Planning Tab */}
          {activeTab === 'planning' && (
            <div className="space-y-6">
              <div className="bg-slate-800 rounded-lg p-6">
                <h2 className="text-xl font-semibold text-gray-100 mb-4">Reliability Test Planning</h2>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                  <div>
                    <label className="block text-sm text-gray-400 mb-2">Test Type</label>
                    <select
                      value={testPlanParams.test_type}
                      onChange={(e) => setTestPlanParams({ ...testPlanParams, test_type: e.target.value })}
                      className="w-full bg-slate-900 border border-slate-600 rounded px-3 py-2 text-gray-100"
                    >
                      <option value="demonstration">Demonstration Test</option>
                      <option value="estimation">Estimation Test</option>
                      <option value="comparison">Comparison Test</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm text-gray-400 mb-2">Distribution</label>
                    <select
                      value={testPlanParams.distribution}
                      onChange={(e) => setTestPlanParams({ ...testPlanParams, distribution: e.target.value })}
                      className="w-full bg-slate-900 border border-slate-600 rounded px-3 py-2 text-gray-100"
                    >
                      <option value="exponential">Exponential</option>
                      <option value="weibull">Weibull</option>
                      <option value="binomial">Binomial</option>
                    </select>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                  <div>
                    <label className="block text-sm text-gray-400 mb-2">Target Reliability</label>
                    <input
                      type="number"
                      value={testPlanParams.target_reliability}
                      onChange={(e) => setTestPlanParams({ ...testPlanParams, target_reliability: parseFloat(e.target.value) })}
                      min="0"
                      max="1"
                      step="0.01"
                      className="w-full bg-slate-900 border border-slate-600 rounded px-3 py-2 text-gray-100"
                    />
                  </div>

                  <div>
                    <label className="block text-sm text-gray-400 mb-2">Test Time</label>
                    <input
                      type="number"
                      value={testPlanParams.test_time}
                      onChange={(e) => setTestPlanParams({ ...testPlanParams, test_time: parseFloat(e.target.value) })}
                      min="0"
                      className="w-full bg-slate-900 border border-slate-600 rounded px-3 py-2 text-gray-100"
                    />
                  </div>

                  <div>
                    <label className="block text-sm text-gray-400 mb-2">Confidence Level</label>
                    <select
                      value={testPlanParams.confidence_level}
                      onChange={(e) => setTestPlanParams({ ...testPlanParams, confidence_level: parseFloat(e.target.value) })}
                      className="w-full bg-slate-900 border border-slate-600 rounded px-3 py-2 text-gray-100"
                    >
                      <option value={0.90}>90%</option>
                      <option value={0.95}>95%</option>
                      <option value={0.99}>99%</option>
                    </select>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                  <div>
                    <label className="block text-sm text-gray-400 mb-2">Allowable Failures</label>
                    <input
                      type="number"
                      value={testPlanParams.allowable_failures}
                      onChange={(e) => setTestPlanParams({ ...testPlanParams, allowable_failures: parseInt(e.target.value) })}
                      min="0"
                      className="w-full bg-slate-900 border border-slate-600 rounded px-3 py-2 text-gray-100"
                    />
                  </div>

                  {testPlanParams.distribution === 'weibull' && (
                    <div>
                      <label className="block text-sm text-gray-400 mb-2">Weibull Shape Parameter</label>
                      <input
                        type="number"
                        value={testPlanParams.shape_parameter}
                        onChange={(e) => setTestPlanParams({ ...testPlanParams, shape_parameter: parseFloat(e.target.value) })}
                        min="0.1"
                        step="0.1"
                        className="w-full bg-slate-900 border border-slate-600 rounded px-3 py-2 text-gray-100"
                      />
                    </div>
                  )}

                  {testPlanParams.test_type === 'comparison' && (
                    <>
                      <div>
                        <label className="block text-sm text-gray-400 mb-2">Comparison Reliability</label>
                        <input
                          type="number"
                          value={testPlanParams.comparison_reliability}
                          onChange={(e) => setTestPlanParams({ ...testPlanParams, comparison_reliability: parseFloat(e.target.value) })}
                          min="0"
                          max="1"
                          step="0.01"
                          className="w-full bg-slate-900 border border-slate-600 rounded px-3 py-2 text-gray-100"
                        />
                      </div>
                      <div>
                        <label className="block text-sm text-gray-400 mb-2">Power</label>
                        <input
                          type="number"
                          value={testPlanParams.power}
                          onChange={(e) => setTestPlanParams({ ...testPlanParams, power: parseFloat(e.target.value) })}
                          min="0"
                          max="1"
                          step="0.05"
                          className="w-full bg-slate-900 border border-slate-600 rounded px-3 py-2 text-gray-100"
                        />
                      </div>
                    </>
                  )}
                </div>

                <button
                  onClick={runTestPlanning}
                  disabled={testPlanLoading}
                  className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 text-white px-4 py-2 rounded-lg transition-colors"
                >
                  {testPlanLoading ? (
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  ) : (
                    <Calculator className="w-4 h-4" />
                  )}
                  Calculate Sample Size
                </button>

                {testPlanError && (
                  <div className="mt-4 p-4 bg-red-900/50 border border-red-700 rounded-lg flex items-center gap-2 text-red-300">
                    <AlertCircle className="w-5 h-5" />
                    {testPlanError}
                  </div>
                )}
              </div>

              {/* Test Planning Results */}
              {testPlanResult && (
                <div className="space-y-6">
                  <div className="bg-slate-800 rounded-lg p-6">
                    <h3 className="text-lg font-semibold text-gray-100 mb-4">
                      {testPlanResult.test_type === 'demonstration' && 'Demonstration Test Plan'}
                      {testPlanResult.test_type === 'estimation' && 'Estimation Test Plan'}
                      {testPlanResult.test_type === 'comparison' && 'Comparison Test Plan'}
                    </h3>

                    {testPlanResult.results?.demonstration_test && (
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          <div className="bg-blue-900/30 border border-blue-700 rounded p-4">
                            <div className="text-3xl font-bold text-blue-300">
                              {testPlanResult.results.demonstration_test.sample_size}
                            </div>
                            <div className="text-sm text-blue-400">Required Sample Size</div>
                          </div>
                          <div className="bg-slate-700/50 rounded p-4">
                            <div className="text-xl font-mono text-gray-100">
                              {testPlanResult.results.demonstration_test.test_time}
                            </div>
                            <div className="text-xs text-gray-400">Test Duration</div>
                          </div>
                          <div className="bg-slate-700/50 rounded p-4">
                            <div className="text-xl font-mono text-gray-100">
                              {testPlanResult.results.demonstration_test.allowable_failures}
                            </div>
                            <div className="text-xs text-gray-400">Allowable Failures</div>
                          </div>
                          <div className="bg-slate-700/50 rounded p-4">
                            <div className="text-xl font-mono text-gray-100">
                              {(testPlanResult.results.demonstration_test.confidence_level * 100).toFixed(0)}%
                            </div>
                            <div className="text-xs text-gray-400">Confidence Level</div>
                          </div>
                        </div>

                        <div className="bg-green-900/30 border border-green-700 rounded p-4">
                          <div className="text-green-300">
                            {testPlanResult.results.demonstration_test.interpretation}
                          </div>
                        </div>
                      </div>
                    )}

                    {testPlanResult.results?.estimation_test && (
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          <div className="bg-blue-900/30 border border-blue-700 rounded p-4">
                            <div className="text-3xl font-bold text-blue-300">
                              {testPlanResult.results.estimation_test.sample_size}
                            </div>
                            <div className="text-sm text-blue-400">Required Sample Size</div>
                          </div>
                          <div className="bg-slate-700/50 rounded p-4">
                            <div className="text-xl font-mono text-gray-100">
                              {testPlanResult.results.estimation_test.precision?.toFixed(3)}
                            </div>
                            <div className="text-xs text-gray-400">Precision (Half-Width)</div>
                          </div>
                          <div className="bg-slate-700/50 rounded p-4">
                            <div className="text-xl font-mono text-gray-100">
                              {testPlanResult.results.estimation_test.expected_ci_width?.toFixed(3)}
                            </div>
                            <div className="text-xs text-gray-400">Expected CI Width</div>
                          </div>
                        </div>

                        <div className="bg-green-900/30 border border-green-700 rounded p-4">
                          <div className="text-green-300">
                            {testPlanResult.results.estimation_test.interpretation}
                          </div>
                        </div>
                      </div>
                    )}

                    {testPlanResult.results?.comparison_test && (
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          <div className="bg-blue-900/30 border border-blue-700 rounded p-4">
                            <div className="text-3xl font-bold text-blue-300">
                              {testPlanResult.results.comparison_test.sample_size_per_group}
                            </div>
                            <div className="text-sm text-blue-400">Sample Size per Group</div>
                          </div>
                          <div className="bg-blue-900/30 border border-blue-700 rounded p-4">
                            <div className="text-3xl font-bold text-blue-300">
                              {testPlanResult.results.comparison_test.total_sample_size}
                            </div>
                            <div className="text-sm text-blue-400">Total Sample Size</div>
                          </div>
                          <div className="bg-slate-700/50 rounded p-4">
                            <div className="text-xl font-mono text-gray-100">
                              {testPlanResult.results.comparison_test.difference_to_detect?.toFixed(3)}
                            </div>
                            <div className="text-xs text-gray-400">Difference to Detect</div>
                          </div>
                          <div className="bg-slate-700/50 rounded p-4">
                            <div className="text-xl font-mono text-gray-100">
                              {(testPlanResult.results.comparison_test.power * 100).toFixed(0)}%
                            </div>
                            <div className="text-xs text-gray-400">Power</div>
                          </div>
                        </div>

                        <div className="bg-green-900/30 border border-green-700 rounded p-4">
                          <div className="text-green-300">
                            {testPlanResult.results.comparison_test.interpretation}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default ReliabilityAnalysis
