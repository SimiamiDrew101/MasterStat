import { useState } from 'react'
import { Target, Calculator, HelpCircle, RefreshCw, Beaker, Zap } from 'lucide-react'
import axios from 'axios'
import PowerCurveChart from '../components/PowerCurveChart'
import ForestPlot from '../components/ForestPlot'
import SensitivityCurve from '../components/SensitivityCurve'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const ExperimentPlanning = () => {
  // Calculator type
  const [calculatorType, setCalculatorType] = useState('t-test') // 't-test', 'anova', or 'proportion'

  // Common parameters
  const [power, setPower] = useState(0.8)
  const [alpha, setAlpha] = useState(0.05)

  // T-test specific
  const [testType, setTestType] = useState('one-sample')
  const [alternative, setAlternative] = useState('two-sided')
  const [effectSizeMethod, setEffectSizeMethod] = useState('direct')
  const [effectSize, setEffectSize] = useState(0.5)
  const [meanDiff, setMeanDiff] = useState(5)
  const [stdDev, setStdDev] = useState(10)
  const [ratio, setRatio] = useState(1.0)
  const [correlation, setCorrelation] = useState(0.5)

  // ANOVA specific
  const [anovaType, setAnovaType] = useState('one-way')
  const [anovaEffectSizeMethod, setAnovaEffectSizeMethod] = useState('cohens-f')
  const [cohensF, setCohensF] = useState(0.25)
  const [etaSquared, setEtaSquared] = useState(0.06)
  const [numGroups, setNumGroups] = useState(3)
  const [numLevelsA, setNumLevelsA] = useState(2)
  const [numLevelsB, setNumLevelsB] = useState(2)
  const [effectOfInterest, setEffectOfInterest] = useState('main_a')

  // Proportion test specific
  const [proportionTestType, setProportionTestType] = useState('one-sample')
  const [p0, setP0] = useState(0.5)
  const [p1, setP1] = useState(0.65)
  const [p1Group1, setP1Group1] = useState(0.5)
  const [p2Group2, setP2Group2] = useState(0.65)
  const [proportionRatio, setProportionRatio] = useState(1.0)
  const [pDiscordant, setPDiscordant] = useState(0.3)
  const [pDiff, setPDiff] = useState(0.1)

  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Effect Size Tools state
  const [activeToolTab, setActiveToolTab] = useState('converter') // 'converter', 'pilot', 'minimum'

  // Effect Size Converter
  const [fromMetric, setFromMetric] = useState('cohens_d')
  const [toMetric, setToMetric] = useState('cohens_f')
  const [converterValue, setConverterValue] = useState(0.5)
  const [converterResult, setConverterResult] = useState(null)
  const [converterLoading, setConverterLoading] = useState(false)

  // Pilot Data Calculator
  const [pilotTestType, setPilotTestType] = useState('independent')
  const [pilotMean1, setPilotMean1] = useState(10)
  const [pilotSd1, setPilotSd1] = useState(2)
  const [pilotN1, setPilotN1] = useState(20)
  const [pilotMean2, setPilotMean2] = useState(12)
  const [pilotSd2, setPilotSd2] = useState(2)
  const [pilotN2, setPilotN2] = useState(20)
  const [pilotMeanDiff, setPilotMeanDiff] = useState(2)
  const [pilotSdDiff, setPilotSdDiff] = useState(3)
  const [pilotGroupMeans, setPilotGroupMeans] = useState([10, 12, 14])
  const [pilotGroupSds, setPilotGroupSds] = useState([2, 2, 2])
  const [pilotGroupNs, setPilotGroupNs] = useState([15, 15, 15])
  const [pilotResult, setPilotResult] = useState(null)
  const [pilotLoading, setPilotLoading] = useState(false)

  // Minimum Detectable Effect Size
  const [minTestFamily, setMinTestFamily] = useState('t-test')
  const [minTestType, setMinTestType] = useState('one-sample')
  const [minSampleSize, setMinSampleSize] = useState(50)
  const [minPower, setMinPower] = useState(0.8)
  const [minAlpha, setMinAlpha] = useState(0.05)
  const [minAlternative, setMinAlternative] = useState('two-sided')
  const [minRatio, setMinRatio] = useState(1.0)
  const [minCorrelation, setMinCorrelation] = useState(0.5)
  const [minNumGroups, setMinNumGroups] = useState(3)
  const [minNumLevelsA, setMinNumLevelsA] = useState(2)
  const [minNumLevelsB, setMinNumLevelsB] = useState(2)
  const [minResult, setMinResult] = useState(null)
  const [minLoading, setMinLoading] = useState(false)

  const handleCalculate = async () => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      let endpoint = ''
      let payload = {}

      if (calculatorType === 'proportion') {
        endpoint = '/api/power/sample-size/proportion'
        payload = {
          test_type: proportionTestType,
          power: parseFloat(power),
          alpha: parseFloat(alpha),
          alternative: alternative
        }

        if (proportionTestType === 'one-sample') {
          payload.p0 = parseFloat(p0)
          payload.p1 = parseFloat(p1)
        } else if (proportionTestType === 'two-sample') {
          payload.p1_group1 = parseFloat(p1Group1)
          payload.p2_group2 = parseFloat(p2Group2)
          payload.ratio = parseFloat(proportionRatio)
        } else { // mcnemar
          payload.p_discordant = parseFloat(pDiscordant)
          payload.p_diff = parseFloat(pDiff)
        }
      } else if (calculatorType === 't-test') {
        endpoint = '/api/power/sample-size/t-test'
        payload = {
          test_type: testType,
          power: parseFloat(power),
          alpha: parseFloat(alpha),
          alternative: alternative
        }

        if (effectSizeMethod === 'direct') {
          payload.effect_size = parseFloat(effectSize)
        } else {
          payload.mean_diff = parseFloat(meanDiff)
          payload.std_dev = parseFloat(stdDev)
        }

        if (testType === 'two-sample') {
          payload.ratio = parseFloat(ratio)
        }

        if (testType === 'paired') {
          payload.correlation = parseFloat(correlation)
        }
      } else {
        // ANOVA
        endpoint = '/api/power/sample-size/anova'
        payload = {
          anova_type: anovaType,
          power: parseFloat(power),
          alpha: parseFloat(alpha)
        }

        if (anovaEffectSizeMethod === 'cohens-f') {
          payload.effect_size = parseFloat(cohensF)
        } else {
          payload.eta_squared = parseFloat(etaSquared)
        }

        if (anovaType === 'one-way') {
          payload.num_groups = parseInt(numGroups)
        } else {
          payload.num_levels_a = parseInt(numLevelsA)
          payload.num_levels_b = parseInt(numLevelsB)
          payload.effect_of_interest = effectOfInterest
        }
      }

      const response = await axios.post(`${API_URL}${endpoint}`, payload)
      setResult(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  const handleConvertEffectSize = async () => {
    setConverterLoading(true)
    setConverterResult(null)

    try {
      const response = await axios.post(`${API_URL}/api/power/convert-effect-size`, {
        from_metric: fromMetric,
        to_metric: toMetric,
        value: parseFloat(converterValue)
      })
      setConverterResult(response.data)
    } catch (err) {
      setConverterResult({ error: err.response?.data?.detail || err.message })
    } finally {
      setConverterLoading(false)
    }
  }

  const handleCalculatePilotEffectSize = async () => {
    setPilotLoading(true)
    setPilotResult(null)

    try {
      let payload = { test_type: pilotTestType }

      if (pilotTestType === 'independent') {
        payload = {
          ...payload,
          mean1: parseFloat(pilotMean1),
          sd1: parseFloat(pilotSd1),
          n1: parseInt(pilotN1),
          mean2: parseFloat(pilotMean2),
          sd2: parseFloat(pilotSd2),
          n2: parseInt(pilotN2)
        }
      } else if (pilotTestType === 'paired') {
        payload = {
          ...payload,
          mean_diff: parseFloat(pilotMeanDiff),
          sd_diff: parseFloat(pilotSdDiff)
        }
      } else {
        payload = {
          ...payload,
          group_means: pilotGroupMeans.map(v => parseFloat(v)),
          group_sds: pilotGroupSds.map(v => parseFloat(v)),
          group_ns: pilotGroupNs.map(v => parseInt(v))
        }
      }

      const response = await axios.post(`${API_URL}/api/power/effect-size-from-pilot`, payload)
      setPilotResult(response.data)
    } catch (err) {
      setPilotResult({ error: err.response?.data?.detail || err.message })
    } finally {
      setPilotLoading(false)
    }
  }

  const handleCalculateMinimumEffectSize = async () => {
    setMinLoading(true)
    setMinResult(null)

    try {
      let payload = {
        test_family: minTestFamily,
        test_type: minTestType,
        sample_size: parseInt(minSampleSize),
        power: parseFloat(minPower),
        alpha: parseFloat(minAlpha)
      }

      if (minTestFamily === 't-test') {
        payload.alternative = minAlternative
        if (minTestType === 'two-sample') {
          payload.ratio = parseFloat(minRatio)
        }
        if (minTestType === 'paired') {
          payload.correlation = parseFloat(minCorrelation)
        }
      } else {
        if (minTestType === 'one-way') {
          payload.num_groups = parseInt(minNumGroups)
        } else {
          payload.num_levels_a = parseInt(minNumLevelsA)
          payload.num_levels_b = parseInt(minNumLevelsB)
        }
      }

      const response = await axios.post(`${API_URL}/api/power/minimum-effect-size`, payload)
      setMinResult(response.data)
    } catch (err) {
      setMinResult({ error: err.response?.data?.detail || err.message })
    } finally {
      setMinLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <div className="flex items-center space-x-3 mb-4">
          <Target className="w-8 h-8 text-cyan-400" />
          <h2 className="text-3xl font-bold text-gray-100">Sample Size Calculator</h2>
        </div>
        <p className="text-gray-300">
          Calculate the required sample size for your study based on desired power, significance level, and effect size.
        </p>
      </div>

      {/* Calculator Type Selector */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <label className="block text-gray-200 font-medium mb-3">Select Analysis Type</label>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button
            onClick={() => setCalculatorType('t-test')}
            className={`p-4 rounded-lg border-2 transition-all ${
              calculatorType === 't-test'
                ? 'border-cyan-500 bg-cyan-500/10'
                : 'border-slate-600 bg-slate-700/30 hover:bg-slate-700/50'
            }`}
          >
            <h3 className="text-lg font-semibold text-gray-100 mb-1">t-Test</h3>
            <p className="text-sm text-gray-400">One-sample, two-sample, or paired t-tests</p>
          </button>
          <button
            onClick={() => setCalculatorType('anova')}
            className={`p-4 rounded-lg border-2 transition-all ${
              calculatorType === 'anova'
                ? 'border-cyan-500 bg-cyan-500/10'
                : 'border-slate-600 bg-slate-700/30 hover:bg-slate-700/50'
            }`}
          >
            <h3 className="text-lg font-semibold text-gray-100 mb-1">ANOVA</h3>
            <p className="text-sm text-gray-400">One-way or two-way ANOVA designs</p>
          </button>
          <button
            onClick={() => setCalculatorType('proportion')}
            className={`p-4 rounded-lg border-2 transition-all ${
              calculatorType === 'proportion'
                ? 'border-cyan-500 bg-cyan-500/10'
                : 'border-slate-600 bg-slate-700/30 hover:bg-slate-700/50'
            }`}
          >
            <h3 className="text-lg font-semibold text-gray-100 mb-1">Proportion Test</h3>
            <p className="text-sm text-gray-400">One-sample, two-sample, or McNemar's test</p>
          </button>
        </div>
      </div>

      {/* Calculator Form */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <div className="space-y-6">
          {calculatorType === 't-test' ? (
            <>
              {/* T-Test Type Selection */}
              <div>
                <label className="block text-gray-200 font-medium mb-2">Test Type</label>
                <select
                  value={testType}
                  onChange={(e) => setTestType(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                >
                  <option value="one-sample">One-Sample t-Test</option>
                  <option value="two-sample">Two-Sample t-Test (Independent)</option>
                  <option value="paired">Paired t-Test</option>
                </select>
                <p className="text-gray-400 text-xs mt-1">
                  {testType === 'one-sample' && 'Compare a sample mean to a known population mean'}
                  {testType === 'two-sample' && 'Compare means of two independent groups'}
                  {testType === 'paired' && 'Compare means of matched pairs (e.g., before/after)'}
                </p>
              </div>

              {/* Alternative Hypothesis */}
              <div>
                <label className="block text-gray-200 font-medium mb-2">Alternative Hypothesis</label>
                <select
                  value={alternative}
                  onChange={(e) => setAlternative(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                >
                  <option value="two-sided">Two-Sided (≠)</option>
                  <option value="greater">Greater (&gt;)</option>
                  <option value="less">Less (&lt;)</option>
                </select>
              </div>

              {/* Effect Size Specification for t-test */}
              <div className="border-t border-slate-700 pt-4">
                <div className="flex items-center space-x-2 mb-3">
                  <label className="block text-gray-200 font-medium">Effect Size (Cohen's d)</label>
                  <HelpCircle className="w-4 h-4 text-gray-400" />
                </div>

                <div className="mb-3">
                  <label className="flex items-center space-x-2 text-gray-200 mb-2">
                    <input
                      type="radio"
                      value="direct"
                      checked={effectSizeMethod === 'direct'}
                      onChange={() => setEffectSizeMethod('direct')}
                      className="w-4 h-4"
                    />
                    <span>Specify Cohen's d directly</span>
                  </label>
                  <label className="flex items-center space-x-2 text-gray-200">
                    <input
                      type="radio"
                      value="parameters"
                      checked={effectSizeMethod === 'parameters'}
                      onChange={() => setEffectSizeMethod('parameters')}
                      className="w-4 h-4"
                    />
                    <span>Calculate from mean difference and standard deviation</span>
                  </label>
                </div>

                {effectSizeMethod === 'direct' ? (
                  <div>
                    <input
                      type="number"
                      step="0.01"
                      min="0.01"
                      value={effectSize}
                      onChange={(e) => setEffectSize(parseFloat(e.target.value))}
                      className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                      placeholder="Enter Cohen's d"
                    />
                    <p className="text-gray-400 text-xs mt-1">
                      Small: 0.2, Medium: 0.5, Large: 0.8 (Cohen's conventions)
                    </p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <div>
                      <label className="block text-gray-300 text-sm mb-1">Expected Mean Difference</label>
                      <input
                        type="number"
                        step="0.1"
                        value={meanDiff}
                        onChange={(e) => setMeanDiff(parseFloat(e.target.value))}
                        className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                        placeholder="Mean difference"
                      />
                    </div>
                    <div>
                      <label className="block text-gray-300 text-sm mb-1">Standard Deviation</label>
                      <input
                        type="number"
                        step="0.1"
                        min="0.01"
                        value={stdDev}
                        onChange={(e) => setStdDev(parseFloat(e.target.value))}
                        className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                        placeholder="Standard deviation"
                      />
                    </div>
                    <p className="text-gray-400 text-xs">
                      Cohen's d = {(meanDiff / stdDev).toFixed(3)}
                    </p>
                  </div>
                )}
              </div>

              {/* Two-Sample Specific Options */}
              {testType === 'two-sample' && (
                <div className="border-t border-slate-700 pt-4">
                  <label className="block text-gray-200 font-medium mb-2">Sample Size Ratio (n2/n1)</label>
                  <input
                    type="number"
                    step="0.1"
                    min="0.1"
                    max="10"
                    value={ratio}
                    onChange={(e) => setRatio(parseFloat(e.target.value))}
                    className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                  />
                  <p className="text-gray-400 text-xs mt-1">
                    Use 1.0 for equal sample sizes
                  </p>
                </div>
              )}

              {/* Paired Test Specific Options */}
              {testType === 'paired' && (
                <div className="border-t border-slate-700 pt-4">
                  <label className="block text-gray-200 font-medium mb-2">Correlation Between Pairs</label>
                  <div className="flex items-center space-x-4">
                    <input
                      type="range"
                      min="-0.99"
                      max="0.99"
                      step="0.01"
                      value={correlation}
                      onChange={(e) => setCorrelation(parseFloat(e.target.value))}
                      className="flex-1"
                    />
                    <input
                      type="number"
                      min="-0.99"
                      max="0.99"
                      step="0.01"
                      value={correlation}
                      onChange={(e) => setCorrelation(parseFloat(e.target.value))}
                      className="w-20 px-3 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                    />
                  </div>
                  <p className="text-gray-400 text-xs mt-1">
                    Expected correlation between paired measurements
                  </p>
                </div>
              )}
            </>
          ) : calculatorType === 'anova' ? (
            <>
              {/* ANOVA Type Selection */}
              <div>
                <label className="block text-gray-200 font-medium mb-2">ANOVA Type</label>
                <select
                  value={anovaType}
                  onChange={(e) => setAnovaType(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                >
                  <option value="one-way">One-Way ANOVA</option>
                  <option value="two-way">Two-Way ANOVA</option>
                </select>
                <p className="text-gray-400 text-xs mt-1">
                  {anovaType === 'one-way' && 'Compare means across multiple groups (one factor)'}
                  {anovaType === 'two-way' && 'Compare means with two factors and their interaction'}
                </p>
              </div>

              {/* ANOVA Design Parameters */}
              {anovaType === 'one-way' ? (
                <div>
                  <label className="block text-gray-200 font-medium mb-2">Number of Groups</label>
                  <input
                    type="number"
                    min="2"
                    max="20"
                    value={numGroups}
                    onChange={(e) => setNumGroups(parseInt(e.target.value))}
                    className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                  />
                  <p className="text-gray-400 text-xs mt-1">
                    Number of independent groups to compare
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-gray-200 font-medium mb-2">Factor A Levels</label>
                      <input
                        type="number"
                        min="2"
                        max="10"
                        value={numLevelsA}
                        onChange={(e) => setNumLevelsA(parseInt(e.target.value))}
                        className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                      />
                    </div>
                    <div>
                      <label className="block text-gray-200 font-medium mb-2">Factor B Levels</label>
                      <input
                        type="number"
                        min="2"
                        max="10"
                        value={numLevelsB}
                        onChange={(e) => setNumLevelsB(parseInt(e.target.value))}
                        className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                      />
                    </div>
                  </div>
                  <div>
                    <label className="block text-gray-200 font-medium mb-2">Effect of Interest</label>
                    <select
                      value={effectOfInterest}
                      onChange={(e) => setEffectOfInterest(e.target.value)}
                      className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                    >
                      <option value="main_a">Main Effect of Factor A</option>
                      <option value="main_b">Main Effect of Factor B</option>
                      <option value="interaction">Interaction Effect (A×B)</option>
                    </select>
                    <p className="text-gray-400 text-xs mt-1">
                      Which effect should the study be powered to detect?
                    </p>
                  </div>
                </div>
              )}

              {/* Effect Size Specification for ANOVA */}
              <div className="border-t border-slate-700 pt-4">
                <div className="flex items-center space-x-2 mb-3">
                  <label className="block text-gray-200 font-medium">Effect Size</label>
                  <HelpCircle className="w-4 h-4 text-gray-400" />
                </div>

                <div className="mb-3">
                  <label className="flex items-center space-x-2 text-gray-200 mb-2">
                    <input
                      type="radio"
                      value="cohens-f"
                      checked={anovaEffectSizeMethod === 'cohens-f'}
                      onChange={() => setAnovaEffectSizeMethod('cohens-f')}
                      className="w-4 h-4"
                    />
                    <span>Specify Cohen's f</span>
                  </label>
                  <label className="flex items-center space-x-2 text-gray-200">
                    <input
                      type="radio"
                      value="eta-squared"
                      checked={anovaEffectSizeMethod === 'eta-squared'}
                      onChange={() => setAnovaEffectSizeMethod('eta-squared')}
                      className="w-4 h-4"
                    />
                    <span>Specify Eta-squared (η²)</span>
                  </label>
                </div>

                {anovaEffectSizeMethod === 'cohens-f' ? (
                  <div>
                    <input
                      type="number"
                      step="0.01"
                      min="0.01"
                      value={cohensF}
                      onChange={(e) => setCohensF(parseFloat(e.target.value))}
                      className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                      placeholder="Enter Cohen's f"
                    />
                    <p className="text-gray-400 text-xs mt-1">
                      Small: 0.10, Medium: 0.25, Large: 0.40 (Cohen's conventions for ANOVA)
                    </p>
                  </div>
                ) : (
                  <div>
                    <input
                      type="number"
                      step="0.01"
                      min="0.001"
                      max="0.99"
                      value={etaSquared}
                      onChange={(e) => setEtaSquared(parseFloat(e.target.value))}
                      className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                      placeholder="Enter η²"
                    />
                    <p className="text-gray-400 text-xs mt-1">
                      Proportion of variance explained (0 to 1). Cohen's f = {(etaSquared > 0 && etaSquared < 1) ? Math.sqrt(etaSquared / (1 - etaSquared)).toFixed(3) : 'N/A'}
                    </p>
                  </div>
                )}
              </div>
            </>
          ) : calculatorType === 'proportion' ? (
            <>
              {/* Proportion Test Type Selection */}
              <div>
                <label className="block text-gray-200 font-medium mb-2">Test Type</label>
                <select
                  value={proportionTestType}
                  onChange={(e) => setProportionTestType(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                >
                  <option value="one-sample">One-Sample Proportion Test</option>
                  <option value="two-sample">Two-Sample Proportion Test</option>
                  <option value="mcnemar">McNemar's Test (Paired)</option>
                </select>
                <p className="text-gray-400 text-xs mt-1">
                  {proportionTestType === 'one-sample' && 'Compare a sample proportion to a known population proportion'}
                  {proportionTestType === 'two-sample' && 'Compare proportions between two independent groups (A/B testing)'}
                  {proportionTestType === 'mcnemar' && 'Compare proportions for matched pairs (before/after)'}
                </p>
              </div>

              {/* Alternative Hypothesis */}
              <div>
                <label className="block text-gray-200 font-medium mb-2">Alternative Hypothesis</label>
                <select
                  value={alternative}
                  onChange={(e) => setAlternative(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                >
                  <option value="two-sided">Two-Sided (≠)</option>
                  <option value="greater">Greater (&gt;)</option>
                  <option value="less">Less (&lt;)</option>
                </select>
              </div>

              {/* Proportion Parameters */}
              {proportionTestType === 'one-sample' && (
                <div className="border-t border-slate-700 pt-4 space-y-3">
                  <h4 className="text-gray-200 font-semibold">Proportions</h4>
                  <div>
                    <label className="block text-gray-300 text-sm mb-1">Null Hypothesis Proportion (p₀)</label>
                    <input
                      type="number"
                      step="0.01"
                      min="0"
                      max="1"
                      value={p0}
                      onChange={(e) => setP0(e.target.value)}
                      className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                      placeholder="e.g., 0.5"
                    />
                    <p className="text-gray-400 text-xs mt-1">Expected proportion under null hypothesis</p>
                  </div>
                  <div>
                    <label className="block text-gray-300 text-sm mb-1">Alternative Proportion (p₁)</label>
                    <input
                      type="number"
                      step="0.01"
                      min="0"
                      max="1"
                      value={p1}
                      onChange={(e) => setP1(e.target.value)}
                      className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                      placeholder="e.g., 0.65"
                    />
                    <p className="text-gray-400 text-xs mt-1">Expected proportion you want to detect</p>
                  </div>
                  <div className="bg-blue-900/20 border border-blue-700/50 rounded-lg p-3">
                    <p className="text-gray-300 text-sm">
                      Effect size: {Math.abs(p1 - p0).toFixed(3)} ({(Math.abs(p1 - p0) * 100).toFixed(1)}%)
                    </p>
                  </div>
                </div>
              )}

              {proportionTestType === 'two-sample' && (
                <div className="border-t border-slate-700 pt-4 space-y-3">
                  <h4 className="text-gray-200 font-semibold">Group Proportions</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-gray-300 text-sm mb-1">Group 1 Proportion (p₁)</label>
                      <input
                        type="number"
                        step="0.01"
                        min="0"
                        max="1"
                        value={p1Group1}
                        onChange={(e) => setP1Group1(e.target.value)}
                        className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                        placeholder="e.g., 0.5"
                      />
                      <p className="text-gray-400 text-xs mt-1">Control group proportion</p>
                    </div>
                    <div>
                      <label className="block text-gray-300 text-sm mb-1">Group 2 Proportion (p₂)</label>
                      <input
                        type="number"
                        step="0.01"
                        min="0"
                        max="1"
                        value={p2Group2}
                        onChange={(e) => setP2Group2(e.target.value)}
                        className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                        placeholder="e.g., 0.65"
                      />
                      <p className="text-gray-400 text-xs mt-1">Treatment group proportion</p>
                    </div>
                  </div>
                  <div>
                    <label className="block text-gray-200 font-medium mb-2">Sample Size Ratio (n2/n1)</label>
                    <input
                      type="number"
                      step="0.1"
                      min="0.1"
                      max="10"
                      value={proportionRatio}
                      onChange={(e) => setProportionRatio(e.target.value)}
                      className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                    />
                    <p className="text-gray-400 text-xs mt-1">Use 1.0 for equal sample sizes</p>
                  </div>
                  <div className="bg-blue-900/20 border border-blue-700/50 rounded-lg p-3">
                    <p className="text-gray-300 text-sm">
                      Effect size: {Math.abs(p2Group2 - p1Group1).toFixed(3)} ({(Math.abs(p2Group2 - p1Group1) * 100).toFixed(1)}%)
                    </p>
                    <p className="text-gray-400 text-xs mt-1">
                      Absolute risk difference between groups
                    </p>
                  </div>
                </div>
              )}

              {proportionTestType === 'mcnemar' && (
                <div className="border-t border-slate-700 pt-4 space-y-3">
                  <h4 className="text-gray-200 font-semibold">McNemar's Test Parameters</h4>
                  <div>
                    <label className="block text-gray-300 text-sm mb-1">Proportion of Discordant Pairs</label>
                    <input
                      type="number"
                      step="0.01"
                      min="0"
                      max="1"
                      value={pDiscordant}
                      onChange={(e) => setPDiscordant(e.target.value)}
                      className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                      placeholder="e.g., 0.3"
                    />
                    <p className="text-gray-400 text-xs mt-1">
                      Proportion of pairs that differ between measurements (p₁₀ + p₀₁)
                    </p>
                  </div>
                  <div>
                    <label className="block text-gray-300 text-sm mb-1">Difference in Proportions</label>
                    <input
                      type="number"
                      step="0.01"
                      min="-1"
                      max="1"
                      value={pDiff}
                      onChange={(e) => setPDiff(e.target.value)}
                      className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                      placeholder="e.g., 0.1"
                    />
                    <p className="text-gray-400 text-xs mt-1">
                      Expected difference (p₁₀ - p₀₁). Positive = more +/- pairs than -/+ pairs
                    </p>
                  </div>
                  <div className="bg-blue-900/20 border border-blue-700/50 rounded-lg p-3">
                    <p className="text-gray-300 text-sm">
                      p₁₀ (+ then -): {((parseFloat(pDiscordant) + parseFloat(pDiff)) / 2).toFixed(3)}
                    </p>
                    <p className="text-gray-300 text-sm">
                      p₀₁ (- then +): {((parseFloat(pDiscordant) - parseFloat(pDiff)) / 2).toFixed(3)}
                    </p>
                  </div>
                </div>
              )}
            </>
          ) : null}

          {/* Common Parameters */}
          <div className="border-t border-slate-700 pt-4 space-y-4">
            {/* Power */}
            <div>
              <label className="block text-gray-200 font-medium mb-2">Statistical Power (1 - β)</label>
              <div className="flex items-center space-x-4">
                <input
                  type="range"
                  min="0.5"
                  max="0.99"
                  step="0.01"
                  value={power}
                  onChange={(e) => setPower(parseFloat(e.target.value))}
                  className="flex-1"
                />
                <input
                  type="number"
                  min="0.5"
                  max="0.99"
                  step="0.01"
                  value={power}
                  onChange={(e) => setPower(parseFloat(e.target.value))}
                  className="w-20 px-3 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                />
              </div>
              <p className="text-gray-400 text-xs mt-1">
                Probability of detecting a true effect (typically 0.8 or 0.9)
              </p>
            </div>

            {/* Significance Level */}
            <div>
              <label className="block text-gray-200 font-medium mb-2">Significance Level (α)</label>
              <select
                value={alpha}
                onChange={(e) => setAlpha(parseFloat(e.target.value))}
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
              >
                <option value="0.001">0.001</option>
                <option value="0.01">0.01</option>
                <option value="0.05">0.05</option>
                <option value="0.10">0.10</option>
              </select>
              <p className="text-gray-400 text-xs mt-1">
                Maximum acceptable probability of Type I error (typically 0.05)
              </p>
            </div>
          </div>

          {/* Calculate Button */}
          <button
            onClick={handleCalculate}
            disabled={loading}
            className="w-full bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-bold py-3 px-6 rounded-lg hover:from-cyan-600 hover:to-blue-600 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
          >
            <Calculator className="w-5 h-5" />
            <span>{loading ? 'Calculating...' : 'Calculate Sample Size'}</span>
          </button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-900/30 backdrop-blur-lg rounded-xl p-4 border border-red-700/50">
          <p className="text-red-200 font-medium">Error: {error}</p>
        </div>
      )}

      {/* Results Display */}
      {result && (
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
          <h3 className="text-2xl font-bold text-gray-100 mb-4">Sample Size Results</h3>

          {/* Main Result */}
          <div className="bg-gradient-to-r from-cyan-500/20 to-blue-500/20 rounded-lg p-6 mb-6">
            <h4 className="text-lg font-semibold text-gray-200 mb-2">{result.test_type}</h4>

            {/* T-test results */}
            {result.sample_size !== undefined && (
              <div className="text-3xl font-bold text-cyan-400">
                n = {result.sample_size}
              </div>
            )}

            {/* Two-sample t-test results */}
            {result.per_group && (
              <div className="space-y-2">
                <div className="text-2xl font-bold text-cyan-400">
                  Group 1: n₁ = {result.per_group.group_1}
                </div>
                <div className="text-2xl font-bold text-cyan-400">
                  Group 2: n₂ = {result.per_group.group_2}
                </div>
                <div className="text-xl font-semibold text-gray-300 mt-2">
                  Total: N = {result.total_sample_size}
                </div>
              </div>
            )}

            {/* One-way ANOVA results */}
            {result.sample_size_per_group && (
              <div className="space-y-2">
                <div className="text-2xl font-bold text-cyan-400">
                  Per Group: n = {result.sample_size_per_group}
                </div>
                <div className="text-xl font-semibold text-gray-300">
                  {result.num_groups} groups × {result.sample_size_per_group} = {result.total_sample_size} total
                </div>
              </div>
            )}

            {/* Two-way ANOVA results */}
            {result.sample_size_per_cell && (
              <div className="space-y-2">
                <div className="text-2xl font-bold text-cyan-400">
                  Per Cell: n = {result.sample_size_per_cell}
                </div>
                <div className="text-xl font-semibold text-gray-300">
                  {result.design.factor_a_levels} × {result.design.factor_b_levels} design ({result.num_cells} cells)
                </div>
                <div className="text-xl font-semibold text-gray-300">
                  Total: N = {result.total_sample_size}
                </div>
              </div>
            )}
          </div>

          {/* Interpretation */}
          <div className="bg-slate-700/30 rounded-lg p-4 mb-4">
            <h5 className="text-gray-200 font-semibold mb-2">Interpretation</h5>
            <p className="text-gray-300">{result.interpretation}</p>
          </div>

          {/* Parameters */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div className="bg-slate-700/30 rounded-lg p-4">
              <h5 className="text-gray-200 font-semibold mb-2">Effect Size</h5>
              <div className="space-y-1">
                <p className="text-gray-300">
                  {result.parameters.effect_size && `Cohen's ${calculatorType === 't-test' ? 'd' : 'f'}: `}
                  <span className="font-bold text-cyan-400">{result.parameters.effect_size}</span>
                </p>
                {result.eta_squared && (
                  <p className="text-gray-300">
                    η²: <span className="font-bold text-cyan-400">{result.eta_squared}</span>
                  </p>
                )}
                {result.parameters.adjusted_effect_size && (
                  <p className="text-gray-300 text-sm">
                    Adjusted: {result.parameters.adjusted_effect_size}
                  </p>
                )}
                <p className="text-gray-400 text-sm">
                  Classification: <span className="capitalize">{result.effect_size_classification.classification}</span>
                </p>
                <p className="text-gray-400 text-xs">
                  {result.effect_size_classification.description}
                </p>
              </div>
            </div>

            <div className="bg-slate-700/30 rounded-lg p-4">
              <h5 className="text-gray-200 font-semibold mb-2">Study Parameters</h5>
              <div className="space-y-1 text-gray-300">
                <p>Power: <span className="font-bold text-cyan-400">{(result.parameters.power * 100).toFixed(0)}%</span></p>
                <p>Significance (α): <span className="font-bold text-cyan-400">{result.parameters.alpha}</span></p>
                {result.parameters.alternative && (
                  <p>Test: <span className="font-bold text-cyan-400">{result.parameters.alternative}</span></p>
                )}
                {result.parameters.num_groups && (
                  <p>Groups: <span className="font-bold text-cyan-400">{result.parameters.num_groups}</span></p>
                )}
                {result.parameters.effect_of_interest && (
                  <p className="text-sm">Powered for: <span className="font-bold text-cyan-400">{result.parameters.effect_of_interest}</span></p>
                )}
              </div>
            </div>
          </div>

          {/* Recommendations */}
          <div className="bg-blue-900/20 border border-blue-700/50 rounded-lg p-4">
            <h5 className="text-gray-200 font-semibold mb-2">Recommendations</h5>
            <ul className="text-gray-300 space-y-1 text-sm list-disc list-inside">
              <li>Consider recruiting {Math.ceil(result.total_sample_size * 1.1)} participants (10% buffer for dropouts)</li>
              <li>Ensure your measurement instruments have adequate reliability</li>
              <li>Pre-register your study design and analysis plan</li>
              {result.effect_size_classification.classification === 'small' && (
                <li className="text-yellow-300">Small effect sizes require larger samples for adequate power</li>
              )}
              {result.anova_type === 'two-way' && (
                <li>Consider if the sample size is adequate for all effects of interest, not just the one selected</li>
              )}
            </ul>
          </div>
        </div>
      )}

      {/* Power Curve Visualization */}
      {result && (
        <PowerCurveChart
          calculatorType={calculatorType}
          testType={testType}
          anovaType={anovaType}
          alpha={alpha}
          alternative={alternative}
          effectSize={result.parameters.effect_size}
          calculatedSampleSize={
            result.sample_size ||
            result.sample_size_per_group ||
            result.sample_size_per_cell
          }
          calculatedPower={result.parameters.power}
          numGroups={numGroups}
          numLevelsA={numLevelsA}
          numLevelsB={numLevelsB}
          effectOfInterest={effectOfInterest}
          ratio={ratio}
          correlation={correlation}
        />
      )}

      {/* Effect Size Tools Section */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <div className="flex items-center space-x-3 mb-4">
          <Zap className="w-8 h-8 text-cyan-400" />
          <h2 className="text-3xl font-bold text-gray-100">Effect Size Tools</h2>
        </div>
        <p className="text-gray-300 mb-6">
          Estimate, convert, and analyze effect sizes for your research design.
        </p>

        {/* Tool Tabs */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <button
            onClick={() => setActiveToolTab('converter')}
            className={`p-4 rounded-lg border-2 transition-all ${
              activeToolTab === 'converter'
                ? 'border-cyan-500 bg-cyan-500/10'
                : 'border-slate-600 bg-slate-700/30 hover:bg-slate-700/50'
            }`}
          >
            <div className="flex items-center space-x-2 mb-2">
              <RefreshCw className="w-5 h-5 text-cyan-400" />
              <h3 className="text-lg font-semibold text-gray-100">Effect Size Converter</h3>
            </div>
            <p className="text-sm text-gray-400">Convert between d, f, η², r, and OR</p>
          </button>

          <button
            onClick={() => setActiveToolTab('pilot')}
            className={`p-4 rounded-lg border-2 transition-all ${
              activeToolTab === 'pilot'
                ? 'border-cyan-500 bg-cyan-500/10'
                : 'border-slate-600 bg-slate-700/30 hover:bg-slate-700/50'
            }`}
          >
            <div className="flex items-center space-x-2 mb-2">
              <Beaker className="w-5 h-5 text-cyan-400" />
              <h3 className="text-lg font-semibold text-gray-100">From Pilot Data</h3>
            </div>
            <p className="text-sm text-gray-400">Calculate effect size from pilot study</p>
          </button>

          <button
            onClick={() => setActiveToolTab('minimum')}
            className={`p-4 rounded-lg border-2 transition-all ${
              activeToolTab === 'minimum'
                ? 'border-cyan-500 bg-cyan-500/10'
                : 'border-slate-600 bg-slate-700/30 hover:bg-slate-700/50'
            }`}
          >
            <div className="flex items-center space-x-2 mb-2">
              <Target className="w-5 h-5 text-cyan-400" />
              <h3 className="text-lg font-semibold text-gray-100">Minimum Detectable</h3>
            </div>
            <p className="text-sm text-gray-400">Find smallest detectable effect</p>
          </button>
        </div>

        {/* Effect Size Converter */}
        {activeToolTab === 'converter' && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-gray-200 font-medium mb-2">From Metric</label>
                <select
                  value={fromMetric}
                  onChange={(e) => setFromMetric(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                >
                  <option value="cohens_d">Cohen's d</option>
                  <option value="cohens_f">Cohen's f</option>
                  <option value="eta_squared">Eta-squared (η²)</option>
                  <option value="r">Correlation (r)</option>
                  <option value="odds_ratio">Odds Ratio</option>
                </select>
              </div>

              <div>
                <label className="block text-gray-200 font-medium mb-2">Value</label>
                <input
                  type="number"
                  step="0.01"
                  value={converterValue}
                  onChange={(e) => setConverterValue(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                  placeholder="Enter value"
                />
              </div>

              <div>
                <label className="block text-gray-200 font-medium mb-2">To Metric</label>
                <select
                  value={toMetric}
                  onChange={(e) => setToMetric(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                >
                  <option value="cohens_d">Cohen's d</option>
                  <option value="cohens_f">Cohen's f</option>
                  <option value="eta_squared">Eta-squared (η²)</option>
                  <option value="r">Correlation (r)</option>
                  <option value="odds_ratio">Odds Ratio</option>
                </select>
              </div>
            </div>

            <button
              onClick={handleConvertEffectSize}
              disabled={converterLoading}
              className="w-full bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-bold py-3 px-6 rounded-lg hover:from-cyan-600 hover:to-blue-600 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {converterLoading ? 'Converting...' : 'Convert Effect Size'}
            </button>

            {converterResult && !converterResult.error && (
              <div className="bg-slate-700/30 rounded-lg p-4">
                <h4 className="text-gray-200 font-semibold mb-3">Conversion Results</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="bg-gradient-to-r from-cyan-500/20 to-blue-500/20 rounded-lg p-4">
                    <p className="text-gray-400 text-sm mb-1">Original</p>
                    <p className="text-2xl font-bold text-cyan-400">
                      {converterResult.original.metric.replace('_', ' ')}: {converterResult.original.value}
                    </p>
                  </div>
                  <div className="bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-lg p-4">
                    <p className="text-gray-400 text-sm mb-1">Converted</p>
                    <p className="text-2xl font-bold text-purple-400">
                      {converterResult.converted.metric.replace('_', ' ')}: {converterResult.converted.value}
                    </p>
                  </div>
                </div>

                <div className="mt-4 p-4 bg-slate-700/50 rounded-lg">
                  <h5 className="text-gray-200 font-semibold mb-2">All Equivalent Values</h5>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
                    <div>
                      <span className="text-gray-400">Cohen's d:</span>
                      <span className="text-gray-200 ml-2 font-medium">{converterResult.all_conversions.cohens_d}</span>
                      <p className="text-xs text-gray-500">{converterResult.interpretations.cohens_d}</p>
                    </div>
                    <div>
                      <span className="text-gray-400">Cohen's f:</span>
                      <span className="text-gray-200 ml-2 font-medium">{converterResult.all_conversions.cohens_f}</span>
                      <p className="text-xs text-gray-500">{converterResult.interpretations.cohens_f}</p>
                    </div>
                    <div>
                      <span className="text-gray-400">η²:</span>
                      <span className="text-gray-200 ml-2 font-medium">{converterResult.all_conversions.eta_squared}</span>
                      <p className="text-xs text-gray-500">{converterResult.interpretations.eta_squared}</p>
                    </div>
                    <div>
                      <span className="text-gray-400">r:</span>
                      <span className="text-gray-200 ml-2 font-medium">{converterResult.all_conversions.r}</span>
                    </div>
                    <div>
                      <span className="text-gray-400">Odds Ratio:</span>
                      <span className="text-gray-200 ml-2 font-medium">{converterResult.all_conversions.odds_ratio}</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {converterResult && converterResult.error && (
              <div className="bg-red-900/30 backdrop-blur-lg rounded-xl p-4 border border-red-700/50">
                <p className="text-red-200 font-medium">Error: {converterResult.error}</p>
              </div>
            )}
          </div>
        )}

        {/* Pilot Data Calculator */}
        {activeToolTab === 'pilot' && (
          <div className="space-y-4">
            <div>
              <label className="block text-gray-200 font-medium mb-2">Study Design</label>
              <select
                value={pilotTestType}
                onChange={(e) => setPilotTestType(e.target.value)}
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
              >
                <option value="independent">Independent Samples (Two Groups)</option>
                <option value="paired">Paired Samples (Before/After)</option>
                <option value="anova">ANOVA (Multiple Groups)</option>
              </select>
            </div>

            {pilotTestType === 'independent' && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-3 p-4 bg-slate-700/30 rounded-lg">
                  <h4 className="text-gray-200 font-semibold">Group 1</h4>
                  <div>
                    <label className="block text-gray-300 text-sm mb-1">Mean</label>
                    <input
                      type="number"
                      step="0.1"
                      value={pilotMean1}
                      onChange={(e) => setPilotMean1(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                    />
                  </div>
                  <div>
                    <label className="block text-gray-300 text-sm mb-1">Standard Deviation</label>
                    <input
                      type="number"
                      step="0.1"
                      value={pilotSd1}
                      onChange={(e) => setPilotSd1(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                    />
                  </div>
                  <div>
                    <label className="block text-gray-300 text-sm mb-1">Sample Size</label>
                    <input
                      type="number"
                      value={pilotN1}
                      onChange={(e) => setPilotN1(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                    />
                  </div>
                </div>

                <div className="space-y-3 p-4 bg-slate-700/30 rounded-lg">
                  <h4 className="text-gray-200 font-semibold">Group 2</h4>
                  <div>
                    <label className="block text-gray-300 text-sm mb-1">Mean</label>
                    <input
                      type="number"
                      step="0.1"
                      value={pilotMean2}
                      onChange={(e) => setPilotMean2(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                    />
                  </div>
                  <div>
                    <label className="block text-gray-300 text-sm mb-1">Standard Deviation</label>
                    <input
                      type="number"
                      step="0.1"
                      value={pilotSd2}
                      onChange={(e) => setPilotSd2(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                    />
                  </div>
                  <div>
                    <label className="block text-gray-300 text-sm mb-1">Sample Size</label>
                    <input
                      type="number"
                      value={pilotN2}
                      onChange={(e) => setPilotN2(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                    />
                  </div>
                </div>
              </div>
            )}

            {pilotTestType === 'paired' && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-gray-200 font-medium mb-2">Mean Difference</label>
                  <input
                    type="number"
                    step="0.1"
                    value={pilotMeanDiff}
                    onChange={(e) => setPilotMeanDiff(e.target.value)}
                    className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                  />
                  <p className="text-gray-400 text-xs mt-1">Average change from before to after</p>
                </div>
                <div>
                  <label className="block text-gray-200 font-medium mb-2">SD of Differences</label>
                  <input
                    type="number"
                    step="0.1"
                    value={pilotSdDiff}
                    onChange={(e) => setPilotSdDiff(e.target.value)}
                    className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                  />
                  <p className="text-gray-400 text-xs mt-1">Variability in the change scores</p>
                </div>
              </div>
            )}

            {pilotTestType === 'anova' && (
              <div className="space-y-3">
                <p className="text-gray-300 text-sm">Enter data for each group (comma-separated values)</p>
                <div>
                  <label className="block text-gray-200 font-medium mb-2">Group Means</label>
                  <input
                    type="text"
                    value={pilotGroupMeans.join(', ')}
                    onChange={(e) => setPilotGroupMeans(e.target.value.split(',').map(v => v.trim()))}
                    className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                    placeholder="e.g., 10, 12, 14"
                  />
                </div>
                <div>
                  <label className="block text-gray-200 font-medium mb-2">Group Standard Deviations</label>
                  <input
                    type="text"
                    value={pilotGroupSds.join(', ')}
                    onChange={(e) => setPilotGroupSds(e.target.value.split(',').map(v => v.trim()))}
                    className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                    placeholder="e.g., 2, 2, 2"
                  />
                </div>
                <div>
                  <label className="block text-gray-200 font-medium mb-2">Group Sample Sizes</label>
                  <input
                    type="text"
                    value={pilotGroupNs.join(', ')}
                    onChange={(e) => setPilotGroupNs(e.target.value.split(',').map(v => v.trim()))}
                    className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                    placeholder="e.g., 15, 15, 15"
                  />
                </div>
              </div>
            )}

            <button
              onClick={handleCalculatePilotEffectSize}
              disabled={pilotLoading}
              className="w-full bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-bold py-3 px-6 rounded-lg hover:from-cyan-600 hover:to-blue-600 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {pilotLoading ? 'Calculating...' : 'Calculate Effect Size'}
            </button>

            {pilotResult && !pilotResult.error && (
              <div className="bg-slate-700/30 rounded-lg p-4">
                <h4 className="text-gray-200 font-semibold mb-3">{pilotResult.test_type}</h4>

                <div className="bg-gradient-to-r from-cyan-500/20 to-blue-500/20 rounded-lg p-4 mb-4">
                  {pilotResult.cohens_d !== undefined && (
                    <div>
                      <p className="text-gray-400 text-sm mb-1">Cohen's d</p>
                      <p className="text-3xl font-bold text-cyan-400">{pilotResult.cohens_d}</p>
                      <p className="text-gray-300 mt-2">{pilotResult.interpretation}</p>
                    </div>
                  )}
                  {pilotResult.cohens_f !== undefined && (
                    <div className="space-y-2">
                      <div>
                        <p className="text-gray-400 text-sm mb-1">Cohen's f</p>
                        <p className="text-3xl font-bold text-cyan-400">{pilotResult.cohens_f}</p>
                        <p className="text-gray-300 text-sm">{pilotResult.interpretation_f}</p>
                      </div>
                      <div>
                        <p className="text-gray-400 text-sm mb-1">Eta-squared (η²)</p>
                        <p className="text-2xl font-bold text-purple-400">{pilotResult.eta_squared}</p>
                        <p className="text-gray-300 text-sm">{pilotResult.interpretation_eta_squared}</p>
                      </div>
                    </div>
                  )}
                </div>

                {pilotResult.confidence_interval_95 && (
                  <div className="bg-blue-900/20 border border-blue-700/50 rounded-lg p-3 mb-3">
                    <p className="text-gray-200 text-sm font-semibold mb-1">95% Confidence Interval</p>
                    <p className="text-gray-300">[{pilotResult.confidence_interval_95.lower}, {pilotResult.confidence_interval_95.upper}]</p>
                  </div>
                )}

                <div className="bg-yellow-900/20 border border-yellow-700/50 rounded-lg p-3">
                  <p className="text-yellow-200 text-sm">{pilotResult.note}</p>
                </div>
              </div>
            )}

            {pilotResult && pilotResult.error && (
              <div className="bg-red-900/30 backdrop-blur-lg rounded-xl p-4 border border-red-700/50">
                <p className="text-red-200 font-medium">Error: {pilotResult.error}</p>
              </div>
            )}

            {/* Forest Plot Visualization */}
            {pilotResult && !pilotResult.error && (
              <ForestPlot pilotResult={pilotResult} />
            )}
          </div>
        )}

        {/* Minimum Detectable Effect Size */}
        {activeToolTab === 'minimum' && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-gray-200 font-medium mb-2">Test Family</label>
                <select
                  value={minTestFamily}
                  onChange={(e) => {
                    setMinTestFamily(e.target.value)
                    setMinTestType(e.target.value === 't-test' ? 'one-sample' : 'one-way')
                  }}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                >
                  <option value="t-test">t-Test</option>
                  <option value="anova">ANOVA</option>
                </select>
              </div>

              <div>
                <label className="block text-gray-200 font-medium mb-2">Test Type</label>
                <select
                  value={minTestType}
                  onChange={(e) => setMinTestType(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                >
                  {minTestFamily === 't-test' ? (
                    <>
                      <option value="one-sample">One-Sample</option>
                      <option value="two-sample">Two-Sample</option>
                      <option value="paired">Paired</option>
                    </>
                  ) : (
                    <>
                      <option value="one-way">One-Way</option>
                      <option value="two-way">Two-Way</option>
                    </>
                  )}
                </select>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-gray-200 font-medium mb-2">
                  {minTestFamily === 't-test' && minTestType !== 'two-sample' ? 'Sample Size' :
                   minTestFamily === 't-test' ? 'Sample Size per Group' :
                   minTestType === 'one-way' ? 'Sample Size per Group' : 'Sample Size per Cell'}
                </label>
                <input
                  type="number"
                  min="2"
                  value={minSampleSize}
                  onChange={(e) => setMinSampleSize(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                />
              </div>

              <div>
                <label className="block text-gray-200 font-medium mb-2">Power</label>
                <input
                  type="number"
                  step="0.01"
                  min="0.5"
                  max="0.99"
                  value={minPower}
                  onChange={(e) => setMinPower(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                />
              </div>

              <div>
                <label className="block text-gray-200 font-medium mb-2">Alpha (α)</label>
                <select
                  value={minAlpha}
                  onChange={(e) => setMinAlpha(parseFloat(e.target.value))}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                >
                  <option value="0.001">0.001</option>
                  <option value="0.01">0.01</option>
                  <option value="0.05">0.05</option>
                  <option value="0.10">0.10</option>
                </select>
              </div>
            </div>

            {minTestFamily === 't-test' && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-gray-200 font-medium mb-2">Alternative Hypothesis</label>
                  <select
                    value={minAlternative}
                    onChange={(e) => setMinAlternative(e.target.value)}
                    className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                  >
                    <option value="two-sided">Two-Sided</option>
                    <option value="greater">Greater</option>
                    <option value="less">Less</option>
                  </select>
                </div>

                {minTestType === 'two-sample' && (
                  <div>
                    <label className="block text-gray-200 font-medium mb-2">Sample Size Ratio (n2/n1)</label>
                    <input
                      type="number"
                      step="0.1"
                      min="0.1"
                      max="10"
                      value={minRatio}
                      onChange={(e) => setMinRatio(e.target.value)}
                      className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                    />
                  </div>
                )}

                {minTestType === 'paired' && (
                  <div>
                    <label className="block text-gray-200 font-medium mb-2">Correlation Between Pairs</label>
                    <input
                      type="number"
                      step="0.01"
                      min="-0.99"
                      max="0.99"
                      value={minCorrelation}
                      onChange={(e) => setMinCorrelation(e.target.value)}
                      className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                    />
                  </div>
                )}
              </div>
            )}

            {minTestFamily === 'anova' && minTestType === 'one-way' && (
              <div>
                <label className="block text-gray-200 font-medium mb-2">Number of Groups</label>
                <input
                  type="number"
                  min="2"
                  max="20"
                  value={minNumGroups}
                  onChange={(e) => setMinNumGroups(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                />
              </div>
            )}

            {minTestFamily === 'anova' && minTestType === 'two-way' && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-gray-200 font-medium mb-2">Factor A Levels</label>
                  <input
                    type="number"
                    min="2"
                    max="10"
                    value={minNumLevelsA}
                    onChange={(e) => setMinNumLevelsA(e.target.value)}
                    className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                  />
                </div>
                <div>
                  <label className="block text-gray-200 font-medium mb-2">Factor B Levels</label>
                  <input
                    type="number"
                    min="2"
                    max="10"
                    value={minNumLevelsB}
                    onChange={(e) => setMinNumLevelsB(e.target.value)}
                    className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                  />
                </div>
              </div>
            )}

            <button
              onClick={handleCalculateMinimumEffectSize}
              disabled={minLoading}
              className="w-full bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-bold py-3 px-6 rounded-lg hover:from-cyan-600 hover:to-blue-600 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {minLoading ? 'Calculating...' : 'Calculate Minimum Effect Size'}
            </button>

            {minResult && !minResult.error && (
              <div className="bg-slate-700/30 rounded-lg p-4">
                <h4 className="text-gray-200 font-semibold mb-3">{minResult.test_type}</h4>

                <div className="bg-gradient-to-r from-cyan-500/20 to-blue-500/20 rounded-lg p-4 mb-4">
                  {minResult.minimum_effect_size !== undefined && (
                    <div>
                      <p className="text-gray-400 text-sm mb-1">Minimum Detectable Effect Size (Cohen's d)</p>
                      <p className="text-3xl font-bold text-cyan-400">{minResult.minimum_effect_size}</p>
                      <p className="text-gray-300 mt-2">{minResult.interpretation}</p>
                    </div>
                  )}
                  {minResult.minimum_effect_size_f !== undefined && (
                    <div className="space-y-3">
                      <div>
                        <p className="text-gray-400 text-sm mb-1">Minimum Cohen's f</p>
                        <p className="text-3xl font-bold text-cyan-400">{minResult.minimum_effect_size_f}</p>
                        <p className="text-gray-300 text-sm">{minResult.interpretation_f}</p>
                      </div>
                      <div>
                        <p className="text-gray-400 text-sm mb-1">Minimum Eta-squared (η²)</p>
                        <p className="text-2xl font-bold text-purple-400">{minResult.minimum_effect_size_eta_squared}</p>
                        <p className="text-gray-300 text-sm">{minResult.interpretation_eta_squared}</p>
                      </div>
                    </div>
                  )}
                </div>

                <div className="bg-slate-700/50 rounded-lg p-3">
                  <h5 className="text-gray-200 font-semibold mb-2">Study Parameters</h5>
                  <div className="grid grid-cols-2 gap-2 text-sm text-gray-300">
                    {minResult.parameters.sample_size && (
                      <div>Sample Size: <span className="font-bold text-cyan-400">{minResult.parameters.sample_size}</span></div>
                    )}
                    {minResult.parameters.sample_size_per_group && (
                      <div>Per Group: <span className="font-bold text-cyan-400">{minResult.parameters.sample_size_per_group}</span></div>
                    )}
                    {minResult.parameters.sample_size_per_cell && (
                      <div>Per Cell: <span className="font-bold text-cyan-400">{minResult.parameters.sample_size_per_cell}</span></div>
                    )}
                    <div>Power: <span className="font-bold text-cyan-400">{(minResult.parameters.power * 100).toFixed(0)}%</span></div>
                    <div>Alpha: <span className="font-bold text-cyan-400">{minResult.parameters.alpha}</span></div>
                    {minResult.parameters.num_groups && (
                      <div>Groups: <span className="font-bold text-cyan-400">{minResult.parameters.num_groups}</span></div>
                    )}
                    {minResult.sample_sizes && (
                      <div className="col-span-2">
                        Total N: <span className="font-bold text-cyan-400">{minResult.sample_sizes.total}</span>
                      </div>
                    )}
                    {minResult.parameters.total_sample_size && (
                      <div className="col-span-2">
                        Total N: <span className="font-bold text-cyan-400">{minResult.parameters.total_sample_size}</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {minResult && minResult.error && (
              <div className="bg-red-900/30 backdrop-blur-lg rounded-xl p-4 border border-red-700/50">
                <p className="text-red-200 font-medium">Error: {minResult.error}</p>
              </div>
            )}

            {/* Sensitivity Curve Visualization */}
            {minResult && !minResult.error && (
              <SensitivityCurve
                testFamily={minTestFamily}
                testType={minTestType}
                sampleSize={minSampleSize}
                power={minPower}
                alpha={minAlpha}
                alternative={minAlternative}
                ratio={minRatio}
                correlation={minCorrelation}
                numGroups={minNumGroups}
                numLevelsA={minNumLevelsA}
                numLevelsB={minNumLevelsB}
              />
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default ExperimentPlanning
