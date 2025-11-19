import { useState } from 'react'
import { Layers, Download, PieChart, BarChart3, Table2 } from 'lucide-react'
import axios from 'axios'
import InteractionPlot from '../components/InteractionPlot'
import MainEffectsPlot from '../components/MainEffectsPlot'
import ResidualPlots from '../components/ResidualPlots'
import BoxPlot from '../components/BoxPlot'
import VarianceComponentsChart from '../components/VarianceComponentsChart'
import HierarchicalMeansPlot from '../components/HierarchicalMeansPlot'
import NestedBoxPlots from '../components/NestedBoxPlots'
import ProfilePlot from '../components/ProfilePlot'
import WithinSubjectVariabilityPlot from '../components/WithinSubjectVariabilityPlot'
import ICCDisplay from '../components/ICCDisplay'
import ModelComparisonTable from '../components/ModelComparisonTable'
import VarianceDecomposition from '../components/VarianceDecomposition'
import BLUPsPlot from '../components/BLUPsPlot'
import RandomEffectsQQPlot from '../components/RandomEffectsQQPlot'
import GrowthCurvePlot from '../components/GrowthCurvePlot'
import GrowthCurveResults from '../components/GrowthCurveResults'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const MixedModels = () => {
  const [analysisType, setAnalysisType] = useState('mixed-anova')

  // Mixed ANOVA state
  const [mixedTableData, setMixedTableData] = useState(
    Array(15).fill(null).map(() => Array(3).fill('')) // Factor A, Factor B, Response
  )
  const [mixedFactorNames, setMixedFactorNames] = useState(['Treatment', 'Subject'])
  const [factorTypes, setFactorTypes] = useState({
    'Treatment': 'fixed',
    'Subject': 'random'
  })

  // Split-Plot state
  const [splitPlotTableData, setSplitPlotTableData] = useState(
    Array(15).fill(null).map(() => Array(4).fill('')) // Block, Whole-plot, Sub-plot, Response
  )
  const [splitPlotFactorNames, setSplitPlotFactorNames] = useState(['Irrigation', 'Variety'])
  const [blockName, setBlockName] = useState('Block')
  const [includeBlocks, setIncludeBlocks] = useState(true)

  // Nested Design state
  const [nestedTableData, setNestedTableData] = useState(
    Array(15).fill(null).map(() => Array(3).fill('')) // Factor A, Factor B (nested), Response
  )
  const [nestedFactorNames, setNestedFactorNames] = useState(['School', 'Teacher'])

  // Repeated Measures state
  const [repeatedTableData, setRepeatedTableData] = useState(
    Array(15).fill(null).map(() => Array(3).fill('')) // Subject, Condition, Response
  )
  const [subjectName, setSubjectName] = useState('Subject')
  const [withinFactorName, setWithinFactorName] = useState('Time')

  // Growth Curve state
  const [growthCurveTableData, setGrowthCurveTableData] = useState(
    Array(20).fill(null).map(() => Array(3).fill('')) // SubjectID, Time, Response
  )
  const [growthSubjectID, setGrowthSubjectID] = useState('SubjectID')
  const [growthTimeVar, setGrowthTimeVar] = useState('Time')
  const [polynomialOrder, setPolynomialOrder] = useState('linear')
  const [randomEffectsStructure, setRandomEffectsStructure] = useState('intercept_slope')

  // Shared state
  const [responseName, setResponseName] = useState('Response')
  const [includeInteractions, setIncludeInteractions] = useState(true)
  const [alpha, setAlpha] = useState(0.05)

  // Results
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Get current data based on analysis type
  const tableData = analysisType === 'mixed-anova' ? mixedTableData :
                    analysisType === 'split-plot' ? splitPlotTableData :
                    analysisType === 'nested' ? nestedTableData :
                    analysisType === 'growth-curve' ? growthCurveTableData : repeatedTableData
  const setTableData = analysisType === 'mixed-anova' ? setMixedTableData :
                       analysisType === 'split-plot' ? setSplitPlotTableData :
                       analysisType === 'nested' ? setNestedTableData :
                       analysisType === 'growth-curve' ? setGrowthCurveTableData : setRepeatedTableData
  const factorNames = analysisType === 'mixed-anova' ? mixedFactorNames :
                      analysisType === 'split-plot' ? splitPlotFactorNames : nestedFactorNames
  const setFactorNames = analysisType === 'mixed-anova' ? setMixedFactorNames :
                         analysisType === 'split-plot' ? setSplitPlotFactorNames : setNestedFactorNames
  const numColumns = analysisType === 'mixed-anova' ? 3 :
                     analysisType === 'split-plot' ? 4 : 3

  // Handle cell changes
  const handleCellChange = (rowIndex, colIndex, value) => {
    const newData = [...tableData]
    newData[rowIndex][colIndex] = value
    setTableData(newData)

    // Auto-add row if typing in last row
    if (rowIndex === tableData.length - 1 && value.trim() !== '') {
      setTableData([...newData, Array(numColumns).fill('')])
    }
  }

  // Handle factor name change
  const handleFactorNameChange = (index, value) => {
    const newFactorNames = [...factorNames]
    const oldName = newFactorNames[index]
    newFactorNames[index] = value
    setFactorNames(newFactorNames)

    // Update factor types with new name
    if (factorTypes[oldName] !== undefined) {
      const newFactorTypes = { ...factorTypes }
      newFactorTypes[value] = newFactorTypes[oldName]
      delete newFactorTypes[oldName]
      setFactorTypes(newFactorTypes)
    } else {
      setFactorTypes({ ...factorTypes, [value]: 'fixed' })
    }
  }

  // Handle factor type change
  const handleFactorTypeChange = (factorName, type) => {
    setFactorTypes({ ...factorTypes, [factorName]: type })
  }

  // Convert table data to API format
  const prepareData = () => {
    if (analysisType === 'mixed-anova') {
      return tableData
        .filter(row => row[0] && row[1] && row[2])
        .map(row => ({
          [factorNames[0]]: row[0],
          [factorNames[1]]: row[1],
          [responseName]: parseFloat(row[2])
        }))
    } else if (analysisType === 'split-plot') {
      // Split-plot
      return tableData
        .filter(row => {
          if (includeBlocks) {
            return row[0] && row[1] && row[2] && row[3]
          } else {
            return row[1] && row[2] && row[3]
          }
        })
        .map(row => {
          const dataRow = {
            [factorNames[0]]: row[1], // Whole-plot factor
            [factorNames[1]]: row[2], // Sub-plot factor
            [responseName]: parseFloat(row[3])
          }
          if (includeBlocks) {
            dataRow[blockName] = row[0]
          }
          return dataRow
        })
    } else if (analysisType === 'nested') {
      // Nested design
      return tableData
        .filter(row => row[0] && row[1] && row[2])
        .map(row => ({
          [factorNames[0]]: row[0], // Factor A (higher level)
          [factorNames[1]]: row[1], // Factor B (nested in A)
          [responseName]: parseFloat(row[2])
        }))
    } else if (analysisType === 'repeated-measures') {
      // Repeated measures
      return tableData
        .filter(row => row[0] && row[1] && row[2])
        .map(row => ({
          [subjectName]: row[0], // Subject ID
          [withinFactorName]: row[1], // Within-subjects factor (Time, Condition, etc.)
          [responseName]: parseFloat(row[2])
        }))
    } else if (analysisType === 'growth-curve') {
      // Growth curve - use growthCurveTableData
      return growthCurveTableData
        .filter(row => row[0] && row[1] && row[2])
        .map(row => ({
          [growthSubjectID]: row[0], // Subject ID
          [growthTimeVar]: parseFloat(row[1]), // Time (numeric)
          [responseName]: parseFloat(row[2]) // Response
        }))
    }
  }

  // Run analysis
  const runAnalysis = async () => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const data = prepareData()

      if (data.length === 0) {
        throw new Error('Please enter data in the table')
      }

      let response
      if (analysisType === 'mixed-anova') {
        // Separate fixed and random factors
        const fixedFactors = factorNames.filter(f => factorTypes[f] === 'fixed')
        const randomFactors = factorNames.filter(f => factorTypes[f] === 'random')

        const payload = {
          data: data,
          fixed_factors: fixedFactors,
          random_factors: randomFactors,
          response: responseName,
          alpha: alpha,
          include_interactions: includeInteractions
        }

        response = await axios.post(`${API_URL}/api/mixed/mixed-model-anova`, payload)
      } else if (analysisType === 'split-plot') {
        // Split-plot
        const payload = {
          data: data,
          whole_plot_factor: factorNames[0],
          subplot_factor: factorNames[1],
          block: includeBlocks ? blockName : null,
          response: responseName,
          alpha: alpha
        }

        response = await axios.post(`${API_URL}/api/mixed/split-plot`, payload)
      } else if (analysisType === 'nested') {
        // Nested design
        const payload = {
          data: data,
          factor_a: factorNames[0],
          factor_b_nested: factorNames[1],
          response: responseName,
          alpha: alpha
        }

        response = await axios.post(`${API_URL}/api/mixed/nested-design`, payload)
      } else if (analysisType === 'repeated-measures') {
        // Repeated measures
        const payload = {
          data: data,
          subject: subjectName,
          within_factor: withinFactorName,
          response: responseName,
          alpha: alpha
        }

        response = await axios.post(`${API_URL}/api/mixed/repeated-measures`, payload)
      } else if (analysisType === 'growth-curve') {
        // Growth curve
        const payload = {
          data: data,
          subject_id: growthSubjectID,
          time_var: growthTimeVar,
          response: responseName,
          polynomial_order: polynomialOrder,
          random_effects: randomEffectsStructure,
          alpha: alpha
        }

        response = await axios.post(`${API_URL}/api/mixed/growth-curve`, payload)
      }

      setResult(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  // Load example data
  // Generate random number from normal distribution using Box-Muller transform
  const randomNormal = (mean = 50, stdDev = 5) => {
    const u1 = Math.random()
    const u2 = Math.random()
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2)
    return mean + z0 * stdDev
  }

  const loadExampleData = () => {
    if (analysisType === 'mixed-anova') {
      // Generate random data for Mixed Model ANOVA
      // Design: 2 treatments × 3 subjects × 2 replicates
      const treatments = ['A1', 'A2']
      const subjects = ['S1', 'S2', 'S3']
      const replicates = 2

      const exampleData = []

      // Generate data with treatment and subject effects
      treatments.forEach((treatment, tIdx) => {
        subjects.forEach((subject, sIdx) => {
          // Base mean for this treatment
          const treatmentEffect = tIdx * 5 // A2 has higher response than A1

          // Random subject effect
          const subjectEffect = (sIdx - 1) * 2

          // Generate replicates
          for (let rep = 0; rep < replicates; rep++) {
            const baseMean = 50 + treatmentEffect + subjectEffect
            const response = randomNormal(baseMean, 1.5)
            exampleData.push([
              treatment,
              subject,
              response.toFixed(1)
            ])
          }
        })
      })

      const newTableData = [...exampleData, ...Array(Math.max(0, 15 - exampleData.length)).fill(null).map(() => Array(3).fill(''))]
      setMixedTableData(newTableData)
      setMixedFactorNames(['Treatment', 'Subject'])
      setFactorTypes({ 'Treatment': 'fixed', 'Subject': 'random' })
      setResponseName('Response')
    } else if (analysisType === 'split-plot') {
      // Generate random data for Split-Plot Design
      // Design: 3 blocks × 2 irrigation levels × 3 varieties
      const blocks = ['B1', 'B2', 'B3']
      const irrigations = ['I1', 'I2']
      const varieties = ['V1', 'V2', 'V3']

      const exampleData = []

      blocks.forEach((block, bIdx) => {
        irrigations.forEach((irrigation, iIdx) => {
          varieties.forEach((variety, vIdx) => {
            // Whole-plot effect (irrigation)
            const irrigationEffect = iIdx * 7 // I2 gives higher yield

            // Sub-plot effect (variety)
            const varietyEffect = (vIdx - 1) * 3 // V2 highest, V3 lowest

            // Interaction effect
            const interactionEffect = iIdx === 1 && vIdx === 1 ? 2 : 0

            // Block effect (small random variation)
            const blockEffect = (bIdx - 1) * 0.5

            const baseMean = 45 + irrigationEffect + varietyEffect + interactionEffect + blockEffect
            const response = randomNormal(baseMean, 0.8)

            exampleData.push([
              block,
              irrigation,
              variety,
              response.toFixed(1)
            ])
          })
        })
      })

      const newTableData = [...exampleData, ...Array(Math.max(0, 15 - exampleData.length)).fill(null).map(() => Array(4).fill(''))]
      setSplitPlotTableData(newTableData)
      setSplitPlotFactorNames(['Irrigation', 'Variety'])
      setBlockName('Block')
      setIncludeBlocks(true)
      setResponseName('Yield')
    } else if (analysisType === 'nested') {
      // Generate random data for Nested Design
      // Design: 3 schools × 4 teachers per school × 3 students per teacher
      const schools = ['S1', 'S2', 'S3']
      const teachersPerSchool = 4
      const studentsPerTeacher = 3

      const exampleData = []

      schools.forEach((school, sIdx) => {
        for (let t = 1; t <= teachersPerSchool; t++) {
          const teacher = `T${t}` // Teacher labels are unique within each school

          // School effect
          const schoolEffect = (sIdx - 1) * 5

          // Teacher within school effect
          const teacherEffect = (t - 2) * 2

          for (let st = 0; st < studentsPerTeacher; st++) {
            const baseMean = 75 + schoolEffect + teacherEffect
            const response = randomNormal(baseMean, 3)

            exampleData.push([
              school,
              teacher,
              response.toFixed(1)
            ])
          }
        }
      })

      const newTableData = [...exampleData, ...Array(Math.max(0, 15 - exampleData.length)).fill(null).map(() => Array(3).fill(''))]
      setNestedTableData(newTableData)
      setNestedFactorNames(['School', 'Teacher'])
      setResponseName('Score')
    } else {
      // Generate random data for Repeated Measures ANOVA
      // Design: 5 subjects × 4 time points with systematic increase
      const subjects = ['S1', 'S2', 'S3', 'S4', 'S5']
      const timePoints = ['T1', 'T2', 'T3', 'T4']

      const exampleData = []

      subjects.forEach((subject, sIdx) => {
        // Random baseline for each subject
        const subjectBaseline = 60 + (sIdx - 2) * 4

        timePoints.forEach((time, tIdx) => {
          // Time effect: systematic increase over time
          const timeEffect = tIdx * 5

          // Within-subject variability
          const baseMean = subjectBaseline + timeEffect
          const response = randomNormal(baseMean, 2)

          exampleData.push([
            subject,
            time,
            response.toFixed(1)
          ])
        })
      })

      const newTableData = [...exampleData, ...Array(Math.max(0, 15 - exampleData.length)).fill(null).map(() => Array(3).fill(''))]
      setRepeatedTableData(newTableData)
      setSubjectName('Subject')
      setWithinFactorName('Time')
      setResponseName('Score')
    } else if (analysisType === 'growth-curve') {
      // Generate random growth curve data
      // Design: 8 subjects × 5 time points with varying growth rates
      const subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
      const times = [0, 1, 2, 3, 4]

      const exampleData = []

      subjects.forEach((subject, sIdx) => {
        // Random baseline for each subject (initial value at time 0)
        const baseline = 20 + (sIdx - 3.5) * 3

        // Random growth rate for each subject
        const growthRate = 2 + (sIdx % 4) * 0.5 + randomNormal(0, 0.3)

        times.forEach((time) => {
          // Linear growth with individual variation
          const expected = baseline + growthRate * time
          const response = randomNormal(expected, 0.8)

          exampleData.push([
            subject,
            time.toString(),
            response.toFixed(2)
          ])
        })
      })

      const newTableData = [...exampleData, ...Array(Math.max(0, 20 - exampleData.length)).fill(null).map(() => Array(3).fill(''))]
      setGrowthCurveTableData(newTableData)
      setGrowthSubjectID('SubjectID')
      setGrowthTimeVar('Time')
      setResponseName('Value')
      setPolynomialOrder('linear')
      setRandomEffectsStructure('intercept_slope')
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <div className="flex items-center space-x-3 mb-4">
          <Layers className="w-8 h-8 text-indigo-400" />
          <h2 className="text-3xl font-bold text-gray-100">Mixed Models</h2>
        </div>
        <p className="text-gray-300">
          Analyze designs with both fixed and random effects. Includes Expected Mean Squares (EMS) and variance component estimation.
        </p>
      </div>

      {/* Analysis Type Selection */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <h3 className="text-xl font-bold text-gray-100 mb-4">Analysis Type</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <button
            onClick={() => {
              setAnalysisType('mixed-anova')
              setResult(null)
              setError(null)
            }}
            className={`p-4 rounded-lg border-2 transition ${
              analysisType === 'mixed-anova'
                ? 'border-indigo-500 bg-indigo-500/20'
                : 'border-slate-600 hover:border-indigo-400'
            }`}
          >
            <h4 className="text-lg font-semibold text-gray-100 mb-2">Mixed Model ANOVA</h4>
            <p className="text-sm text-gray-300">
              Analyze designs with fixed and random factors. Tests each factor with appropriate error terms.
            </p>
          </button>
          <button
            onClick={() => {
              setAnalysisType('split-plot')
              setResult(null)
              setError(null)
            }}
            className={`p-4 rounded-lg border-2 transition ${
              analysisType === 'split-plot'
                ? 'border-indigo-500 bg-indigo-500/20'
                : 'border-slate-600 hover:border-indigo-400'
            }`}
          >
            <h4 className="text-lg font-semibold text-gray-100 mb-2">Split-Plot Design</h4>
            <p className="text-sm text-gray-300">
              Hierarchical design with whole-plot and sub-plot factors. Uses two error terms.
            </p>
          </button>
          <button
            onClick={() => {
              setAnalysisType('nested')
              setResult(null)
              setError(null)
            }}
            className={`p-4 rounded-lg border-2 transition ${
              analysisType === 'nested'
                ? 'border-indigo-500 bg-indigo-500/20'
                : 'border-slate-600 hover:border-indigo-400'
            }`}
          >
            <h4 className="text-lg font-semibold text-gray-100 mb-2">Nested Design</h4>
            <p className="text-sm text-gray-300">
              Factor B nested within Factor A. Analyzes hierarchical structures like students within schools.
            </p>
          </button>
          <button
            onClick={() => {
              setAnalysisType('repeated-measures')
              setResult(null)
              setError(null)
            }}
            className={`p-4 rounded-lg border-2 transition ${
              analysisType === 'repeated-measures'
                ? 'border-indigo-500 bg-indigo-500/20'
                : 'border-slate-600 hover:border-indigo-400'
            }`}
          >
            <h4 className="text-lg font-semibold text-gray-100 mb-2">Repeated Measures ANOVA</h4>
            <p className="text-sm text-gray-300">
              Within-subjects design with multiple measurements per subject. Includes sphericity tests and corrections.
            </p>
          </button>
          <button
            onClick={() => {
              setAnalysisType('growth-curve')
              setResult(null)
              setError(null)
            }}
            className={`p-4 rounded-lg border-2 transition ${
              analysisType === 'growth-curve'
                ? 'border-emerald-500 bg-emerald-500/20'
                : 'border-slate-600 hover:border-emerald-400'
            }`}
          >
            <h4 className="text-lg font-semibold text-gray-100 mb-2">Growth Curve Model</h4>
            <p className="text-sm text-gray-300">
              Longitudinal data with linear, quadratic, or cubic trends. Random intercepts and slopes for individual trajectories.
            </p>
          </button>
        </div>
      </div>

      {/* Data Entry Section */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <h3 className="text-xl font-bold text-gray-100 mb-4">Data Entry</h3>

        {/* Factor Configuration */}
        <div className="mb-6 space-y-4">
          {analysisType === 'mixed-anova' ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Factor A */}
              <div className="bg-slate-700/30 rounded-lg p-4">
                <label className="block text-gray-200 font-medium mb-2">Factor A</label>
              <input
                type="text"
                value={factorNames[0]}
                onChange={(e) => handleFactorNameChange(0, e.target.value)}
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500 mb-3"
                placeholder="e.g., Treatment"
              />
              <div className="flex space-x-4">
                <label className="flex items-center space-x-2 text-gray-300">
                  <input
                    type="radio"
                    checked={factorTypes[factorNames[0]] === 'fixed'}
                    onChange={() => handleFactorTypeChange(factorNames[0], 'fixed')}
                    className="w-4 h-4"
                  />
                  <span>Fixed Effect</span>
                </label>
                <label className="flex items-center space-x-2 text-gray-300">
                  <input
                    type="radio"
                    checked={factorTypes[factorNames[0]] === 'random'}
                    onChange={() => handleFactorTypeChange(factorNames[0], 'random')}
                    className="w-4 h-4"
                  />
                  <span>Random Effect</span>
                </label>
              </div>
              <p className="text-xs text-gray-400 mt-2">
                {factorTypes[factorNames[0]] === 'fixed'
                  ? 'Fixed: Specific chosen levels'
                  : 'Random: Sampled from population'}
              </p>
            </div>

            {/* Factor B */}
            <div className="bg-slate-700/30 rounded-lg p-4">
              <label className="block text-gray-200 font-medium mb-2">Factor B</label>
              <input
                type="text"
                value={factorNames[1]}
                onChange={(e) => handleFactorNameChange(1, e.target.value)}
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500 mb-3"
                placeholder="e.g., Subject"
              />
              <div className="flex space-x-4">
                <label className="flex items-center space-x-2 text-gray-300">
                  <input
                    type="radio"
                    checked={factorTypes[factorNames[1]] === 'fixed'}
                    onChange={() => handleFactorTypeChange(factorNames[1], 'fixed')}
                    className="w-4 h-4"
                  />
                  <span>Fixed Effect</span>
                </label>
                <label className="flex items-center space-x-2 text-gray-300">
                  <input
                    type="radio"
                    checked={factorTypes[factorNames[1]] === 'random'}
                    onChange={() => handleFactorTypeChange(factorNames[1], 'random')}
                    className="w-4 h-4"
                  />
                  <span>Random Effect</span>
                </label>
              </div>
              <p className="text-xs text-gray-400 mt-2">
                {factorTypes[factorNames[1]] === 'fixed'
                  ? 'Fixed: Specific chosen levels'
                  : 'Random: Sampled from population'}
              </p>
            </div>
          </div>
          ) : analysisType === 'split-plot' ? (
            /* Split-Plot Configuration */
            <div className="space-y-4">
              {/* Include Blocks */}
              <div className="bg-slate-700/30 rounded-lg p-4">
                <label className="flex items-center space-x-2 text-gray-200">
                  <input
                    type="checkbox"
                    checked={includeBlocks}
                    onChange={(e) => setIncludeBlocks(e.target.checked)}
                    className="w-4 h-4"
                  />
                  <span className="font-medium">Include Blocks/Replicates</span>
                </label>
                <p className="text-xs text-gray-400 mt-2">
                  Check if you have blocks (replicates) in your design. This uses RCBD at whole-plot level.
                </p>
                {includeBlocks && (
                  <input
                    type="text"
                    value={blockName}
                    onChange={(e) => setBlockName(e.target.value)}
                    className="w-full mt-3 px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    placeholder="Block name (e.g., Block, Replicate)"
                  />
                )}
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Whole-plot Factor */}
                <div className="bg-slate-700/30 rounded-lg p-4">
                  <label className="block text-gray-200 font-medium mb-2">Whole-Plot Factor</label>
                  <input
                    type="text"
                    value={factorNames[0]}
                    onChange={(e) => handleFactorNameChange(0, e.target.value)}
                    className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    placeholder="e.g., Irrigation"
                  />
                  <p className="text-xs text-gray-400 mt-2">
                    Applied to large experimental units (hard to change)
                  </p>
                </div>

                {/* Sub-plot Factor */}
                <div className="bg-slate-700/30 rounded-lg p-4">
                  <label className="block text-gray-200 font-medium mb-2">Sub-Plot Factor</label>
                  <input
                    type="text"
                    value={factorNames[1]}
                    onChange={(e) => handleFactorNameChange(1, e.target.value)}
                    className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    placeholder="e.g., Variety"
                  />
                  <p className="text-xs text-gray-400 mt-2">
                    Applied within whole-plots (easy to change)
                  </p>
                </div>
              </div>
            </div>
          ) : analysisType === 'nested' ? (
            /* Nested Design Configuration */
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Factor A (Higher level) */}
              <div className="bg-slate-700/30 rounded-lg p-4">
                <label className="block text-gray-200 font-medium mb-2">Factor A (Higher Level)</label>
                <input
                  type="text"
                  value={factorNames[0]}
                  onChange={(e) => handleFactorNameChange(0, e.target.value)}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  placeholder="e.g., School"
                />
                <p className="text-xs text-gray-400 mt-2">
                  The grouping factor (e.g., schools, clinics, fields)
                </p>
              </div>

              {/* Factor B (Nested in A) */}
              <div className="bg-slate-700/30 rounded-lg p-4">
                <label className="block text-gray-200 font-medium mb-2">Factor B (Nested in A)</label>
                <input
                  type="text"
                  value={factorNames[1]}
                  onChange={(e) => handleFactorNameChange(1, e.target.value)}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  placeholder="e.g., Teacher"
                />
                <p className="text-xs text-gray-400 mt-2">
                  Nested within each level of Factor A (e.g., teachers within schools)
                </p>
              </div>
            </div>
          ) : analysisType === 'growth-curve' ? (
            /* Growth Curve Configuration */
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Subject Identifier */}
                <div className="bg-slate-700/30 rounded-lg p-4">
                  <label className="block text-gray-200 font-medium mb-2">Subject Identifier</label>
                  <input
                    type="text"
                    value={growthSubjectID}
                    onChange={(e) => setGrowthSubjectID(e.target.value)}
                    className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    placeholder="e.g., SubjectID, PatientID"
                  />
                  <p className="text-xs text-gray-400 mt-2">
                    Column identifying each individual/subject
                  </p>
                </div>

                {/* Time Variable */}
                <div className="bg-slate-700/30 rounded-lg p-4">
                  <label className="block text-gray-200 font-medium mb-2">Time Variable</label>
                  <input
                    type="text"
                    value={growthTimeVar}
                    onChange={(e) => setGrowthTimeVar(e.target.value)}
                    className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    placeholder="e.g., Time, Age, Day"
                  />
                  <p className="text-xs text-gray-400 mt-2">
                    Continuous time variable (numeric values)
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Polynomial Order */}
                <div className="bg-slate-700/30 rounded-lg p-4">
                  <label className="block text-gray-200 font-medium mb-3">Polynomial Order for Time</label>
                  <div className="space-y-2">
                    <label className="flex items-center space-x-2 text-gray-300 cursor-pointer">
                      <input
                        type="radio"
                        checked={polynomialOrder === 'linear'}
                        onChange={() => setPolynomialOrder('linear')}
                        className="w-4 h-4 text-emerald-500"
                      />
                      <span>Linear</span>
                    </label>
                    <label className="flex items-center space-x-2 text-gray-300 cursor-pointer">
                      <input
                        type="radio"
                        checked={polynomialOrder === 'quadratic'}
                        onChange={() => setPolynomialOrder('quadratic')}
                        className="w-4 h-4 text-emerald-500"
                      />
                      <span>Quadratic</span>
                    </label>
                    <label className="flex items-center space-x-2 text-gray-300 cursor-pointer">
                      <input
                        type="radio"
                        checked={polynomialOrder === 'cubic'}
                        onChange={() => setPolynomialOrder('cubic')}
                        className="w-4 h-4 text-emerald-500"
                      />
                      <span>Cubic</span>
                    </label>
                  </div>
                  <p className="text-xs text-gray-400 mt-2">
                    {polynomialOrder === 'linear' && 'Straight line growth'}
                    {polynomialOrder === 'quadratic' && 'Curved growth (acceleration/deceleration)'}
                    {polynomialOrder === 'cubic' && 'S-shaped or complex growth patterns'}
                  </p>
                </div>

                {/* Random Effects Structure */}
                <div className="bg-slate-700/30 rounded-lg p-4">
                  <label className="block text-gray-200 font-medium mb-3">Random Effects Structure</label>
                  <div className="space-y-2">
                    <label className="flex items-center space-x-2 text-gray-300 cursor-pointer">
                      <input
                        type="radio"
                        checked={randomEffectsStructure === 'intercept'}
                        onChange={() => setRandomEffectsStructure('intercept')}
                        className="w-4 h-4 text-emerald-500"
                      />
                      <span>Intercept Only</span>
                    </label>
                    <label className="flex items-center space-x-2 text-gray-300 cursor-pointer">
                      <input
                        type="radio"
                        checked={randomEffectsStructure === 'intercept_slope'}
                        onChange={() => setRandomEffectsStructure('intercept_slope')}
                        className="w-4 h-4 text-emerald-500"
                      />
                      <span>Intercept + Slope</span>
                    </label>
                  </div>
                  <p className="text-xs text-gray-400 mt-2">
                    {randomEffectsStructure === 'intercept' && 'Subjects differ in baseline only (parallel trajectories)'}
                    {randomEffectsStructure === 'intercept_slope' && 'Subjects differ in baseline AND growth rate (non-parallel)'}
                  </p>
                </div>
              </div>
            </div>
          ) : (
            /* Repeated Measures Configuration */
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Subject Identifier */}
              <div className="bg-slate-700/30 rounded-lg p-4">
                <label className="block text-gray-200 font-medium mb-2">Subject Identifier</label>
                <input
                  type="text"
                  value={subjectName}
                  onChange={(e) => setSubjectName(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  placeholder="e.g., Subject, Participant"
                />
                <p className="text-xs text-gray-400 mt-2">
                  The column identifying each participant/subject
                </p>
              </div>

              {/* Within-Subjects Factor */}
              <div className="bg-slate-700/30 rounded-lg p-4">
                <label className="block text-gray-200 font-medium mb-2">Within-Subjects Factor</label>
                <input
                  type="text"
                  value={withinFactorName}
                  onChange={(e) => setWithinFactorName(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  placeholder="e.g., Time, Condition"
                />
                <p className="text-xs text-gray-400 mt-2">
                  The repeated factor (e.g., time points, conditions measured within each subject)
                </p>
              </div>
            </div>
          )}

          {/* Response Variable Name */}
          <div>
            <label className="block text-gray-200 font-medium mb-2">Response Variable Name</label>
            <input
              type="text"
              value={responseName}
              onChange={(e) => setResponseName(e.target.value)}
              className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              placeholder="e.g., Yield"
            />
          </div>
        </div>

        {/* Excel-like Table */}
        <div className="mb-4">
          <div className="flex items-center justify-between mb-2">
            <label className="text-gray-200 font-medium">Data Table</label>
            <button
              onClick={loadExampleData}
              className="px-3 py-1 text-sm bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg transition"
            >
              Load Example Data
            </button>
          </div>
          <div className="overflow-x-auto bg-slate-900/50 rounded-lg border border-slate-600">
            <table className="w-full">
              <thead>
                <tr className="bg-slate-700/50">
                  <th className="px-4 py-2 text-left text-gray-200 font-medium">#</th>
                  {analysisType === 'split-plot' && includeBlocks && (
                    <th className="px-4 py-2 text-left text-gray-200 font-medium">{blockName}</th>
                  )}
                  <th className="px-4 py-2 text-left text-gray-200 font-medium">
                    {analysisType === 'split-plot' ? `${factorNames[0]} (WP)` :
                     analysisType === 'nested' ? `${factorNames[0]}` :
                     analysisType === 'repeated-measures' ? subjectName :
                     analysisType === 'growth-curve' ? growthSubjectID :
                     factorNames[0]}
                  </th>
                  <th className="px-4 py-2 text-left text-gray-200 font-medium">
                    {analysisType === 'split-plot' ? `${factorNames[1]} (SP)` :
                     analysisType === 'nested' ? `${factorNames[1]}(${factorNames[0]})` :
                     analysisType === 'repeated-measures' ? withinFactorName :
                     analysisType === 'growth-curve' ? growthTimeVar :
                     factorNames[1]}
                  </th>
                  <th className="px-4 py-2 text-left text-gray-200 font-medium">{responseName}</th>
                </tr>
              </thead>
              <tbody>
                {tableData.map((row, rowIndex) => (
                  <tr key={rowIndex} className="border-t border-slate-700 hover:bg-slate-700/30">
                    <td className="px-4 py-2 text-gray-400">{rowIndex + 1}</td>
                    {analysisType === 'split-plot' && includeBlocks && (
                      <td className="px-2 py-1">
                        <input
                          type="text"
                          value={row[0]}
                          onChange={(e) => handleCellChange(rowIndex, 0, e.target.value)}
                          className="w-full px-2 py-1 bg-slate-700/50 text-gray-100 border border-slate-600 rounded focus:outline-none focus:ring-1 focus:ring-indigo-500"
                          placeholder="B1"
                        />
                      </td>
                    )}
                    <td className="px-2 py-1">
                      <input
                        type="text"
                        value={analysisType === 'split-plot' && includeBlocks ? row[1] : row[0]}
                        onChange={(e) => handleCellChange(rowIndex, analysisType === 'split-plot' && includeBlocks ? 1 : 0, e.target.value)}
                        className="w-full px-2 py-1 bg-slate-700/50 text-gray-100 border border-slate-600 rounded focus:outline-none focus:ring-1 focus:ring-indigo-500"
                        placeholder={analysisType === 'split-plot' ? 'I1' : 'A1'}
                      />
                    </td>
                    <td className="px-2 py-1">
                      <input
                        type="text"
                        value={analysisType === 'split-plot' && includeBlocks ? row[2] : row[1]}
                        onChange={(e) => handleCellChange(rowIndex, analysisType === 'split-plot' && includeBlocks ? 2 : 1, e.target.value)}
                        className="w-full px-2 py-1 bg-slate-700/50 text-gray-100 border border-slate-600 rounded focus:outline-none focus:ring-1 focus:ring-indigo-500"
                        placeholder="e.g., S1"
                      />
                    </td>
                    <td className="px-2 py-1">
                      <input
                        type="number"
                        step="0.01"
                        value={analysisType === 'split-plot' && includeBlocks ? row[3] : row[2]}
                        onChange={(e) => handleCellChange(rowIndex, analysisType === 'split-plot' && includeBlocks ? 3 : 2, e.target.value)}
                        className="w-full px-2 py-1 bg-slate-700/50 text-gray-100 border border-slate-600 rounded focus:outline-none focus:ring-1 focus:ring-indigo-500"
                        placeholder="Value"
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-gray-400 text-xs mt-2">
            Enter data directly in the table. Rows will be added automatically as you type.
          </p>
        </div>

        {/* Analysis Options */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          {analysisType === 'mixed-anova' && (
            <div>
              <label className="flex items-center space-x-2 text-gray-200">
                <input
                  type="checkbox"
                  checked={includeInteractions}
                  onChange={(e) => setIncludeInteractions(e.target.checked)}
                  className="w-4 h-4"
                />
                <span>Include Interaction Effect</span>
              </label>
              <p className="text-gray-400 text-xs mt-1">
                Test for {factorNames[0]} × {factorNames[1]} interaction
              </p>
            </div>
          )}

          <div>
            <label className="block text-gray-200 font-medium mb-2">Significance Level (α)</label>
            <select
              value={alpha}
              onChange={(e) => setAlpha(parseFloat(e.target.value))}
              className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              <option value="0.001">0.001</option>
              <option value="0.01">0.01</option>
              <option value="0.05">0.05</option>
              <option value="0.10">0.10</option>
            </select>
          </div>
        </div>

        {/* Analyze Button */}
        <button
          onClick={runAnalysis}
          disabled={loading}
          className="w-full bg-gradient-to-r from-indigo-500 to-purple-500 text-white font-bold py-3 px-6 rounded-lg hover:from-indigo-600 hover:to-purple-600 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? 'Analyzing...' :
           analysisType === 'mixed-anova' ? 'Run Mixed Model ANOVA' :
           analysisType === 'split-plot' ? 'Run Split-Plot Analysis' :
           analysisType === 'nested' ? 'Run Nested Design Analysis' :
           analysisType === 'growth-curve' ? 'Run Growth Curve Analysis' :
           'Run Repeated Measures ANOVA'}
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-900/30 backdrop-blur-lg rounded-xl p-4 border border-red-700/50">
          <h4 className="text-red-200 font-semibold mb-2">Error</h4>
          <p className="text-red-100">{error}</p>
        </div>
      )}

      {/* Results Display */}
      {result && (
        <div className="space-y-6">
          {/* Model Summary */}
          {result.model_summary && (
            <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
              <h3 className="text-xl font-bold text-gray-100 mb-4">Model Summary</h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <div className="bg-gradient-to-r from-indigo-500/20 to-purple-500/20 rounded-lg p-4">
                  <p className="text-gray-400 text-sm">R²</p>
                  <p className="text-2xl font-bold text-indigo-400">{result.model_summary.r_squared}</p>
                </div>
                <div className="bg-gradient-to-r from-indigo-500/20 to-purple-500/20 rounded-lg p-4">
                  <p className="text-gray-400 text-sm">Adj. R²</p>
                  <p className="text-2xl font-bold text-indigo-400">{result.model_summary.adj_r_squared}</p>
                </div>
                <div className="bg-gradient-to-r from-indigo-500/20 to-purple-500/20 rounded-lg p-4">
                  <p className="text-gray-400 text-sm">F-statistic</p>
                  <p className="text-2xl font-bold text-indigo-400">{result.model_summary.f_statistic}</p>
                </div>
              </div>
            </div>
          )}

          {/* ICC Display - Phase 1 Enhancement */}
          {result.icc && <ICCDisplay iccData={result.icc} />}

          {/* Model Fit Metrics - Phase 1 Enhancement */}
          {result.model_fit && <ModelComparisonTable modelFit={result.model_fit} modelName={result.model_type || "Current Model"} />}

          {/* Variance Decomposition - Phase 1 Enhancement */}
          {result.variance_components && (
            <VarianceDecomposition
              varianceComponents={result.variance_components}
              iccData={result.icc}
            />
          )}

          {/* BLUPs Caterpillar Plot - Phase 3 Enhancement */}
          {result.blups && <BLUPsPlot blupsData={result.blups} />}

          {/* Random Effects Q-Q Plot - Phase 3 Enhancement */}
          {result.blups && <RandomEffectsQQPlot blupsData={result.blups} />}

          {/* Sphericity Test Results (Repeated Measures only) */}
          {analysisType === 'repeated-measures' && result.sphericity && (
            <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
              <h3 className="text-xl font-bold text-gray-100 mb-4">Sphericity Test (Mauchly's)</h3>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Test Statistics */}
                <div>
                  <div className="bg-slate-900/50 rounded-lg p-4 space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-300">W Statistic:</span>
                      <span className="text-gray-100 font-semibold">{result.sphericity.w_statistic?.toFixed(4)}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-300">p-value:</span>
                      <span className="text-gray-100 font-semibold">
                        {result.sphericity.p_value < 0.001 ? '<0.001' : result.sphericity.p_value?.toFixed(4)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center pt-2 border-t border-slate-700">
                      <span className="text-gray-300">Decision:</span>
                      <span className={`font-semibold ${result.sphericity.sphericity_assumed ? 'text-green-400' : 'text-yellow-400'}`}>
                        {result.sphericity.sphericity_assumed ? '✓ Sphericity Met' : '⚠ Sphericity Violated'}
                      </span>
                    </div>
                  </div>

                  <p className="text-gray-400 text-sm mt-3">
                    {result.sphericity.sphericity_assumed ? (
                      'The sphericity assumption is met (p ≥ 0.05). Use uncorrected F-tests.'
                    ) : (
                      'Sphericity assumption violated (p < 0.05). Use corrected F-tests (GG or HF).'
                    )}
                  </p>
                </div>

                {/* Correction Factors */}
                <div>
                  <h4 className="text-lg font-semibold text-gray-200 mb-3">Correction Factors (ε)</h4>
                  <div className="space-y-3">
                    <div className="bg-slate-900/50 rounded-lg p-3">
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-gray-300">Greenhouse-Geisser:</span>
                        <span className="text-indigo-400 font-semibold">{result.sphericity.epsilon_gg?.toFixed(4)}</span>
                      </div>
                      <p className="text-gray-400 text-xs">Conservative correction</p>
                    </div>
                    <div className="bg-slate-900/50 rounded-lg p-3">
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-gray-300">Huynh-Feldt:</span>
                        <span className="text-purple-400 font-semibold">{result.sphericity.epsilon_hf?.toFixed(4)}</span>
                      </div>
                      <p className="text-gray-400 text-xs">Less conservative correction</p>
                    </div>
                  </div>

                  <div className="mt-3 bg-indigo-900/20 rounded-lg p-3">
                    <p className="text-indigo-300 text-sm font-medium">
                      📋 {result.sphericity.recommendation}
                    </p>
                  </div>
                </div>
              </div>

              {/* Corrected Tests (if sphericity violated) */}
              {!result.sphericity.sphericity_assumed && result.corrected_tests && (
                <div className="mt-4 overflow-x-auto bg-slate-900/50 rounded-lg border border-slate-600">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="bg-slate-700/50">
                        <th className="px-4 py-3 text-left text-gray-200 font-medium">Correction</th>
                        <th className="px-4 py-3 text-right text-gray-200 font-medium">F</th>
                        <th className="px-4 py-3 text-right text-gray-200 font-medium">df (num)</th>
                        <th className="px-4 py-3 text-right text-gray-200 font-medium">df (denom)</th>
                        <th className="px-4 py-3 text-right text-gray-200 font-medium">p-value</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="border-t border-slate-700">
                        <td className="px-4 py-3 text-gray-300">Greenhouse-Geisser</td>
                        <td className="px-4 py-3 text-right text-gray-300">{result.corrected_tests.greenhouse_geisser.F?.toFixed(2)}</td>
                        <td className="px-4 py-3 text-right text-gray-300">{result.corrected_tests.greenhouse_geisser.df_numerator?.toFixed(2)}</td>
                        <td className="px-4 py-3 text-right text-gray-300">{result.corrected_tests.greenhouse_geisser.df_denominator?.toFixed(2)}</td>
                        <td className="px-4 py-3 text-right text-gray-300">
                          {result.corrected_tests.greenhouse_geisser.p_value < 0.001
                            ? '<0.001'
                            : result.corrected_tests.greenhouse_geisser.p_value?.toFixed(4)}
                        </td>
                      </tr>
                      <tr className="border-t border-slate-700">
                        <td className="px-4 py-3 text-gray-300">Huynh-Feldt</td>
                        <td className="px-4 py-3 text-right text-gray-300">{result.corrected_tests.huynh_feldt.F?.toFixed(2)}</td>
                        <td className="px-4 py-3 text-right text-gray-300">{result.corrected_tests.huynh_feldt.df_numerator?.toFixed(2)}</td>
                        <td className="px-4 py-3 text-right text-gray-300">{result.corrected_tests.huynh_feldt.df_denominator?.toFixed(2)}</td>
                        <td className="px-4 py-3 text-right text-gray-300">
                          {result.corrected_tests.huynh_feldt.p_value < 0.001
                            ? '<0.001'
                            : result.corrected_tests.huynh_feldt.p_value?.toFixed(4)}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}

          {/* ANOVA Table with EMS */}
          <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
            <h3 className="text-xl font-bold text-gray-100 mb-4">ANOVA Table</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-slate-700/50">
                    <th className="px-4 py-3 text-left text-gray-200 font-medium">Source</th>
                    <th className="px-4 py-3 text-right text-gray-200 font-medium">Sum of Squares</th>
                    <th className="px-4 py-3 text-right text-gray-200 font-medium">df</th>
                    <th className="px-4 py-3 text-right text-gray-200 font-medium">Mean Square</th>
                    <th className="px-4 py-3 text-left text-gray-200 font-medium">Expected Mean Squares</th>
                    <th className="px-4 py-3 text-right text-gray-200 font-medium">F</th>
                    <th className="px-4 py-3 text-right text-gray-200 font-medium">p-value</th>
                    <th className="px-4 py-3 text-left text-gray-200 font-medium">Error Term</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.anova_table).map(([source, data], idx) => {
                    const isSignificant = data.p_value_corrected
                      ? data.p_value_corrected < alpha
                      : data.p_value
                      ? data.p_value < alpha
                      : false

                    return (
                      <tr
                        key={idx}
                        className={`border-t border-slate-700 ${
                          isSignificant && source !== 'Residual'
                            ? 'bg-green-900/20'
                            : 'hover:bg-slate-700/30'
                        }`}
                      >
                        <td className="px-4 py-3 text-gray-200 font-medium">{source}</td>
                        <td className="px-4 py-3 text-right text-gray-300">{data.sum_sq}</td>
                        <td className="px-4 py-3 text-right text-gray-300">{data.df}</td>
                        <td className="px-4 py-3 text-right text-gray-300">{data.mean_sq}</td>
                        <td className="px-4 py-3 text-left text-gray-300 font-mono text-xs">
                          {data.ems || '-'}
                        </td>
                        <td className="px-4 py-3 text-right text-gray-300">
                          {data.F_corrected !== undefined ? data.F_corrected : data.F || '-'}
                        </td>
                        <td className="px-4 py-3 text-right text-gray-300">
                          {data.p_value_corrected !== undefined
                            ? data.p_value_corrected < 0.001
                              ? '<0.001'
                              : data.p_value_corrected.toFixed(4)
                            : data.p_value
                            ? data.p_value < 0.001
                              ? '<0.001'
                              : data.p_value.toFixed(4)
                            : '-'}
                          {isSignificant && source !== 'Residual' && ' *'}
                        </td>
                        <td className="px-4 py-3 text-left text-gray-400 text-xs">
                          {data.error_term || '-'}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            <p className="text-gray-400 text-xs mt-4">
              * Significant at α = {alpha}. Expected Mean Squares (EMS) show what each MS estimates.
            </p>
          </div>

          {/* Variance Components */}
          {result.variance_components && (
            <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
              <h3 className="text-xl font-bold text-gray-100 mb-4">Variance Components</h3>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Variance Components Table */}
                <div>
                  <h4 className="text-lg font-semibold text-gray-200 mb-3">Estimates</h4>
                  <div className="overflow-x-auto bg-slate-900/50 rounded-lg border border-slate-600">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="bg-slate-700/50">
                          <th className="px-4 py-2 text-left text-gray-200 font-medium">Component</th>
                          <th className="px-4 py-2 text-right text-gray-200 font-medium">Estimate</th>
                          <th className="px-4 py-2 text-right text-gray-200 font-medium">% of Total</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(result.variance_components)
                          .filter(([_, value]) => value !== null)
                          .map(([component, value], idx) => (
                            <tr key={idx} className="border-t border-slate-700">
                              <td className="px-4 py-2 text-gray-300 font-mono text-xs">{component}</td>
                              <td className="px-4 py-2 text-right text-gray-300">{value.toFixed(6)}</td>
                              <td className="px-4 py-2 text-right text-gray-300">
                                {result.variance_percentages[component]
                                  ? result.variance_percentages[component].toFixed(1) + '%'
                                  : '-'}
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="text-gray-400 text-xs mt-2">
                    Variance components partition total variability into sources
                  </p>
                </div>

                {/* Variance Percentage Visualization */}
                <div>
                  <h4 className="text-lg font-semibold text-gray-200 mb-3">Distribution</h4>
                  <div className="space-y-3">
                    {Object.entries(result.variance_percentages || {}).map(([component, percentage], idx) => {
                      const colors = [
                        'bg-indigo-500',
                        'bg-purple-500',
                        'bg-pink-500',
                        'bg-blue-500',
                        'bg-cyan-500'
                      ]
                      const color = colors[idx % colors.length]

                      return (
                        <div key={idx}>
                          <div className="flex justify-between text-sm text-gray-300 mb-1">
                            <span className="font-mono text-xs">{component}</span>
                            <span className="font-semibold">{percentage.toFixed(1)}%</span>
                          </div>
                          <div className="w-full bg-slate-700/50 rounded-full h-6 overflow-hidden">
                            <div
                              className={`${color} h-full flex items-center justify-center text-white text-xs font-bold transition-all duration-500`}
                              style={{ width: `${percentage}%` }}
                            >
                              {percentage > 10 && `${percentage.toFixed(1)}%`}
                            </div>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                  <p className="text-gray-400 text-xs mt-4">
                    Bar chart shows relative contribution of each variance component to total variability
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Interpretation */}
          {result.interpretation && result.interpretation.length > 0 && (
            <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
              <h3 className="text-xl font-bold text-gray-100 mb-4">Interpretation</h3>
              <div className="space-y-2">
                {result.interpretation.map((line, idx) => (
                  <p key={idx} className="text-gray-300">
                    {line}
                  </p>
                ))}
              </div>
            </div>
          )}

          {/* Visualizations */}
          {result.plot_data && (
            <>
              {/* Interaction Plot */}
              {result.plot_data.cell_means && result.plot_data.cell_means.length > 0 && (() => {
                // Transform cell_means data to match InteractionPlot expected format
                const factors = analysisType === 'mixed-anova'
                  ? factorNames
                  : [result.whole_plot_factor, result.subplot_factor]

                const interactionData = {}
                result.plot_data.cell_means.forEach(cell => {
                  const key = `${cell[factors[0]]}, ${cell[factors[1]]}`
                  interactionData[key] = cell.mean
                })

                return (
                  <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
                    <h3 className="text-xl font-bold text-gray-100 mb-4">Interaction Plot</h3>
                    <p className="text-gray-400 text-sm mb-4">
                      Shows cell means for all factor combinations. Non-parallel lines indicate interaction.
                    </p>
                    <InteractionPlot
                      data={interactionData}
                      factorAName={factors[0]}
                      factorBName={factors[1]}
                    />
                  </div>
                )
              })()}

              {/* Main Effects Plots */}
              {result.plot_data.marginal_means && (() => {
                // Transform marginal_means to match MainEffectsPlot expected format
                const transformedData = {}
                Object.entries(result.plot_data.marginal_means).forEach(([factor, data]) => {
                  transformedData[factor] = {
                    levels: data.map(d => d.level),
                    means: data.map(d => d.mean)
                  }
                })

                return (
                  <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
                    <h3 className="text-xl font-bold text-gray-100 mb-4">Main Effects Plots</h3>
                    <p className="text-gray-400 text-sm mb-4">
                      Shows marginal means for each factor level. Steeper slopes indicate stronger effects.
                    </p>
                    <MainEffectsPlot
                      data={transformedData}
                      responseName={responseName}
                    />
                  </div>
                )
              })()}

              {/* Residual Diagnostic Plots */}
              {result.plot_data.residuals && result.plot_data.fitted_values && (() => {
                // Calculate standardized residuals
                const residuals = result.plot_data.residuals.filter(r => r !== null)
                const mean = residuals.reduce((sum, r) => sum + r, 0) / residuals.length
                const variance = residuals.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / residuals.length
                const stdDev = Math.sqrt(variance)
                const standardizedResiduals = residuals.map(r => r / stdDev)

                return (
                  <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
                    <h3 className="text-xl font-bold text-gray-100 mb-4">Residual Diagnostic Plots</h3>
                    <p className="text-gray-400 text-sm mb-4">
                      Check model assumptions: normality (Q-Q plot) and homogeneity of variance (residuals vs fitted).
                    </p>
                    <ResidualPlots
                      residuals={residuals}
                      fittedValues={result.plot_data.fitted_values.filter(f => f !== null)}
                      standardizedResiduals={standardizedResiduals}
                    />
                  </div>
                )
              })()}

              {/* Box Plots by Factor */}
              {result.plot_data.box_plot_data && (() => {
                // Helper function to calculate box plot statistics
                const calculateBoxStats = (values) => {
                  const cleanValues = values.filter(v => v !== null).sort((a, b) => a - b)
                  if (cleanValues.length === 0) return null

                  const q1Index = Math.floor(cleanValues.length * 0.25)
                  const medianIndex = Math.floor(cleanValues.length * 0.5)
                  const q3Index = Math.floor(cleanValues.length * 0.75)

                  const q1 = cleanValues[q1Index]
                  const median = cleanValues[medianIndex]
                  const q3 = cleanValues[q3Index]
                  const iqr = q3 - q1

                  const lowerFence = q1 - 1.5 * iqr
                  const upperFence = q3 + 1.5 * iqr

                  const outliers = cleanValues.filter(v => v < lowerFence || v > upperFence)
                  const nonOutliers = cleanValues.filter(v => v >= lowerFence && v <= upperFence)

                  return {
                    min: nonOutliers.length > 0 ? Math.min(...nonOutliers) : cleanValues[0],
                    q1,
                    median,
                    q3,
                    max: nonOutliers.length > 0 ? Math.max(...nonOutliers) : cleanValues[cleanValues.length - 1],
                    outliers
                  }
                }

                return (
                  <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
                    <h3 className="text-xl font-bold text-gray-100 mb-4">Distribution by Factor</h3>
                    <p className="text-gray-400 text-sm mb-4">
                      Shows the distribution of response values for each level of each factor.
                    </p>
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                      {Object.entries(result.plot_data.box_plot_data).map(([factor, data]) => {
                        // Transform data to match BoxPlot expected format
                        const transformedData = data
                          .map(d => {
                            const stats = calculateBoxStats(d.values)
                            if (!stats) return null
                            return {
                              label: d.level,
                              ...stats
                            }
                          })
                          .filter(d => d !== null)

                        return (
                          <div key={factor}>
                            <h4 className="text-lg font-semibold text-gray-200 mb-3">{factor}</h4>
                            <BoxPlot
                              data={transformedData}
                              factorName={factor}
                              responseName={responseName}
                            />
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )
              })()}

              {/* Nested Design Specific Visualizations */}
              {analysisType === 'nested' && (
                <>
                  {/* Variance Components Pie Chart */}
                  {result.variance_percentages && (
                    <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
                      <VarianceComponentsChart
                        variancePercentages={result.variance_percentages}
                        title="Variance Components Distribution"
                      />
                    </div>
                  )}

                  {/* Hierarchical Means Plot */}
                  {result.plot_data.marginal_means_a && result.plot_data.nested_means && (
                    <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
                      <HierarchicalMeansPlot
                        marginalMeansA={result.plot_data.marginal_means_a}
                        nestedMeans={result.plot_data.nested_means}
                        factorA={result.factor_a}
                        factorB={result.factor_b_nested}
                      />
                    </div>
                  )}

                  {/* Nested Box Plots */}
                  {result.plot_data.box_plot_data_nested && (
                    <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
                      <NestedBoxPlots
                        boxPlotDataNested={result.plot_data.box_plot_data_nested}
                        factorA={result.factor_a}
                      />
                    </div>
                  )}

                  {/* ICC Interpretation Card */}
                  {result.icc && (
                    <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
                      <h3 className="text-xl font-bold text-gray-100 mb-4">Intraclass Correlation (ICC)</h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {Object.entries(result.icc).map(([key, value]) => (
                          <div key={key} className="bg-slate-700/50 rounded-lg p-4">
                            <div className="text-gray-400 text-sm mb-1">{key}</div>
                            <div className="text-3xl font-bold text-indigo-400 mb-2">
                              {(value * 100).toFixed(1)}%
                            </div>
                            <div className="w-full bg-slate-600 rounded-full h-2">
                              <div
                                className="bg-indigo-500 h-2 rounded-full transition-all"
                                style={{ width: `${value * 100}%` }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                      <div className="mt-4 bg-slate-700/30 rounded-lg p-4">
                        <p className="text-gray-300 text-sm">
                          <strong className="text-gray-100">Interpretation:</strong> ICC measures the proportion of total variance
                          attributable to grouping. Higher ICC values indicate stronger clustering effects.
                          {result.icc[`ICC(${result.factor_a})`] > 0.3 && (
                            <span className="text-indigo-300"> The high ICC({result.factor_a}) suggests substantial between-{result.factor_a} variability.</span>
                          )}
                        </p>
                      </div>
                    </div>
                  )}
                </>
              )}

              {/* Repeated Measures Specific Visualizations */}
              {analysisType === 'repeated-measures' && result.plot_data && (
                <>
                  {/* Profile Plot */}
                  {result.plot_data.profile_data && (
                    <ProfilePlot
                      profileData={result.plot_data.profile_data}
                      withinFactor={result.within_factor || withinFactorName}
                      responseName={responseName}
                    />
                  )}

                  {/* Individual Trajectories */}
                  {result.plot_data.trajectories && result.plot_data.profile_data && (
                    <WithinSubjectVariabilityPlot
                      trajectories={result.plot_data.trajectories}
                      profileData={result.plot_data.profile_data}
                      withinFactor={result.within_factor || withinFactorName}
                      responseName={responseName}
                    />
                  )}
                </>
              )}

              {/* Growth Curve Specific Visualizations */}
              {analysisType === 'growth-curve' && (
                <>
                  {/* Growth Curve Results Summary */}
                  <GrowthCurveResults result={result} />

                  {/* Growth Curve Spaghetti Plot */}
                  {result.individual_trajectories && result.population_curve && (
                    <GrowthCurvePlot
                      individualTrajectories={result.individual_trajectories}
                      populationCurve={result.population_curve}
                      timeVar={growthTimeVar}
                      responseVar={responseName}
                    />
                  )}
                </>
              )}
            </>
          )}

          {/* Export Button */}
          <div className="flex justify-end">
            <button
              onClick={() => {
                const exportData = {
                  model_type: result.model_type,
                  fixed_factors: result.fixed_factors,
                  random_factors: result.random_factors,
                  anova_table: result.anova_table,
                  variance_components: result.variance_components,
                  variance_percentages: result.variance_percentages,
                  model_summary: result.model_summary,
                  interpretation: result.interpretation
                }
                const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
                const url = URL.createObjectURL(blob)
                const a = document.createElement('a')
                a.href = url
                a.download = `mixed-model-anova-${new Date().toISOString().slice(0, 10)}.json`
                a.click()
                URL.revokeObjectURL(url)
              }}
              className="px-6 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg transition flex items-center space-x-2"
            >
              <Download className="w-4 h-4" />
              <span>Export Results (JSON)</span>
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export default MixedModels
