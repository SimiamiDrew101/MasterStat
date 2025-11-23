import { useState, useEffect } from 'react'
import axios from 'axios'
import { Mountain, Plus, Trash2, Target, TrendingUp, Sparkles } from 'lucide-react'
import ResultCard from '../components/ResultCard'
import ResponseSurface3D from '../components/ResponseSurface3D'
import ContourPlot from '../components/ContourPlot'
import SlicedVisualization from '../components/SlicedVisualization'
import ResidualAnalysis from '../components/ResidualAnalysis'
import EnhancedANOVA from '../components/EnhancedANOVA'
import PredictionProfiler from '../components/PredictionProfiler'
import AdvancedDiagnostics from '../components/AdvancedDiagnostics'
import ExperimentWizard from '../components/ExperimentWizard'
import CrossValidationResults from '../components/CrossValidationResults'
import MultiResponseManager from '../components/MultiResponseManager'
import MultiResponseContourOverlay from '../components/MultiResponseContourOverlay'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const RSM = () => {
  // Design configuration
  const [designType, setDesignType] = useState('ccd')  // 'ccd' or 'box-behnken'
  const [ccdType, setCCDType] = useState('face-centered')  // 'face-centered', 'rotatable', 'inscribed'
  const [numFactors, setNumFactors] = useState(2)
  const [numCenterPoints, setNumCenterPoints] = useState(4)
  const [factorNames, setFactorNames] = useState(['X1', 'X2'])
  const [responseName, setResponseName] = useState('Y')
  const [alpha, setAlpha] = useState(0.05)

  // Data and results
  const [designData, setDesignData] = useState(null)
  const [tableData, setTableData] = useState([])
  const [modelResult, setModelResult] = useState(null)
  const [canonicalResult, setCanonicalResult] = useState(null)
  const [optimizationResult, setOptimizationResult] = useState(null)
  const [steepestAscentResult, setSteepestAscentResult] = useState(null)
  const [confirmationRunsResult, setConfirmationRunsResult] = useState(null)
  const [ridgeAnalysisResult, setRidgeAnalysisResult] = useState(null)
  const [targetResponse, setTargetResponse] = useState('')
  const [nConfirmationRuns, setNConfirmationRuns] = useState(3)
  const [surfaceData, setSurfaceData] = useState(null)
  const [advancedDiagnostics, setAdvancedDiagnostics] = useState(null)
  const [cvResults, setCvResults] = useState(null)
  const [cvLoading, setCvLoading] = useState(false)
  const [cvError, setCvError] = useState(null)
  const [kFolds, setKFolds] = useState(5)

  // Multi-response state (Phase 2 Feature 3)
  const [responseNames, setResponseNames] = useState(['Y'])  // Array of response names
  const [multiResponseMode, setMultiResponseMode] = useState(false)  // Toggle for multi-response
  const [multiModelResults, setMultiModelResults] = useState(null)  // Results from /fit-multi-model
  const [multiSurfaceData, setMultiSurfaceData] = useState(null)  // Surface data for multiple responses
  const [responseConfigs, setResponseConfigs] = useState({})  // Visualization config per response
  const [normalizationMethod, setNormalizationMethod] = useState('zscore')  // 'none', 'zscore', 'minmax'

  // Advanced features state (Features 4-6)
  const [augmentationStrategy, setAugmentationStrategy] = useState('space-filling')
  const [nAugmentRuns, setNAugmentRuns] = useState(5)
  const [augmentationResult, setAugmentationResult] = useState(null)
  const [constrainedOptResult, setConstrainedOptResult] = useState(null)
  const [constraints, setConstraints] = useState({ box: {}, linear: [] })
  const [multiResponseModels, setMultiResponseModels] = useState([]) // Array of {name, coefficients}
  const [desirabilitySpecs, setDesirabilitySpecs] = useState([])
  const [desirabilityResult, setDesirabilityResult] = useState(null)

  // UI state
  const [loading, setLoading] = useState(false)
  const [wizardOpen, setWizardOpen] = useState(false)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('design')  // 'design', 'model', 'optimize', 'visualize'

  // Update factor names when number changes
  useEffect(() => {
    const newFactorNames = Array.from({ length: numFactors }, (_, i) => `X${i + 1}`)
    console.log('Setting factor names:', newFactorNames)
    setFactorNames(newFactorNames)
  }, [numFactors])

  // Debug table data
  useEffect(() => {
    console.log('TableData updated:', tableData.length, 'rows')
    console.log('Factor names:', factorNames)
    console.log('Response name:', responseName)
  }, [tableData, factorNames, responseName])

  // Debug surface data
  useEffect(() => {
    console.log('surfaceData state changed:', surfaceData ? `${surfaceData.length} points` : 'null')
  }, [surfaceData])

  // Handle wizard completion (Phase 2 Feature)
  const handleWizardComplete = async (wizardData) => {
    console.log('Wizard completed with data:', wizardData)

    // Apply wizard recommendations to the design settings
    if (wizardData.selectedDesign) {
      setNumFactors(wizardData.nFactors)

      const designCode = wizardData.selectedDesign.design_code

      // Check if this is a sequential approach (not directly generatable)
      const isSequentialApproach = ['screening-first', 'dsd'].includes(designCode)

      // Map design code to our design type
      if (designCode === 'box-behnken') {
        setDesignType('box-behnken')
      } else if (designCode === 'face-centered') {
        setDesignType('ccd')
        setCCDType('face-centered')
      } else if (designCode === 'rotatable') {
        setDesignType('ccd')
        setCCDType('rotatable')
      } else if (designCode === 'inscribed') {
        setDesignType('ccd')
        setCCDType('inscribed')
      } else if (isSequentialApproach) {
        // For sequential approaches, default to face-centered CCD
        setDesignType('ccd')
        setCCDType('face-centered')
      }

      // Set factor names if provided
      if (wizardData.factorNames && wizardData.factorNames.length > 0) {
        const names = wizardData.factorNames.filter(n => n && n.trim())
        if (names.length > 0) {
          setFactorNames(names)
        }
      }

      // Handle sequential approaches differently
      if (isSequentialApproach) {
        // Show informational message instead of auto-generating
        setError(null)
        setActiveTab('design')

        // Create a helpful info message
        setTimeout(() => {
          alert(
            `📋 Sequential Approach Selected: ${wizardData.selectedDesign.type}\n\n` +
            `This is a two-stage approach:\n` +
            `1. First, run a screening experiment (Factorial Designs page)\n` +
            `2. Then, use RSM on the 2-3 most important factors\n\n` +
            `For now, the wizard has set up a Face-Centered CCD design.\n` +
            `You can adjust the settings and generate when ready.`
          )
        }, 500)
        return
      }

      // Auto-generate the design after wizard completes (for supported designs)
      setTimeout(async () => {
        try {
          setLoading(true)
          let response
          if (designCode === 'box-behnken') {
            response = await axios.post(`${API_URL}/api/rsm/box-behnken/generate`, {
              n_factors: wizardData.nFactors
            })
          } else {
            // CCD variants (face-centered, rotatable, inscribed)
            response = await axios.post(`${API_URL}/api/rsm/ccd/generate`, {
              n_factors: wizardData.nFactors,
              design_type: designCode,
              n_center: numCenterPoints
            })
          }

          setDesignData(response.data)
          setTableData(initializeTableData(response.data.design))
          setActiveTab('design') // Switch to design tab to see results
        } catch (err) {
          setError(err.response?.data?.detail || err.message || 'Design generation failed')
        } finally {
          setLoading(false)
        }
      }, 300) // Small delay for UX smoothness
    }
  }

  // Generate design
  const handleGenerateDesign = async () => {
    setLoading(true)
    setError(null)

    try {
      let response
      if (designType === 'ccd') {
        response = await axios.post(`${API_URL}/api/rsm/ccd/generate`, {
          n_factors: numFactors,
          design_type: ccdType,
          n_center: numCenterPoints
        })
      } else {
        response = await axios.post(`${API_URL}/api/rsm/box-behnken/generate`, {
          n_factors: numFactors,
          n_center: numCenterPoints
        })
      }

      setDesignData(response.data)

      // Convert to table format with empty response column(s)
      const table = response.data.design_matrix.map(row => {
        const tableRow = factorNames.map(factor => row[factor] || 0)
        // Add empty columns for each response
        responseNames.forEach(() => tableRow.push(''))
        return tableRow
      })

      console.log('Generated table data:', table)
      console.log('Factor names:', factorNames)
      console.log('Table row example:', table[0])
      setTableData(table)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to generate design')
    } finally {
      setLoading(false)
    }
  }

  // Fit RSM model
  const handleFitModel = async () => {
    setLoading(true)
    setError(null)

    try {
      // Validate data - check if response column (last column) is filled
      console.log('Table data:', tableData)
      const validRows = tableData.filter(row => {
        const responseValue = row[row.length - 1]
        const isValid = responseValue !== '' && responseValue !== null && responseValue !== undefined
        console.log('Row:', row, 'Response:', responseValue, 'Valid:', isValid)
        return isValid
      })

      console.log('Valid rows:', validRows.length, 'Total rows:', tableData.length)

      if (validRows.length < 5) {
        throw new Error(`Need at least 5 complete data points for RSM (found ${validRows.length} rows with response values)`)
      }

      // Convert to API format
      const data = validRows.map(row => {
        const point = {}
        factorNames.forEach((factor, i) => {
          point[factor] = parseFloat(row[i])
        })
        point[responseName] = parseFloat(row[row.length - 1])
        return point
      })

      const response = await axios.post(`${API_URL}/api/rsm/fit-model`, {
        data: data,
        factors: factorNames,
        response: responseName,
        alpha: alpha
      })

      setModelResult(response.data)

      // Automatically perform canonical analysis
      if (response.data.coefficients) {
        const canonicalResponse = await axios.post(`${API_URL}/api/rsm/canonical-analysis`, {
          coefficients: Object.fromEntries(
            Object.entries(response.data.coefficients).map(([k, v]) => [k, v.estimate])
          ),
          factors: factorNames
        })
        setCanonicalResult(canonicalResponse.data)

        // Fetch advanced diagnostics
        try {
          const diagnosticsResponse = await axios.post(`${API_URL}/api/rsm/advanced-diagnostics`, {
            data: data,
            factors: factorNames,
            response: responseName,
            coefficients: response.data.coefficients
          })
          setAdvancedDiagnostics(diagnosticsResponse.data)
        } catch (diagError) {
          console.error('Error fetching diagnostics:', diagError)
          // Don't fail the entire model fitting if diagnostics fail
        }
      }

      // Generate surface data for visualization (2 factors for simple viz)
      if (numFactors === 2) {
        console.log('Generating surface data for 2-factor visualization...')
        console.log('Coefficients:', response.data.coefficients)
        try {
          generateSurfaceData(response.data.coefficients)
          console.log('Surface data generated successfully')
        } catch (surfaceError) {
          console.error('Error generating surface data:', surfaceError)
          setError('Visualization data generation failed: ' + surfaceError.message)
        }
      } else {
        console.log('Multi-factor design - will use sliced visualization')
      }

      setActiveTab('model')
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to fit model')
    } finally {
      setLoading(false)
    }
  }

  // Run K-fold Cross-Validation (Phase 2 Feature)
  const handleCrossValidation = async () => {
    setCvLoading(true)
    setCvError(null)
    setCvResults(null)

    try {
      // Validate data
      const validRows = tableData.filter(row => {
        const responseValue = row[row.length - 1]
        return responseValue !== '' && responseValue !== null && responseValue !== undefined
      })

      if (validRows.length < kFolds * 2) {
        throw new Error(`Need at least ${kFolds * 2} complete data points for ${kFolds}-fold cross-validation (found ${validRows.length})`)
      }

      // Convert to API format
      const data = validRows.map(row => {
        const point = {}
        factorNames.forEach((factor, i) => {
          point[factor] = parseFloat(row[i])
        })
        point[responseName] = parseFloat(row[row.length - 1])
        return point
      })

      const response = await axios.post(`${API_URL}/api/rsm/cross-validate`, {
        data: data,
        factors: factorNames,
        response: responseName,
        k_folds: kFolds
      })

      setCvResults(response.data)
    } catch (err) {
      setCvError(err.response?.data?.detail || err.message || 'Failed to run cross-validation')
    } finally {
      setCvLoading(false)
    }
  }

  // Multi-response model fitting (Phase 2 Feature 3)
  const handleFitMultiModel = async () => {
    setLoading(true)
    setError(null)

    try {
      // Validate that we have data for all responses
      const validRows = tableData.filter(row => {
        // Check all response columns (last N columns where N = responseNames.length)
        const numResponses = responseNames.length
        const numFactors = factorNames.length
        return responseNames.every((_, idx) => {
          const responseValue = row[numFactors + idx]
          return responseValue !== '' && responseValue !== null && responseValue !== undefined
        })
      })

      if (validRows.length < 5) {
        throw new Error(`Need at least 5 complete data points for multi-response RSM (found ${validRows.length})`)
      }

      // Convert to API format
      const data = validRows.map(row => {
        const point = {}
        factorNames.forEach((factor, i) => {
          point[factor] = parseFloat(row[i])
        })
        responseNames.forEach((respName, idx) => {
          point[respName] = parseFloat(row[factorNames.length + idx])
        })
        return point
      })

      const response = await axios.post(`${API_URL}/api/rsm/fit-multi-model`, {
        data: data,
        factors: factorNames,
        responses: responseNames,
        alpha: alpha
      })

      setMultiModelResults(response.data)

      // Initialize response configs for visualization
      const configs = {}
      const colorSchemes = ['Viridis', 'Plasma', 'Inferno', 'Cividis', 'Turbo']
      responseNames.forEach((name, idx) => {
        configs[name] = {
          visible: true,
          colorScale: colorSchemes[idx % colorSchemes.length],
          opacity: 0.7
        }
      })
      setResponseConfigs(configs)

      setActiveTab('model')
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to fit multi-response models')
    } finally {
      setLoading(false)
    }
  }

  // Generate multi-surface data (Phase 2 Feature 3)
  const handleGenerateMultiSurface = async () => {
    if (!multiModelResults || !multiModelResults.models) {
      setError('Fit multi-response models first')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const response = await axios.post(`${API_URL}/api/rsm/generate-multi-surface`, {
        models: multiModelResults.models,
        factors: factorNames,
        grid_resolution: 20,
        normalize: normalizationMethod
      })

      setMultiSurfaceData(response.data)
      setActiveTab('visualize')
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to generate multi-surface data')
    } finally {
      setLoading(false)
    }
  }

  // Handle response names update from MultiResponseManager
  const handleUpdateResponseNames = (newResponseNames) => {
    setResponseNames(newResponseNames)

    // Update table data to match new response count
    if (tableData.length > 0) {
      const numFactors = factorNames.length
      const updatedTable = tableData.map(row => {
        // Keep factor columns
        const factorCols = row.slice(0, numFactors)

        // Adjust response columns
        const currentResponses = row.slice(numFactors)
        const newResponses = newResponseNames.map((_, idx) => {
          // Keep existing data if available, otherwise empty string
          return currentResponses[idx] !== undefined ? currentResponses[idx] : ''
        })

        return [...factorCols, ...newResponses]
      })

      setTableData(updatedTable)
    }

    // Update multi-response mode
    setMultiResponseMode(newResponseNames.length > 1)
  }

  // Generate surface data for visualization
  const generateSurfaceData = (coefficients) => {
    console.log('generateSurfaceData called with coefficients:', coefficients)
    const points = []
    const steps = 20
    const coefObj = Object.fromEntries(
      Object.entries(coefficients).map(([k, v]) => [k, v.estimate || v])
    )
    console.log('Coefficient object:', coefObj)
    console.log('Factor names:', factorNames)

    for (let i = 0; i <= steps; i++) {
      for (let j = 0; j <= steps; j++) {
        const x = -2 + (4 * i) / steps
        const y = -2 + (4 * j) / steps

        // Calculate z using second-order model
        let z = coefObj['Intercept'] || 0
        z += (coefObj[factorNames[0]] || 0) * x
        z += (coefObj[factorNames[1]] || 0) * y
        z += (coefObj[`I(${factorNames[0]}**2)`] || 0) * x * x
        z += (coefObj[`I(${factorNames[1]}**2)`] || 0) * y * y
        z += (coefObj[`${factorNames[0]}:${factorNames[1]}`] || 0) * x * y

        points.push({ x, y, z })
      }
    }

    console.log('Generated surface points:', points.length, 'points')
    console.log('First few points:', points.slice(0, 3))
    setSurfaceData(points)
    console.log('surfaceData state updated')
  }

  // Optimize response
  const handleOptimize = async (target) => {
    setLoading(true)
    setError(null)

    try {
      if (!modelResult || !modelResult.coefficients) {
        throw new Error('Fit a model first')
      }

      const response = await axios.post(`${API_URL}/api/rsm/optimize`, {
        coefficients: Object.fromEntries(
          Object.entries(modelResult.coefficients).map(([k, v]) => [k, v.estimate])
        ),
        factors: factorNames,
        target: target,
        variance_estimate: modelResult.anova?.Residual?.mean_sq || null
      })

      setOptimizationResult(response.data)
      setActiveTab('optimize')
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Optimization failed')
    } finally {
      setLoading(false)
    }
  }

  // Export model to industry standards
  const handleExport = async (format) => {
    try {
      if (!modelResult || !modelResult.coefficients) {
        throw new Error('Fit a model first before exporting')
      }

      // Prepare experimental data
      const validRows = tableData.filter(row => {
        const responseValue = row[row.length - 1]
        return responseValue !== '' && responseValue !== null && responseValue !== undefined
      })

      const data = validRows.map(row => {
        const point = {}
        factorNames.forEach((factor, i) => {
          point[factor] = parseFloat(row[i])
        })
        point[responseName] = parseFloat(row[row.length - 1])
        return point
      })

      const response = await axios.post(`${API_URL}/api/rsm/export`, {
        format: format,
        model_data: modelResult,
        factors: factorNames,
        response: responseName,
        data: data
      })

      // Download the file
      let blob
      if (format === 'pdf') {
        // PDF comes as base64-encoded, need to decode it
        const binaryString = atob(response.data.content)
        const bytes = new Uint8Array(binaryString.length)
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i)
        }
        blob = new Blob([bytes], { type: 'application/pdf' })
      } else {
        // Text files (JMP, R, Python) as-is
        blob = new Blob([response.data.content], { type: response.data.mime_type })
      }

      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = response.data.filename
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(url)

    } catch (err) {
      setError(err.response?.data?.detail || err.message || `Export to ${format.toUpperCase()} failed`)
    }
  }

  // Calculate steepest ascent
  const handleSteepestAscent = async () => {
    setLoading(true)
    setError(null)

    try {
      if (!modelResult || !modelResult.coefficients) {
        throw new Error('Fit a model first')
      }

      // Use linear coefficients only for steepest ascent
      const linearCoefs = {}
      factorNames.forEach(factor => {
        if (modelResult.coefficients[factor]) {
          linearCoefs[factor] = modelResult.coefficients[factor].estimate
        }
      })

      // Use origin as current point
      const currentPoint = {}
      factorNames.forEach(factor => {
        currentPoint[factor] = 0
      })

      const response = await axios.post(`${API_URL}/api/rsm/steepest-ascent`, {
        current_point: currentPoint,
        coefficients: linearCoefs,
        step_size: 0.5,
        n_steps: 10
      })

      setSteepestAscentResult(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Steepest ascent failed')
    } finally {
      setLoading(false)
    }
  }

  // Calculate confirmation runs
  const handleConfirmationRuns = async () => {
    setLoading(true)
    setError(null)

    try {
      if (!optimizationResult || !optimizationResult.optimal_point) {
        throw new Error('Run optimization first to get optimal point')
      }

      if (!modelResult || !modelResult.coefficients) {
        throw new Error('Model not fitted')
      }

      // Get residual mean square from model (if available)
      const varianceEstimate = modelResult.anova?.Residual?.mean_sq || null

      const response = await axios.post(`${API_URL}/api/rsm/confirmation-runs`, {
        optimal_point: optimizationResult.optimal_point,
        coefficients: Object.fromEntries(
          Object.entries(modelResult.coefficients).map(([k, v]) => [k, v.estimate])
        ),
        factors: factorNames,
        n_runs: nConfirmationRuns,
        variance_estimate: varianceEstimate
      })

      setConfirmationRunsResult(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Confirmation runs calculation failed')
    } finally {
      setLoading(false)
    }
  }

  // Calculate ridge analysis
  const handleRidgeAnalysis = async () => {
    setLoading(true)
    setError(null)

    try {
      if (!modelResult || !modelResult.coefficients) {
        throw new Error('Fit a model first')
      }

      if (!targetResponse || targetResponse === '') {
        throw new Error('Enter target response value')
      }

      const response = await axios.post(`${API_URL}/api/rsm/ridge-analysis`, {
        coefficients: Object.fromEntries(
          Object.entries(modelResult.coefficients).map(([k, v]) => [k, v.estimate])
        ),
        factors: factorNames,
        target_response: parseFloat(targetResponse),
        n_points: 50
      })

      setRidgeAnalysisResult(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Ridge analysis failed')
    } finally {
      setLoading(false)
    }
  }

  // Feature 5: Design Augmentation
  const handleDesignAugmentation = async () => {
    setLoading(true)
    setError(null)

    try {
      if (tableData.length === 0) {
        throw new Error('Generate a design first')
      }

      // Convert current design to API format
      const currentDesign = tableData.map(row => {
        const point = {}
        factorNames.forEach((factor, i) => {
          point[factor] = parseFloat(row[i]) || 0
        })
        return point
      })

      const response = await axios.post(`${API_URL}/api/rsm/design-augmentation`, {
        current_design: currentDesign,
        factors: factorNames,
        n_new_runs: nAugmentRuns,
        strategy: augmentationStrategy,
        coefficients: modelResult ? Object.fromEntries(
          Object.entries(modelResult.coefficients).map(([k, v]) => [k, v.estimate])
        ) : null
      })

      setAugmentationResult(response.data)

      // Add new runs to table
      const newRuns = response.data.augmented_design.slice(-nAugmentRuns)
      const newTableRows = newRuns.map(run => {
        const row = factorNames.map(factor => run[factor].toFixed(4))
        row.push('') // Empty response column
        return row
      })
      setTableData([...tableData, ...newTableRows])
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Design augmentation failed')
    } finally {
      setLoading(false)
    }
  }

  // Feature 6: Constrained Optimization
  const handleConstrainedOptimization = async (target) => {
    setLoading(true)
    setError(null)

    try {
      if (!modelResult || !modelResult.coefficients) {
        throw new Error('Fit a model first')
      }

      // Build constraints object
      const constraintsPayload = {
        box: {},
        linear: []
      }

      // Add box constraints if specified
      factorNames.forEach(factor => {
        if (constraints.box[factor]) {
          constraintsPayload.box[factor] = constraints.box[factor]
        }
      })

      // Add linear constraints if specified
      if (constraints.linear && constraints.linear.length > 0) {
        constraintsPayload.linear = constraints.linear
      }

      const response = await axios.post(`${API_URL}/api/rsm/constrained-optimization`, {
        coefficients: Object.fromEntries(
          Object.entries(modelResult.coefficients).map(([k, v]) => [k, v.estimate])
        ),
        factors: factorNames,
        target: target,
        constraints: constraintsPayload,
        variance_estimate: modelResult.anova?.Residual?.mean_sq || null
      })

      setConstrainedOptResult(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Constrained optimization failed')
    } finally {
      setLoading(false)
    }
  }

  // Feature 4: Desirability Functions
  const handleDesirabilityOptimization = async () => {
    setLoading(true)
    setError(null)

    try {
      if (desirabilitySpecs.length === 0) {
        throw new Error('Add at least one response with desirability specification')
      }

      // Validate all specs have required fields
      for (const spec of desirabilitySpecs) {
        if (!spec.response_name || !spec.coefficients || !spec.goal) {
          throw new Error(`Incomplete specification for response: ${spec.response_name || 'unknown'}`)
        }
      }

      const response = await axios.post(`${API_URL}/api/rsm/desirability-optimization`, {
        responses: desirabilitySpecs,
        factors: factorNames
      })

      setDesirabilityResult(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Desirability optimization failed')
    } finally {
      setLoading(false)
    }
  }

  // Helper: Save current model to multi-response models
  const saveCurrentModelToMultiResponse = () => {
    if (!modelResult || !modelResult.coefficients) {
      setError('Fit a model first before saving')
      return
    }

    const newModel = {
      name: responseName,
      coefficients: Object.fromEntries(
        Object.entries(modelResult.coefficients).map(([k, v]) => [k, v.estimate])
      )
    }

    setMultiResponseModels([...multiResponseModels, newModel])
    setError(null)
  }

  // Table editing
  const handleCellChange = (rowIndex, colIndex, value) => {
    const newData = [...tableData]
    newData[rowIndex][colIndex] = value
    setTableData(newData)
  }

  const addRow = () => {
    const newRow = Array(factorNames.length + 1).fill('')
    setTableData([...tableData, newRow])
  }

  const removeRow = (rowIndex) => {
    if (tableData.length > 1) {
      setTableData(tableData.filter((_, idx) => idx !== rowIndex))
    }
  }

  // Fill test data for demonstration
  const fillTestData = () => {
    const newData = tableData.map(row => {
      // Keep factor values, generate random response
      const newRow = [...row]
      const factors = row.slice(0, -1) // All except last column
      // Generate response based on a simple quadratic model for testing
      let response = 10
      factors.forEach((factor, i) => {
        response += 5 * factor + 2 * factor * factor
      })
      // Add some noise
      response += (Math.random() - 0.5) * 5
      newRow[newRow.length - 1] = response.toFixed(2)
      return newRow
    })
    setTableData(newData)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-orange-900/30 to-red-900/30 backdrop-blur-lg rounded-2xl p-8 border border-orange-700/50">
        <div className="flex items-center space-x-3 mb-4">
          <Mountain className="w-10 h-10 text-orange-400" />
          <h2 className="text-4xl font-bold text-gray-100">Response Surface Methodology</h2>
        </div>
        <p className="text-gray-300 text-lg">
          Design experiments, fit second-order models, and optimize responses using Central Composite and Box-Behnken designs.
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="flex space-x-2 bg-slate-800/50 p-2 rounded-lg">
        <button
          onClick={() => setActiveTab('design')}
          className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
            activeTab === 'design'
              ? 'bg-orange-600 text-white'
              : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
          }`}
        >
          1. Design
        </button>
        <button
          onClick={() => setActiveTab('model')}
          className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
            activeTab === 'model'
              ? 'bg-orange-600 text-white'
              : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
          }`}
          disabled={!modelResult}
        >
          2. Model & Analysis
        </button>
        <button
          onClick={() => setActiveTab('optimize')}
          className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
            activeTab === 'optimize'
              ? 'bg-orange-600 text-white'
              : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
          }`}
          disabled={!modelResult}
        >
          3. Optimization
        </button>
        <button
          onClick={() => setActiveTab('visualize')}
          className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
            activeTab === 'visualize'
              ? 'bg-orange-600 text-white'
              : !modelResult
              ? 'bg-slate-700/50 text-gray-500 cursor-not-allowed'
              : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
          }`}
          disabled={!modelResult}
          title={!modelResult ? 'Fit a model first' : 'View response surface visualization'}
        >
          4. Visualize
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-900/30 backdrop-blur-lg rounded-xl p-4 border border-red-700/50">
          <p className="text-red-200 font-medium">Error: {error}</p>
        </div>
      )}

      {/* Tab Content */}
      {activeTab === 'design' && (
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
          <h3 className="text-2xl font-bold text-gray-100 mb-6">Experimental Design</h3>

          <div className="space-y-6">
            {/* Design Type Selection */}
            <div>
              <label className="block text-gray-100 font-medium mb-2">Design Type</label>
              <select
                value={designType}
                onChange={(e) => setDesignType(e.target.value)}
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-orange-500"
              >
                <option value="ccd">Central Composite Design (CCD)</option>
                <option value="box-behnken">Box-Behnken Design</option>
              </select>
            </div>

            {/* CCD Type Selection */}
            {designType === 'ccd' && (
              <div>
                <label className="block text-gray-100 font-medium mb-2">CCD Type</label>
                <select
                  value={ccdType}
                  onChange={(e) => setCCDType(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-orange-500"
                >
                  <option value="face-centered">Face-Centered (α = 1)</option>
                  <option value="rotatable">Rotatable (α = 2^(k/4))</option>
                  <option value="inscribed">Inscribed (α = 1)</option>
                </select>
                <p className="text-gray-400 text-xs mt-1">
                  {ccdType === 'face-centered' && 'Axial points on the faces of the factorial cube'}
                  {ccdType === 'rotatable' && 'Equal prediction variance at all points equidistant from center'}
                  {ccdType === 'inscribed' && 'All points within the original cube'}
                </p>
              </div>
            )}

            {/* Number of Factors */}
            <div>
              <label className="block text-gray-100 font-medium mb-2">Number of Factors</label>
              <input
                type="number"
                min={designType === 'box-behnken' ? 3 : 2}
                max={designType === 'box-behnken' ? 7 : 6}
                value={numFactors}
                onChange={(e) => setNumFactors(parseInt(e.target.value) || 2)}
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-orange-500"
              />
              <p className="text-gray-400 text-xs mt-1">
                {designType === 'box-behnken'
                  ? 'Box-Behnken designs require 3-7 factors'
                  : 'CCD supports 2-6 factors'}
              </p>
            </div>

            {/* Center Points */}
            <div>
              <label className="block text-gray-100 font-medium mb-2">Number of Center Points</label>
              <input
                type="number"
                min={1}
                max={10}
                value={numCenterPoints}
                onChange={(e) => setNumCenterPoints(parseInt(e.target.value) || 4)}
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-orange-500"
              />
              <p className="text-gray-400 text-xs mt-1">
                Center points enable curvature detection and pure error estimation
              </p>
            </div>

            {/* Response Name */}
            <div>
              <label className="block text-gray-100 font-medium mb-2">Response Variable Name</label>
              <input
                type="text"
                value={responseName}
                onChange={(e) => setResponseName(e.target.value)}
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-orange-500"
              />
            </div>

            {/* Experiment Wizard Button (Phase 2 Feature) */}
            <div className="bg-gradient-to-r from-purple-900/30 to-blue-900/30 rounded-lg p-4 border border-purple-700/50">
              <div className="flex items-center gap-2 mb-2">
                <Sparkles className="w-5 h-5 text-purple-400" />
                <span className="text-purple-300 font-semibold text-sm">NEW FEATURE</span>
              </div>
              <button
                onClick={() => setWizardOpen(true)}
                className="w-full bg-gradient-to-r from-purple-600 to-blue-600 text-white font-bold py-3 px-6 rounded-lg hover:from-purple-700 hover:to-blue-700 transition-all shadow-lg hover:shadow-purple-500/50"
              >
                <div className="flex items-center justify-center gap-2">
                  <Sparkles className="w-5 h-5" />
                  <span>Launch Experiment Wizard</span>
                </div>
              </button>
              <p className="text-gray-300 text-xs mt-2 text-center">
                Step-by-step guidance for choosing the perfect design
              </p>
            </div>

            {/* OR Divider */}
            <div className="flex items-center gap-3">
              <div className="flex-1 h-px bg-slate-600"></div>
              <span className="text-gray-400 text-sm">OR</span>
              <div className="flex-1 h-px bg-slate-600"></div>
            </div>

            {/* Generate Design Button */}
            <button
              onClick={handleGenerateDesign}
              disabled={loading}
              className="w-full bg-orange-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-orange-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Generating...' : 'Generate Design Manually'}
            </button>

            {/* Design Information */}
            {designData && (
              <div className="bg-gradient-to-r from-orange-900/20 to-red-900/20 rounded-lg p-5 border border-orange-700/30">
                <h4 className="text-orange-200 font-semibold mb-3">Design Properties</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-400">Design Type:</span>
                    <span className="text-gray-100 ml-2 font-medium">{designData.design_type}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Total Runs:</span>
                    <span className="text-gray-100 ml-2 font-medium">{designData.n_runs.total}</span>
                  </div>
                  {designType === 'ccd' && (
                    <>
                      <div>
                        <span className="text-gray-400">Factorial Points:</span>
                        <span className="text-gray-100 ml-2">{designData.n_runs.factorial}</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Axial Points:</span>
                        <span className="text-gray-100 ml-2">{designData.n_runs.axial}</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Alpha (α):</span>
                        <span className="text-gray-100 ml-2">{designData.alpha}</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Rotatable:</span>
                        <span className="text-gray-100 ml-2">{designData.properties.rotatable ? 'Yes' : 'No'}</span>
                      </div>
                    </>
                  )}
                  {designType === 'box-behnken' && (
                    <div>
                      <span className="text-gray-400">Edge Points:</span>
                      <span className="text-gray-100 ml-2">{designData.n_runs.edge_points}</span>
                    </div>
                  )}
                  <div className="col-span-2">
                    <span className="text-gray-400">Description:</span>
                    <span className="text-gray-100 ml-2">{designData.properties.description || designData.properties.alpha_description}</span>
                  </div>
                </div>
              </div>
            )}

            {/* Data Entry Table */}
            {tableData.length > 0 && (
              <div>
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-gray-100 font-semibold text-lg">Experimental Data</h4>
                  <div className="flex items-center space-x-2">
                    <MultiResponseManager
                      responseNames={responseNames}
                      onUpdate={handleUpdateResponseNames}
                      disabled={!!modelResult || !!multiModelResults}
                    />
                    <button
                      onClick={fillTestData}
                      disabled={!designData}
                      className="flex items-center space-x-1 px-3 py-1 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm disabled:opacity-50 disabled:cursor-not-allowed"
                      title="Fill response column with test data"
                    >
                      <Target className="w-4 h-4" />
                      <span>Fill Test Data</span>
                    </button>
                    <button
                      onClick={addRow}
                      className="flex items-center space-x-1 px-3 py-1 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition-colors text-sm"
                    >
                      <Plus className="w-4 h-4" />
                      <span>Add Row</span>
                    </button>
                  </div>
                </div>

                <div className="overflow-x-auto bg-slate-700/30 rounded-lg border-2 border-slate-600">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr className="bg-slate-700/70">
                        <th className="px-3 py-2 text-center text-gray-100 font-semibold text-sm border-b-2 border-r border-slate-600 w-14">
                          #
                        </th>
                        {factorNames.map((factor, idx) => (
                          <th
                            key={idx}
                            className="px-3 py-2 text-center text-gray-100 font-semibold text-sm border-b-2 border-r border-slate-600 min-w-[80px]"
                          >
                            {factor}
                          </th>
                        ))}
                        {responseNames.map((respName, idx) => {
                          return (
                            <th
                              key={`resp-${idx}`}
                              className="px-3 py-2 text-center text-gray-100 font-semibold text-sm border-b-2 border-r border-slate-600 min-w-[100px] bg-orange-900/20"
                            >
                              {respName}
                            </th>
                          )
                        })}
                        <th className="px-3 py-2 text-center text-gray-100 font-semibold text-sm border-b-2 border-slate-600 w-16"></th>
                      </tr>
                    </thead>
                    <tbody>
                      {tableData.map((row, rowIndex) => (
                        <tr key={rowIndex} className="border-b border-slate-700/30 hover:bg-slate-600/10">
                          <td className="px-3 py-2 text-center text-gray-300 text-sm font-medium bg-slate-700/30 border-r border-slate-600">
                            {rowIndex + 1}
                          </td>
                          {row.map((cell, colIndex) => (
                            <td key={colIndex} className="px-1 py-1 border-r border-slate-700/20">
                              <input
                                type="text"
                                value={cell}
                                onChange={(e) => handleCellChange(rowIndex, colIndex, e.target.value)}
                                disabled={colIndex < factorNames.length && designData}
                                className="w-full px-2 py-1.5 bg-slate-800/50 text-gray-100 border border-slate-600/50 focus:border-orange-500 focus:bg-slate-700/50 hover:border-slate-500 rounded-sm focus:outline-none focus:ring-2 focus:ring-orange-500/50 text-sm transition-all disabled:opacity-60"
                                placeholder={colIndex === row.length - 1 ? '0.0' : ''}
                              />
                            </td>
                          ))}
                          <td className="px-2 py-2 text-center">
                            <button
                              onClick={() => removeRow(rowIndex)}
                              disabled={tableData.length === 1}
                              className="p-1 text-red-400 hover:text-red-300 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <p className="text-gray-400 text-xs mt-2">
                  Enter response values for each run. Factor levels from the design cannot be modified.
                </p>

                {multiResponseMode ? (
                  <button
                    onClick={handleFitMultiModel}
                    disabled={loading}
                    className="w-full mt-4 bg-green-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                  >
                    {loading ? 'Fitting Models...' : `Fit ${responseNames.length} Response Surface Models`}
                  </button>
                ) : (
                  <button
                    onClick={handleFitModel}
                    disabled={loading}
                    className="w-full mt-4 bg-green-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {loading ? 'Fitting Model...' : 'Fit Response Surface Model'}
                  </button>
                )}
              </div>
            )}

            {/* Design Augmentation (Feature 5) */}
            {tableData.length > 0 && (
              <div className="mt-6 bg-gradient-to-r from-cyan-900/30 to-blue-900/30 backdrop-blur-lg rounded-2xl p-6 border border-cyan-700/50">
                <h4 className="text-xl font-bold text-gray-100 mb-4">Design Augmentation</h4>
                <p className="text-gray-300 text-sm mb-4">
                  Add more experimental runs to your existing design using sequential strategies.
                </p>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                  <div>
                    <label className="block text-gray-100 font-medium mb-2">Strategy</label>
                    <select
                      value={augmentationStrategy}
                      onChange={(e) => setAugmentationStrategy(e.target.value)}
                      className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                    >
                      <option value="space-filling">Space-Filling</option>
                      <option value="steep-ascent">Steepest Ascent</option>
                      <option value="model-based">Model-Based</option>
                    </select>
                    <p className="text-gray-400 text-xs mt-1">
                      {augmentationStrategy === 'space-filling' && 'Maximize coverage of design space'}
                      {augmentationStrategy === 'steep-ascent' && 'Follow gradient toward optimum'}
                      {augmentationStrategy === 'model-based' && 'Improve model fit in uncertain regions'}
                    </p>
                  </div>

                  <div>
                    <label className="block text-gray-100 font-medium mb-2">Number of New Runs</label>
                    <input
                      type="number"
                      min="1"
                      max="20"
                      value={nAugmentRuns}
                      onChange={(e) => setNAugmentRuns(parseInt(e.target.value) || 5)}
                      className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                    />
                  </div>

                  <div className="flex items-end">
                    <button
                      onClick={handleDesignAugmentation}
                      disabled={loading || (augmentationStrategy !== 'space-filling' && !modelResult)}
                      className="w-full bg-cyan-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-cyan-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {loading ? 'Augmenting...' : 'Add Runs'}
                    </button>
                  </div>
                </div>

                {augmentationResult && (
                  <div className="bg-slate-700/30 rounded-lg p-4">
                    <p className="text-green-300 font-medium">
                      ✓ Added {augmentationResult.n_new_runs} runs using {augmentationResult.strategy} strategy.
                      New runs have been appended to the design table.
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === 'model' && modelResult && (
        <div className="space-y-6">
          {/* Model Summary */}
          <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
            <h3 className="text-2xl font-bold text-gray-100 mb-4">Second-Order Model</h3>
            <div className="grid grid-cols-3 gap-4 mb-6">
              <div className="bg-blue-900/20 rounded-lg p-4 border border-blue-700/30">
                <p className="text-blue-200 text-sm">R-Squared</p>
                <p className="text-2xl font-bold text-blue-100">{modelResult.r_squared}</p>
              </div>
              <div className="bg-green-900/20 rounded-lg p-4 border border-green-700/30">
                <p className="text-green-200 text-sm">Adj R-Squared</p>
                <p className="text-2xl font-bold text-green-100">{modelResult.adj_r_squared}</p>
              </div>
              <div className="bg-purple-900/20 rounded-lg p-4 border border-purple-700/30">
                <p className="text-purple-200 text-sm">RMSE</p>
                <p className="text-2xl font-bold text-purple-100">{modelResult.rmse}</p>
              </div>
            </div>

            {/* Coefficients Table */}
            <h4 className="text-gray-100 font-semibold mb-3">Model Coefficients</h4>
            <div className="overflow-x-auto bg-slate-700/30 rounded-lg">
              <table className="w-full">
                <thead>
                  <tr className="bg-slate-700/70">
                    <th className="px-4 py-2 text-left text-gray-100 font-semibold border-b border-slate-600">Term</th>
                    <th className="px-4 py-2 text-right text-gray-100 font-semibold border-b border-slate-600">Estimate</th>
                    <th className="px-4 py-2 text-right text-gray-100 font-semibold border-b border-slate-600">Std Error</th>
                    <th className="px-4 py-2 text-right text-gray-100 font-semibold border-b border-slate-600">t-value</th>
                    <th className="px-4 py-2 text-right text-gray-100 font-semibold border-b border-slate-600">p-value</th>
                    <th className="px-4 py-2 text-center text-gray-100 font-semibold border-b border-slate-600">Sig.</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(modelResult.coefficients).map(([term, values], idx) => (
                    <tr key={idx} className="border-b border-slate-700/30 hover:bg-slate-600/10">
                      <td className="px-4 py-2 text-gray-100 font-mono text-sm">{term}</td>
                      <td className="px-4 py-2 text-right text-gray-100">{values.estimate}</td>
                      <td className="px-4 py-2 text-right text-gray-300">{values.std_error}</td>
                      <td className="px-4 py-2 text-right text-gray-100">{values.t_value}</td>
                      <td className="px-4 py-2 text-right text-gray-100">{values.p_value}</td>
                      <td className="px-4 py-2 text-center">
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          values.p_value < alpha ? 'bg-green-900/50 text-green-200' : 'bg-slate-700 text-gray-400'
                        }`}>
                          {values.p_value < alpha ? '***' : 'ns'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Canonical Analysis */}
          {canonicalResult && !canonicalResult.error && (
            <div className="bg-gradient-to-r from-purple-900/30 to-indigo-900/30 backdrop-blur-lg rounded-2xl p-6 border border-purple-700/50">
              <h3 className="text-2xl font-bold text-gray-100 mb-4">Canonical Analysis</h3>

              <div className="mb-6">
                <div className={`inline-block px-4 py-2 rounded-lg font-bold text-lg ${
                  canonicalResult.surface_type === 'Maximum' ? 'bg-green-900/50 text-green-200'
                  : canonicalResult.surface_type === 'Minimum' ? 'bg-blue-900/50 text-blue-200'
                  : 'bg-orange-900/50 text-orange-200'
                }`}>
                  Surface Type: {canonicalResult.surface_type}
                </div>
              </div>

              <h4 className="text-gray-100 font-semibold mb-2">Stationary Point:</h4>
              <div className="grid grid-cols-3 gap-3 mb-6">
                {Object.entries(canonicalResult.stationary_point).map(([factor, value]) => (
                  <div key={factor} className="bg-slate-700/50 rounded-lg p-3">
                    <p className="text-gray-400 text-sm">{factor}</p>
                    <p className="text-xl font-bold text-gray-100">{value}</p>
                  </div>
                ))}
              </div>

              <h4 className="text-gray-100 font-semibold mb-2">Eigenvalues:</h4>
              <div className="flex flex-wrap gap-2">
                {canonicalResult.eigenvalues.map((eigenvalue, idx) => (
                  <div key={idx} className={`px-3 py-1 rounded-lg ${
                    eigenvalue < 0 ? 'bg-red-900/50 text-red-200' : 'bg-green-900/50 text-green-200'
                  }`}>
                    λ{idx + 1} = {eigenvalue}
                  </div>
                ))}
              </div>

              <p className="text-gray-300 text-sm mt-4">
                <strong>Interpretation:</strong> {canonicalResult.interpretation}
              </p>
            </div>
          )}

          {/* Curvature Test */}
          {modelResult.curvature_test && (
            <div className="bg-slate-800/50 rounded-lg p-5 border border-slate-700/50">
              <h4 className="text-gray-100 font-semibold mb-3">Curvature Detection</h4>
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-slate-700/50 rounded-lg p-3">
                  <p className="text-gray-400 text-sm">F-Statistic</p>
                  <p className="text-xl font-bold text-gray-100">{modelResult.curvature_test.f_statistic}</p>
                </div>
                <div className="bg-slate-700/50 rounded-lg p-3">
                  <p className="text-gray-400 text-sm">p-value</p>
                  <p className="text-xl font-bold text-gray-100">{modelResult.curvature_test.p_value}</p>
                </div>
                <div className="bg-slate-700/50 rounded-lg p-3">
                  <p className="text-gray-400 text-sm">Significant Curvature?</p>
                  <p className={`text-xl font-bold ${modelResult.curvature_test.significant_curvature ? 'text-green-400' : 'text-gray-400'}`}>
                    {modelResult.curvature_test.significant_curvature ? 'Yes' : 'No'}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Enhanced ANOVA Table */}
          {modelResult.enhanced_anova && (
            <EnhancedANOVA
              enhancedAnova={modelResult.enhanced_anova}
              lackOfFitTest={modelResult.lack_of_fit_test}
              alpha={alpha}
            />
          )}

          {/* Residual Analysis */}
          {modelResult.diagnostics && (
            <ResidualAnalysis
              diagnostics={modelResult.diagnostics}
              responseName={responseName}
            />
          )}

          {/* Advanced Model Diagnostics (Phase 1 Feature) */}
          {advancedDiagnostics && (
            <div className="bg-gradient-to-r from-emerald-900/30 to-teal-900/30 backdrop-blur-lg rounded-2xl p-6 border border-emerald-700/50">
              <h3 className="text-2xl font-bold text-gray-100 mb-4">Advanced Model Diagnostics</h3>
              <p className="text-gray-300 text-sm mb-6">
                Comprehensive diagnostics including leverage, Cook's Distance, DFFITS, VIF, and PRESS statistics to identify influential observations and multicollinearity.
              </p>
              <AdvancedDiagnostics diagnosticsData={advancedDiagnostics} />
            </div>
          )}

          {/* K-fold Cross-Validation (Phase 2 Feature) */}
          <div className="bg-gradient-to-r from-indigo-900/30 to-blue-900/30 backdrop-blur-lg rounded-2xl p-6 border border-indigo-700/50">
            <h3 className="text-2xl font-bold text-gray-100 mb-4">Model Validation (K-Fold Cross-Validation)</h3>
            <p className="text-gray-300 text-sm mb-6">
              Validate your model's predictive performance using K-fold cross-validation. This technique provides a robust estimate of how well your model generalizes to unseen data by repeatedly training on different subsets of your data.
            </p>

            {/* Configuration */}
            {!cvResults && (
              <div className="mb-6 bg-slate-700/30 rounded-lg p-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-end">
                  <div>
                    <label className="block text-gray-100 font-medium mb-2">Number of Folds (K)</label>
                    <select
                      value={kFolds}
                      onChange={(e) => setKFolds(parseInt(e.target.value))}
                      disabled={cvLoading}
                      className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:opacity-50"
                    >
                      <option value={3}>3-Fold CV</option>
                      <option value={5}>5-Fold CV (Recommended)</option>
                      <option value={10}>10-Fold CV</option>
                    </select>
                    <p className="text-gray-400 text-xs mt-1">
                      Higher K = more reliable but slower. 5-fold is a good balance.
                    </p>
                  </div>

                  <button
                    onClick={handleCrossValidation}
                    disabled={cvLoading || !tableData.length}
                    className="flex items-center justify-center gap-2 bg-indigo-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <Target className="w-5 h-5" />
                    <span>{cvLoading ? 'Running Cross-Validation...' : 'Run Cross-Validation'}</span>
                  </button>
                </div>
              </div>
            )}

            {/* Results */}
            <CrossValidationResults
              cvResults={cvResults}
              loading={cvLoading}
              error={cvError}
            />

            {/* Reset button if results are shown */}
            {cvResults && !cvLoading && (
              <button
                onClick={() => {
                  setCvResults(null)
                  setCvError(null)
                }}
                className="mt-4 text-indigo-400 hover:text-indigo-300 text-sm font-medium transition-colors"
              >
                Run New Cross-Validation
              </button>
            )}
          </div>

          {/* Export to Industry Standards (Phase 1 Feature) */}
          <div className="bg-gradient-to-r from-violet-900/30 to-fuchsia-900/30 backdrop-blur-lg rounded-2xl p-6 border border-violet-700/50">
            <h3 className="text-2xl font-bold text-gray-100 mb-4">Export to Industry Standards</h3>
            <p className="text-gray-300 text-sm mb-6">
              Export your complete RSM analysis to industry-standard formats. Share with colleagues or continue analysis in other tools.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <button
                onClick={() => handleExport('pdf')}
                className="flex flex-col items-center justify-center bg-red-600 hover:bg-red-700 text-white px-6 py-4 rounded-xl font-semibold transition-all duration-200 hover:scale-105"
              >
                <span className="text-2xl mb-2">📄</span>
                <span>PDF Report</span>
                <span className="text-xs opacity-80 mt-1">Complete Report (.pdf)</span>
              </button>

              <button
                onClick={() => handleExport('jmp')}
                className="flex flex-col items-center justify-center bg-orange-600 hover:bg-orange-700 text-white px-6 py-4 rounded-xl font-semibold transition-all duration-200 hover:scale-105"
              >
                <span className="text-2xl mb-2">📊</span>
                <span>Export to JMP</span>
                <span className="text-xs opacity-80 mt-1">JSL Script (.jsl)</span>
              </button>

              <button
                onClick={() => handleExport('r')}
                className="flex flex-col items-center justify-center bg-blue-600 hover:bg-blue-700 text-white px-6 py-4 rounded-xl font-semibold transition-all duration-200 hover:scale-105"
              >
                <span className="text-2xl mb-2">📈</span>
                <span>Export to R</span>
                <span className="text-xs opacity-80 mt-1">R Script (.R)</span>
              </button>

              <button
                onClick={() => handleExport('python')}
                className="flex flex-col items-center justify-center bg-green-600 hover:bg-green-700 text-white px-6 py-4 rounded-xl font-semibold transition-all duration-200 hover:scale-105"
              >
                <span className="text-2xl mb-2">🐍</span>
                <span>Export to Python</span>
                <span className="text-xs opacity-80 mt-1">Python Script (.py)</span>
              </button>
            </div>

            <div className="mt-4 bg-slate-700/30 rounded-lg p-4">
              <p className="text-gray-300 text-sm">
                <strong>What's exported:</strong> Complete analysis including data, model formula, coefficients, ANOVA, diagnostics, and visualization code ready to run in your preferred tool. PDF report includes publication-ready tables and summaries.
              </p>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'optimize' && modelResult && (
        <div className="space-y-6">
          {/* Optimization Controls */}
          <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
            <h3 className="text-2xl font-bold text-gray-100 mb-6">Response Optimization</h3>

            <div className="grid grid-cols-2 gap-4 mb-6">
              <button
                onClick={() => handleOptimize('maximize')}
                disabled={loading}
                className="flex items-center justify-center space-x-2 bg-green-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50"
              >
                <Target className="w-5 h-5" />
                <span>Maximize Response</span>
              </button>
              <button
                onClick={() => handleOptimize('minimize')}
                disabled={loading}
                className="flex items-center justify-center space-x-2 bg-blue-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
              >
                <Target className="w-5 h-5" />
                <span>Minimize Response</span>
              </button>
            </div>

            <button
              onClick={handleSteepestAscent}
              disabled={loading}
              className="w-full flex items-center justify-center space-x-2 bg-purple-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-purple-700 transition-colors disabled:opacity-50"
            >
              <TrendingUp className="w-5 h-5" />
              <span>Calculate Steepest Ascent Path</span>
            </button>
          </div>

          {/* Constrained Optimization (Feature 6) */}
          <div className="bg-gradient-to-r from-amber-900/30 to-orange-900/30 backdrop-blur-lg rounded-2xl p-6 border border-amber-700/50">
            <h3 className="text-2xl font-bold text-gray-100 mb-4">Constrained Optimization</h3>
            <p className="text-gray-300 text-sm mb-4">
              Optimize response subject to box constraints (factor bounds) and/or linear constraints.
            </p>

            {/* Box Constraints */}
            <div className="mb-6">
              <h4 className="text-gray-100 font-semibold mb-3">Box Constraints (Optional)</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {factorNames.map((factor, idx) => (
                  <div key={idx} className="bg-slate-700/30 rounded-lg p-3">
                    <p className="text-gray-300 font-medium mb-2">{factor}</p>
                    <div className="flex items-center gap-2">
                      <input
                        type="number"
                        step="0.1"
                        placeholder="Min"
                        value={constraints.box[factor]?.[0] || ''}
                        onChange={(e) => {
                          const newConstraints = {...constraints}
                          if (!newConstraints.box[factor]) newConstraints.box[factor] = [null, null]
                          newConstraints.box[factor][0] = e.target.value ? parseFloat(e.target.value) : null
                          setConstraints(newConstraints)
                        }}
                        className="w-full px-3 py-1.5 bg-slate-800/50 text-gray-100 rounded border border-slate-600 focus:border-amber-500 focus:outline-none text-sm"
                      />
                      <span className="text-gray-400">≤</span>
                      <span className="text-gray-100 font-medium">{factor}</span>
                      <span className="text-gray-400">≤</span>
                      <input
                        type="number"
                        step="0.1"
                        placeholder="Max"
                        value={constraints.box[factor]?.[1] || ''}
                        onChange={(e) => {
                          const newConstraints = {...constraints}
                          if (!newConstraints.box[factor]) newConstraints.box[factor] = [null, null]
                          newConstraints.box[factor][1] = e.target.value ? parseFloat(e.target.value) : null
                          setConstraints(newConstraints)
                        }}
                        className="w-full px-3 py-1.5 bg-slate-800/50 text-gray-100 rounded border border-slate-600 focus:border-amber-500 focus:outline-none text-sm"
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Optimization Buttons */}
            <div className="grid grid-cols-2 gap-4">
              <button
                onClick={() => handleConstrainedOptimization('maximize')}
                disabled={loading}
                className="flex items-center justify-center space-x-2 bg-amber-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-amber-700 transition-colors disabled:opacity-50"
              >
                <Target className="w-5 h-5" />
                <span>Maximize (Constrained)</span>
              </button>
              <button
                onClick={() => handleConstrainedOptimization('minimize')}
                disabled={loading}
                className="flex items-center justify-center space-x-2 bg-orange-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-orange-700 transition-colors disabled:opacity-50"
              >
                <Target className="w-5 h-5" />
                <span>Minimize (Constrained)</span>
              </button>
            </div>

            <p className="text-gray-400 text-xs mt-3">
              Leave constraint fields empty for unconstrained factors. Linear constraints (e.g., X1 + X2 ≤ 5) are not currently supported in the UI.
            </p>
          </div>

          {/* Constrained Optimization Results */}
          {constrainedOptResult && (
            <div className="bg-gradient-to-r from-amber-900/30 to-yellow-900/30 backdrop-blur-lg rounded-2xl p-6 border border-amber-700/50">
              <h3 className="text-2xl font-bold text-gray-100 mb-4">
                Constrained Optimization Results ({constrainedOptResult.target})
              </h3>

              <div className="mb-6">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="bg-slate-700/50 rounded-lg p-4">
                    <p className="text-gray-400 text-sm">Predicted Response</p>
                    <p className="text-3xl font-bold text-amber-300">{constrainedOptResult.predicted_response}</p>
                  </div>
                  {constrainedOptResult.intervals && (
                    <>
                      <div className="bg-slate-700/50 rounded-lg p-4">
                        <p className="text-gray-400 text-sm">95% Confidence Interval</p>
                        <p className="text-lg font-semibold text-blue-300">
                          {constrainedOptResult.intervals.confidence_interval.lower} to {constrainedOptResult.intervals.confidence_interval.upper}
                        </p>
                        <p className="text-xs text-gray-400 mt-1">Mean response</p>
                      </div>
                      <div className="bg-slate-700/50 rounded-lg p-4">
                        <p className="text-gray-400 text-sm">95% Prediction Interval</p>
                        <p className="text-lg font-semibold text-purple-300">
                          {constrainedOptResult.intervals.prediction_interval.lower} to {constrainedOptResult.intervals.prediction_interval.upper}
                        </p>
                        <p className="text-xs text-gray-400 mt-1">Single observation</p>
                      </div>
                    </>
                  )}
                </div>
              </div>

              <h4 className="text-gray-100 font-semibold mb-3">Optimal Factor Settings (Constrained):</h4>
              <div className="grid grid-cols-3 gap-4 mb-4">
                {Object.entries(constrainedOptResult.optimal_point).map(([factor, value]) => (
                  <div key={factor} className="bg-slate-700/50 rounded-lg p-4">
                    <p className="text-gray-400 text-sm">{factor}</p>
                    <p className="text-2xl font-bold text-gray-100">{value}</p>
                  </div>
                ))}
              </div>

              <div className="bg-slate-700/30 rounded-lg p-4">
                <p className="text-gray-300 text-sm">
                  <strong>Method:</strong> {constrainedOptResult.method}
                </p>
                <p className="text-gray-300 text-sm mt-2">
                  <strong>Feasible:</strong> <span className="text-green-400">Yes</span> (solution satisfies all constraints)
                </p>
                {constrainedOptResult.active_constraints && constrainedOptResult.active_constraints.length > 0 && (
                  <p className="text-gray-300 text-sm mt-2">
                    <strong>Active Constraints:</strong> {constrainedOptResult.active_constraints.join(', ')}
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Optimization Results */}
          {optimizationResult && (
            <div className="bg-gradient-to-r from-green-900/30 to-blue-900/30 backdrop-blur-lg rounded-2xl p-6 border border-green-700/50">
              <h3 className="text-2xl font-bold text-gray-100 mb-4">
                Optimization Results ({optimizationResult.target})
              </h3>

              <div className="mb-6">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="bg-slate-700/50 rounded-lg p-4">
                    <p className="text-gray-400 text-sm">Predicted Response</p>
                    <p className="text-3xl font-bold text-green-300">{optimizationResult.predicted_response}</p>
                  </div>
                  {optimizationResult.intervals && (
                    <>
                      <div className="bg-slate-700/50 rounded-lg p-4">
                        <p className="text-gray-400 text-sm">95% Confidence Interval</p>
                        <p className="text-lg font-semibold text-blue-300">
                          {optimizationResult.intervals.confidence_interval.lower} to {optimizationResult.intervals.confidence_interval.upper}
                        </p>
                        <p className="text-xs text-gray-400 mt-1">Mean response</p>
                      </div>
                      <div className="bg-slate-700/50 rounded-lg p-4">
                        <p className="text-gray-400 text-sm">95% Prediction Interval</p>
                        <p className="text-lg font-semibold text-purple-300">
                          {optimizationResult.intervals.prediction_interval.lower} to {optimizationResult.intervals.prediction_interval.upper}
                        </p>
                        <p className="text-xs text-gray-400 mt-1">Single observation</p>
                      </div>
                    </>
                  )}
                </div>
              </div>

              <h4 className="text-gray-100 font-semibold mb-3">Optimal Factor Settings:</h4>
              <div className="grid grid-cols-3 gap-4">
                {Object.entries(optimizationResult.optimal_point).map(([factor, value]) => (
                  <div key={factor} className="bg-slate-700/50 rounded-lg p-4">
                    <p className="text-gray-400 text-sm">{factor}</p>
                    <p className="text-2xl font-bold text-gray-100">{value}</p>
                  </div>
                ))}
              </div>

              <p className="text-gray-300 text-sm mt-4">
                <strong>Method:</strong> {optimizationResult.method}
              </p>

              {/* Confirmation Runs Button */}
              <div className="mt-6 flex items-center gap-4">
                <input
                  type="number"
                  value={nConfirmationRuns}
                  onChange={(e) => setNConfirmationRuns(parseInt(e.target.value) || 3)}
                  min="1"
                  max="10"
                  className="px-4 py-2 bg-slate-700 text-gray-100 rounded-lg w-24"
                />
                <button
                  onClick={handleConfirmationRuns}
                  disabled={loading}
                  className="px-6 py-3 bg-gradient-to-r from-cyan-600 to-blue-600 text-white rounded-lg font-semibold hover:from-cyan-700 hover:to-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
                >
                  <span>Calculate Confirmation Runs</span>
                </button>
              </div>
            </div>
          )}

          {/* Confirmation Runs Results */}
          {confirmationRunsResult && (
            <div className="bg-gradient-to-r from-cyan-900/30 to-teal-900/30 backdrop-blur-lg rounded-2xl p-6 border border-cyan-700/50">
              <h3 className="text-2xl font-bold text-gray-100 mb-4">Confirmation Runs</h3>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div className="bg-slate-700/50 rounded-lg p-4">
                  <p className="text-gray-400 text-sm">Predicted Response</p>
                  <p className="text-2xl font-bold text-cyan-300">{confirmationRunsResult.predicted_response}</p>
                </div>
                {confirmationRunsResult.prediction_interval && (
                  <>
                    <div className="bg-slate-700/50 rounded-lg p-4">
                      <p className="text-gray-400 text-sm">95% PI Lower</p>
                      <p className="text-2xl font-bold text-gray-100">{confirmationRunsResult.prediction_interval.lower}</p>
                    </div>
                    <div className="bg-slate-700/50 rounded-lg p-4">
                      <p className="text-gray-400 text-sm">95% PI Upper</p>
                      <p className="text-2xl font-bold text-gray-100">{confirmationRunsResult.prediction_interval.upper}</p>
                    </div>
                  </>
                )}
              </div>

              <h4 className="text-gray-100 font-semibold mb-3">Recommended Confirmation Runs:</h4>
              <div className="overflow-x-auto bg-slate-700/30 rounded-lg">
                <table className="w-full">
                  <thead>
                    <tr className="bg-slate-700/70">
                      <th className="px-4 py-2 text-center text-gray-100 font-semibold border-b border-slate-600">Run</th>
                      {factorNames.map((factor, idx) => (
                        <th key={idx} className="px-4 py-2 text-center text-gray-100 font-semibold border-b border-slate-600">
                          {factor}
                        </th>
                      ))}
                      <th className="px-4 py-2 text-center text-gray-100 font-semibold border-b border-slate-600">Predicted</th>
                    </tr>
                  </thead>
                  <tbody>
                    {confirmationRunsResult.confirmation_runs.map((run, idx) => (
                      <tr key={idx} className="border-b border-slate-700/30 hover:bg-slate-600/10">
                        <td className="px-4 py-2 text-center text-gray-100 font-bold">{run.run_number}</td>
                        {factorNames.map((factor, fIdx) => (
                          <td key={fIdx} className="px-4 py-2 text-center text-gray-100">
                            {run[factor]}
                          </td>
                        ))}
                        <td className="px-4 py-2 text-center text-cyan-300 font-semibold">{run.predicted_response}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {confirmationRunsResult.recommendations && (
                <div className="mt-6 bg-slate-700/30 rounded-lg p-4">
                  <h5 className="font-semibold text-gray-100 mb-2">Recommendations:</h5>
                  <ul className="list-disc list-inside space-y-1 text-gray-300 text-sm">
                    {confirmationRunsResult.recommendations.map((rec, idx) => (
                      <li key={idx}>{rec}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* Steepest Ascent Results */}
          {steepestAscentResult && (
            <div className="bg-gradient-to-r from-purple-900/30 to-pink-900/30 backdrop-blur-lg rounded-2xl p-6 border border-purple-700/50">
              <h3 className="text-2xl font-bold text-gray-100 mb-4">Steepest Ascent Path</h3>

              <div className="mb-4">
                <p className="text-gray-300">
                  <strong>Gradient Magnitude:</strong> {steepestAscentResult.gradient_magnitude}
                </p>
              </div>

              <h4 className="text-gray-100 font-semibold mb-3">Path of Steepest Ascent:</h4>
              <div className="overflow-x-auto bg-slate-700/30 rounded-lg">
                <table className="w-full">
                  <thead>
                    <tr className="bg-slate-700/70">
                      <th className="px-4 py-2 text-center text-gray-100 font-semibold border-b border-slate-600">Step</th>
                      {factorNames.map((factor, idx) => (
                        <th key={idx} className="px-4 py-2 text-center text-gray-100 font-semibold border-b border-slate-600">
                          {factor}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {steepestAscentResult.path.map((point, idx) => (
                      <tr key={idx} className="border-b border-slate-700/30 hover:bg-slate-600/10">
                        <td className="px-4 py-2 text-center text-gray-100 font-bold">{point.step}</td>
                        {factorNames.map((factor, fIdx) => (
                          <td key={fIdx} className="px-4 py-2 text-center text-gray-100">
                            {point[factor]}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <p className="text-gray-300 text-sm mt-4">
                <strong>Interpretation:</strong> Follow this path from the current design center to move in the direction of maximum increase in the response. Conduct experiments at these points to verify improvement.
              </p>
            </div>
          )}

          {/* Ridge Analysis Controls */}
          {modelResult && numFactors === 2 && (
            <div className="bg-gradient-to-r from-indigo-900/30 to-violet-900/30 backdrop-blur-lg rounded-2xl p-6 border border-indigo-700/50">
              <h3 className="text-2xl font-bold text-gray-100 mb-4">Ridge Analysis</h3>
              <p className="text-gray-300 text-sm mb-4">
                Find all factor combinations that achieve a specific target response value.
              </p>

              <div className="flex items-center gap-4">
                <div>
                  <label className="block text-gray-400 text-sm mb-2">Target Response:</label>
                  <input
                    type="number"
                    step="0.1"
                    value={targetResponse}
                    onChange={(e) => setTargetResponse(e.target.value)}
                    placeholder="Enter target value"
                    className="px-4 py-2 bg-slate-700 text-gray-100 rounded-lg w-40"
                  />
                </div>
                <button
                  onClick={handleRidgeAnalysis}
                  disabled={loading || !targetResponse}
                  className="mt-6 px-6 py-3 bg-gradient-to-r from-indigo-600 to-violet-600 text-white rounded-lg font-semibold hover:from-indigo-700 hover:to-violet-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
                >
                  <span>Calculate Ridge</span>
                </button>
              </div>
            </div>
          )}

          {/* Ridge Analysis Results */}
          {ridgeAnalysisResult && (
            <div className="bg-gradient-to-r from-indigo-900/30 to-violet-900/30 backdrop-blur-lg rounded-2xl p-6 border border-indigo-700/50">
              <h3 className="text-2xl font-bold text-gray-100 mb-4">
                Ridge Analysis Results
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div className="bg-slate-700/50 rounded-lg p-4">
                  <p className="text-gray-400 text-sm">Target Response</p>
                  <p className="text-2xl font-bold text-indigo-300">{ridgeAnalysisResult.target_response}</p>
                </div>
                {ridgeAnalysisResult.stationary_response && (
                  <div className="bg-slate-700/50 rounded-lg p-4">
                    <p className="text-gray-400 text-sm">Stationary Response</p>
                    <p className="text-2xl font-bold text-gray-100">{ridgeAnalysisResult.stationary_response}</p>
                  </div>
                )}
                {ridgeAnalysisResult.distance_to_target && (
                  <div className="bg-slate-700/50 rounded-lg p-4">
                    <p className="text-gray-400 text-sm">Distance to Target</p>
                    <p className="text-2xl font-bold text-gray-100">{ridgeAnalysisResult.distance_to_target}</p>
                  </div>
                )}
              </div>

              {ridgeAnalysisResult.ridge_points && ridgeAnalysisResult.ridge_points.length > 0 && (
                <div className="mb-4">
                  <p className="text-gray-300 text-sm">
                    <strong>Ridge Points:</strong> {ridgeAnalysisResult.n_points} factor combinations found
                  </p>
                </div>
              )}

              <p className="text-gray-300 text-sm mt-4 bg-slate-700/30 rounded-lg p-4">
                <strong>Interpretation:</strong> {ridgeAnalysisResult.interpretation}
              </p>
            </div>
          )}

          {/* Desirability Functions (Feature 4) */}
          <div className="bg-gradient-to-r from-pink-900/30 to-rose-900/30 backdrop-blur-lg rounded-2xl p-6 border border-pink-700/50">
            <h3 className="text-2xl font-bold text-gray-100 mb-4">Multi-Response Optimization (Desirability)</h3>
            <p className="text-gray-300 text-sm mb-4">
              Optimize multiple responses simultaneously using desirability functions. First, save models for each response, then specify desirability criteria.
            </p>

            {/* Save Current Model */}
            {modelResult && (
              <div className="mb-6 bg-slate-700/30 rounded-lg p-4">
                <h4 className="text-gray-100 font-semibold mb-2">Current Model: {responseName}</h4>
                <button
                  onClick={saveCurrentModelToMultiResponse}
                  disabled={multiResponseModels.some(m => m.name === responseName)}
                  className="px-4 py-2 bg-pink-600 text-white rounded-lg hover:bg-pink-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {multiResponseModels.some(m => m.name === responseName) ? 'Already Saved' : 'Save to Multi-Response Collection'}
                </button>
              </div>
            )}

            {/* Saved Models List */}
            {multiResponseModels.length > 0 && (
              <div className="mb-6">
                <h4 className="text-gray-100 font-semibold mb-3">Saved Response Models ({multiResponseModels.length})</h4>
                <div className="space-y-2">
                  {multiResponseModels.map((model, idx) => (
                    <div key={idx} className="bg-slate-700/30 rounded-lg p-3 flex items-center justify-between">
                      <span className="text-gray-100 font-medium">{model.name}</span>
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => {
                            // Add to desirability specs if not already present
                            if (!desirabilitySpecs.some(s => s.response_name === model.name)) {
                              setDesirabilitySpecs([...desirabilitySpecs, {
                                response_name: model.name,
                                coefficients: model.coefficients,
                                goal: 'maximize',
                                lower_bound: null,
                                upper_bound: null,
                                target: null,
                                weight: 1.0,
                                importance: 1.0
                              }])
                            }
                          }}
                          disabled={desirabilitySpecs.some(s => s.response_name === model.name)}
                          className="px-3 py-1 bg-green-600 text-white text-sm rounded hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          {desirabilitySpecs.some(s => s.response_name === model.name) ? 'Added' : 'Add to Desirability'}
                        </button>
                        <button
                          onClick={() => setMultiResponseModels(multiResponseModels.filter((_, i) => i !== idx))}
                          className="px-3 py-1 bg-red-600 text-white text-sm rounded hover:bg-red-700"
                        >
                          Remove
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Desirability Specifications */}
            {desirabilitySpecs.length > 0 && (
              <div className="mb-6">
                <h4 className="text-gray-100 font-semibold mb-3">Desirability Specifications</h4>
                <div className="space-y-4">
                  {desirabilitySpecs.map((spec, idx) => (
                    <div key={idx} className="bg-slate-700/30 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-3">
                        <h5 className="text-gray-100 font-medium">{spec.response_name}</h5>
                        <button
                          onClick={() => setDesirabilitySpecs(desirabilitySpecs.filter((_, i) => i !== idx))}
                          className="px-2 py-1 bg-red-600 text-white text-xs rounded hover:bg-red-700"
                        >
                          Remove
                        </button>
                      </div>

                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        <div>
                          <label className="block text-gray-400 text-xs mb-1">Goal</label>
                          <select
                            value={spec.goal}
                            onChange={(e) => {
                              const newSpecs = [...desirabilitySpecs]
                              newSpecs[idx].goal = e.target.value
                              setDesirabilitySpecs(newSpecs)
                            }}
                            className="w-full px-2 py-1 bg-slate-800 text-gray-100 rounded text-sm border border-slate-600"
                          >
                            <option value="maximize">Maximize</option>
                            <option value="minimize">Minimize</option>
                            <option value="target">Target</option>
                          </select>
                        </div>

                        <div>
                          <label className="block text-gray-400 text-xs mb-1">Lower Bound</label>
                          <input
                            type="number"
                            step="0.1"
                            value={spec.lower_bound || ''}
                            onChange={(e) => {
                              const newSpecs = [...desirabilitySpecs]
                              newSpecs[idx].lower_bound = e.target.value ? parseFloat(e.target.value) : null
                              setDesirabilitySpecs(newSpecs)
                            }}
                            className="w-full px-2 py-1 bg-slate-800 text-gray-100 rounded text-sm border border-slate-600"
                            placeholder="Min"
                          />
                        </div>

                        <div>
                          <label className="block text-gray-400 text-xs mb-1">Upper Bound</label>
                          <input
                            type="number"
                            step="0.1"
                            value={spec.upper_bound || ''}
                            onChange={(e) => {
                              const newSpecs = [...desirabilitySpecs]
                              newSpecs[idx].upper_bound = e.target.value ? parseFloat(e.target.value) : null
                              setDesirabilitySpecs(newSpecs)
                            }}
                            className="w-full px-2 py-1 bg-slate-800 text-gray-100 rounded text-sm border border-slate-600"
                            placeholder="Max"
                          />
                        </div>

                        {spec.goal === 'target' && (
                          <div>
                            <label className="block text-gray-400 text-xs mb-1">Target Value</label>
                            <input
                              type="number"
                              step="0.1"
                              value={spec.target || ''}
                              onChange={(e) => {
                                const newSpecs = [...desirabilitySpecs]
                                newSpecs[idx].target = e.target.value ? parseFloat(e.target.value) : null
                                setDesirabilitySpecs(newSpecs)
                              }}
                              className="w-full px-2 py-1 bg-slate-800 text-gray-100 rounded text-sm border border-slate-600"
                              placeholder="Target"
                            />
                          </div>
                        )}

                        <div>
                          <label className="block text-gray-400 text-xs mb-1">Importance</label>
                          <input
                            type="number"
                            step="0.1"
                            min="0.1"
                            value={spec.importance}
                            onChange={(e) => {
                              const newSpecs = [...desirabilitySpecs]
                              newSpecs[idx].importance = parseFloat(e.target.value) || 1.0
                              setDesirabilitySpecs(newSpecs)
                            }}
                            className="w-full px-2 py-1 bg-slate-800 text-gray-100 rounded text-sm border border-slate-600"
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                <button
                  onClick={handleDesirabilityOptimization}
                  disabled={loading || desirabilitySpecs.length === 0}
                  className="w-full mt-4 bg-gradient-to-r from-pink-600 to-rose-600 text-white font-bold py-3 px-6 rounded-lg hover:from-pink-700 hover:to-rose-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? 'Optimizing...' : 'Optimize Composite Desirability'}
                </button>
              </div>
            )}

            {multiResponseModels.length === 0 && (
              <div className="bg-slate-700/30 rounded-lg p-4 text-gray-400 text-sm">
                To use desirability functions: (1) Fit a model for your first response, (2) Save it, (3) Change your response variable and fit another model, (4) Save that model too, (5) Add desirability specs for each response, then optimize.
              </div>
            )}
          </div>

          {/* Desirability Results */}
          {desirabilityResult && (
            <div className="bg-gradient-to-r from-pink-900/30 to-purple-900/30 backdrop-blur-lg rounded-2xl p-6 border border-pink-700/50">
              <h3 className="text-2xl font-bold text-gray-100 mb-4">Desirability Optimization Results</h3>

              <div className="mb-6">
                <div className="bg-slate-700/50 rounded-lg p-4 inline-block">
                  <p className="text-gray-400 text-sm">Composite Desirability</p>
                  <p className="text-3xl font-bold text-pink-300">{desirabilityResult.composite_desirability.toFixed(4)}</p>
                  <p className="text-gray-400 text-xs mt-1">(0 = undesirable, 1 = perfectly desirable)</p>
                </div>
              </div>

              <h4 className="text-gray-100 font-semibold mb-3">Optimal Factor Settings:</h4>
              <div className="grid grid-cols-3 gap-4 mb-6">
                {Object.entries(desirabilityResult.optimal_point).map(([factor, value]) => (
                  <div key={factor} className="bg-slate-700/50 rounded-lg p-4">
                    <p className="text-gray-400 text-sm">{factor}</p>
                    <p className="text-2xl font-bold text-gray-100">{typeof value === 'number' ? value.toFixed(4) : value}</p>
                  </div>
                ))}
              </div>

              <h4 className="text-gray-100 font-semibold mb-3">Predicted Responses:</h4>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-6">
                {Object.entries(desirabilityResult.predicted_responses).map(([response, value]) => (
                  <div key={response} className="bg-slate-700/50 rounded-lg p-3">
                    <p className="text-gray-400 text-xs">{response}</p>
                    <p className="text-xl font-bold text-pink-200">{typeof value === 'number' ? value.toFixed(4) : value}</p>
                  </div>
                ))}
              </div>

              <h4 className="text-gray-100 font-semibold mb-3">Individual Desirabilities:</h4>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {Object.entries(desirabilityResult.individual_desirabilities).map(([response, value]) => (
                  <div key={response} className="bg-slate-700/50 rounded-lg p-3">
                    <p className="text-gray-400 text-xs">{response}</p>
                    <p className="text-xl font-bold text-purple-200">{typeof value === 'number' ? value.toFixed(4) : value}</p>
                  </div>
                ))}
              </div>

              <div className="mt-6 bg-slate-700/30 rounded-lg p-4">
                <p className="text-gray-300 text-sm">
                  <strong>Interpretation:</strong> The composite desirability is the weighted geometric mean of individual desirabilities.
                  Values closer to 1 indicate the optimal point achieves desirable levels across all responses.
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === 'visualize' && modelResult && (
        <div className="space-y-6">
          {/* Prediction Profiler - Interactive Response Explorer */}
          <div className="bg-gradient-to-r from-purple-900/30 to-indigo-900/30 backdrop-blur-lg rounded-2xl p-6 border border-purple-700/50">
            <div className="mb-4">
              <h3 className="text-2xl font-bold text-gray-100 mb-2">Interactive Prediction Profiler</h3>
              <p className="text-gray-300 text-sm">
                Explore how factor settings affect the predicted response in real-time. Adjust sliders to see predictions, confidence intervals, and trace plots.
              </p>
            </div>
            <PredictionProfiler
              coefficients={modelResult.coefficients}
              factors={factorNames}
              responseName={responseName}
              varianceEstimate={modelResult.anova?.Residual?.mean_sq}
            />
          </div>

          {numFactors === 2 && surfaceData ? (
            // Simple 2-factor visualization
            <>
              <ResponseSurface3D
                surfaceData={surfaceData}
                factor1={factorNames[0]}
                factor2={factorNames[1]}
                responseName={responseName}
              />
              <ContourPlot
                surfaceData={surfaceData}
                factor1={factorNames[0]}
                factor2={factorNames[1]}
                responseName={responseName}
                experimentalData={(() => {
                  // Convert tableData to experimental data format
                  const validRows = tableData.filter(row => {
                    const responseValue = row[row.length - 1]
                    return responseValue !== '' && responseValue !== null && responseValue !== undefined
                  })
                  return validRows.map(row => {
                    const point = {}
                    factorNames.forEach((factor, i) => {
                      point[factor] = parseFloat(row[i])
                    })
                    point[responseName] = parseFloat(row[row.length - 1])
                    return point
                  })
                })()}
                optimizationResult={optimizationResult}
                canonicalResult={canonicalResult}
                steepestAscentResult={steepestAscentResult}
                ridgeAnalysisResult={ridgeAnalysisResult}
              />
            </>
          ) : (
            // Sliced visualization for any number of factors
            <SlicedVisualization
              factorNames={factorNames}
              responseName={responseName}
              coefficients={modelResult.coefficients}
              experimentalData={(() => {
                // Convert tableData to experimental data format
                const validRows = tableData.filter(row => {
                  const responseValue = row[row.length - 1]
                  return responseValue !== '' && responseValue !== null && responseValue !== undefined
                })
                return validRows.map(row => {
                  const point = {}
                  factorNames.forEach((factor, i) => {
                    point[factor] = parseFloat(row[i])
                  })
                  point[responseName] = parseFloat(row[row.length - 1])
                  return point
                })
              })()}
              optimizationResult={optimizationResult}
              canonicalResult={canonicalResult}
            />
          )}
        </div>
      )}

      {/* Multi-Response Visualization (Phase 2 Feature 3) */}
      {activeTab === 'visualize' && multiModelResults && numFactors === 2 && (
        <div className="space-y-6">
          {/* Multi-Response Models Summary */}
          <div className="bg-gradient-to-r from-indigo-900/30 to-purple-900/30 backdrop-blur-lg rounded-2xl p-6 border border-indigo-700/50">
            <h3 className="text-2xl font-bold text-gray-100 mb-4">Multi-Response Surface Models</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-4">
              {Object.entries(multiModelResults.models).map(([respName, model]) => (
                <div key={respName} className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
                  <h4 className="text-lg font-semibold text-gray-100 mb-2">{respName}</h4>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-400">R²:</span>
                      <span className="text-gray-100 font-medium">{model.r_squared.toFixed(4)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Adj. R²:</span>
                      <span className="text-gray-100 font-medium">{model.r_squared_adj.toFixed(4)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">RMSE:</span>
                      <span className="text-gray-100 font-medium">{model.rmse.toFixed(4)}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Generate Multi-Surface Button */}
            {!multiSurfaceData && (
              <button
                onClick={handleGenerateMultiSurface}
                disabled={loading}
                className="w-full bg-indigo-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? 'Generating Surfaces...' : 'Generate Multi-Response Surface Overlay'}
              </button>
            )}

            {/* Normalization Method Selector */}
            {multiSurfaceData && (
              <div className="mt-4 flex items-center gap-4">
                <label className="text-gray-300 font-medium">Normalization:</label>
                <select
                  value={normalizationMethod}
                  onChange={(e) => setNormalizationMethod(e.target.value)}
                  className="px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-gray-100 focus:outline-none focus:border-indigo-500"
                >
                  <option value="none">None (Raw Values)</option>
                  <option value="zscore">Z-Score (Mean=0, Std=1)</option>
                  <option value="minmax">Min-Max [0,1]</option>
                </select>
                <button
                  onClick={handleGenerateMultiSurface}
                  disabled={loading}
                  className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 flex items-center gap-2"
                >
                  Regenerate
                </button>
              </div>
            )}
          </div>

          {/* Multi-Response Contour Overlay */}
          {multiSurfaceData && (
            <MultiResponseContourOverlay
              surfacesData={multiSurfaceData.surfaces}
              responseConfigs={responseConfigs}
              factor1={factorNames[0]}
              factor2={factorNames[1]}
              experimentalData={(() => {
                // Convert tableData to experimental data format for each response
                const result = {}
                const numFactorCols = factorNames.length

                const validRows = tableData.filter(row => {
                  // Check that all response values are filled
                  return responseNames.every((_, idx) => {
                    const responseValue = row[numFactorCols + idx]
                    return responseValue !== '' && responseValue !== null && responseValue !== undefined
                  })
                })

                responseNames.forEach((respName, respIdx) => {
                  result[respName] = validRows.map(row => {
                    const point = {}
                    factorNames.forEach((factor, i) => {
                      point[factor] = parseFloat(row[i])
                    })
                    point[respName] = parseFloat(row[numFactorCols + respIdx])
                    return point
                  })
                })

                return result
              })()}
              viewMode="side-by-side"
            />
          )}
        </div>
      )}

      {/* Experiment Wizard Modal (Phase 2 Feature) */}
      <ExperimentWizard
        isOpen={wizardOpen}
        onClose={() => setWizardOpen(false)}
        onComplete={handleWizardComplete}
      />
    </div>
  )
}

export default RSM
 
