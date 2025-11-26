import { useState, useEffect } from 'react'
import { ChevronRight, ChevronLeft, CheckCircle, Sparkles, Download, AlertCircle, Shuffle, RotateCcw, FileText, FileSpreadsheet, FileCode, ChevronDown } from 'lucide-react'
import axios from 'axios'
import DesignRecommendationStep from '../components/DesignRecommendationStep'
import DesignPreview from '../components/DesignPreview'
import SmartValidation from '../components/SmartValidation'
import SequentialExperimentGuide from '../components/SequentialExperimentGuide'
import InteractiveTooltip, { InlineTooltip } from '../components/InteractiveTooltip'
import FactorInteractionSelector from '../components/FactorInteractionSelector'
import {
  downloadPDF,
  downloadExcel,
  downloadJMP,
  downloadMinitab,
  downloadCSV
} from '../utils/designExport'
import { validateWizardData } from '../utils/smartValidation'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const ExperimentWizardPage = () => {
  const [currentStep, setCurrentStep] = useState(1)
  const [wizardData, setWizardData] = useState({
    goal: '',
    nFactors: 2,
    factorNames: [],
    factorLevels: [], // Array of { min, max, units } for each factor
    selectedInteractions: [], // Array of { factor1, factor2 } for suspected interactions
    powerAnalysis: {
      effectSize: 'medium',
      desiredPower: 0.80,
      minimumRuns: null
    },
    budget: null,
    timeConstraint: null,
    selectedDesign: null
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [generatedDesign, setGeneratedDesign] = useState(null)

  const steps = [
    { id: 1, title: 'Goal', icon: '🎯' },
    { id: 2, title: 'Factors', icon: '🔬' },
    { id: 3, title: 'Power', icon: '⚡' },
    { id: 4, title: 'Constraints', icon: '📊' },
    { id: 5, title: 'Design', icon: '✨' },
    { id: 6, title: 'Review', icon: '✅' }
  ]

  const updateWizardData = (field, value) => {
    setWizardData(prev => ({ ...prev, [field]: value }))
  }

  const nextStep = () => {
    if (currentStep < steps.length) {
      setCurrentStep(currentStep + 1)
    }
  }

  const prevStep = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1)
    }
  }

  // Helper function to transform coded values to actual values
  const transformCodedToActual = (codedValue, min, max) => {
    if (!min || !max || min === '' || max === '') return codedValue
    const minNum = parseFloat(min)
    const maxNum = parseFloat(max)
    const center = (minNum + maxNum) / 2
    const range = (maxNum - minNum) / 2
    return center + codedValue * range
  }

  const handleComplete = async () => {
    console.log('Experiment design completed:', wizardData)
    setLoading(true)
    setError(null)

    try {
      const designCode = wizardData.selectedDesign?.design_code

      // Check if this is a sequential approach (not directly generatable)
      const isSequentialApproach = ['screening-first', 'dsd'].includes(designCode)

      if (isSequentialApproach) {
        alert(
          `📋 Sequential Approach Selected: ${wizardData.selectedDesign.type}\n\n` +
          `This is a two-stage approach:\n` +
          `1. First, run a screening experiment (Factorial Designs page)\n` +
          `2. Then, use RSM on the 2-3 most important factors\n\n` +
          `For direct design generation, please select a CCD or Box-Behnken design.`
        )
        setLoading(false)
        return
      }

      let response
      const numCenterPoints = 4 // Default

      if (designCode === 'box-behnken') {
        response = await axios.post(`${API_URL}/api/rsm/box-behnken/generate`, {
          n_factors: wizardData.nFactors,
          n_center: numCenterPoints
        })
      } else if (['face-centered', 'rotatable', 'inscribed'].includes(designCode)) {
        response = await axios.post(`${API_URL}/api/rsm/ccd/generate`, {
          n_factors: wizardData.nFactors,
          design_type: designCode,
          n_center: numCenterPoints
        })
      } else {
        // Default to face-centered CCD
        response = await axios.post(`${API_URL}/api/rsm/ccd/generate`, {
          n_factors: wizardData.nFactors,
          design_type: 'face-centered',
          n_center: numCenterPoints
        })
      }

      // Prepare factor names
      const factorNames = wizardData.factorNames.filter(n => n && n.trim()).length > 0
        ? wizardData.factorNames.filter(n => n && n.trim())
        : Array.from({ length: wizardData.nFactors }, (_, i) => `X${i + 1}`)

      // Transform coded values to actual values if factor levels are specified
      let designMatrix = response.data.design_matrix
      const hasFactorLevels = wizardData.factorLevels && wizardData.factorLevels.some(level =>
        level && level.min !== '' && level.max !== ''
      )

      if (hasFactorLevels) {
        // Get the original column names from the backend (X1, X2, etc.)
        const backendColumns = designMatrix.length > 0 ? Object.keys(designMatrix[0]) : []

        designMatrix = designMatrix.map(row => {
          const transformedRow = {}
          backendColumns.forEach((backendCol, idx) => {
            const level = wizardData.factorLevels[idx]
            const targetName = factorNames[idx] || backendCol

            if (level && level.min !== '' && level.max !== '') {
              const codedValue = row[backendCol]
              transformedRow[targetName] = transformCodedToActual(codedValue, level.min, level.max)
            } else {
              // No transformation, just rename column
              transformedRow[targetName] = row[backendCol]
            }
          })
          return transformedRow
        })
      } else {
        // No transformation needed, but rename columns to use custom factor names
        const backendColumns = designMatrix.length > 0 ? Object.keys(designMatrix[0]) : []
        designMatrix = designMatrix.map(row => {
          const renamedRow = {}
          backendColumns.forEach((backendCol, idx) => {
            const targetName = factorNames[idx] || backendCol
            renamedRow[targetName] = row[backendCol]
          })
          return renamedRow
        })
      }

      // Store the generated design
      setGeneratedDesign({
        ...response.data,
        design_matrix: designMatrix,
        factorNames: factorNames,
        factorLevels: wizardData.factorLevels,
        useActualValues: hasFactorLevels
      })

      // Move to results view
      setCurrentStep(7) // New step for showing results
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to generate design')
      console.error('Design generation error:', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <div className="max-w-5xl mx-auto py-8">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-900/30 to-purple-900/30 rounded-2xl p-8 mb-8 border border-slate-700/50">
          <div className="flex items-center gap-3 mb-4">
            <Sparkles className="w-10 h-10 text-blue-400" />
            <h1 className="text-4xl font-bold text-gray-100">Experiment Wizard</h1>
          </div>
          <p className="text-gray-300 text-lg">
            Let's design your experiment step by step. We'll recommend the best design for your needs.
          </p>
        </div>

        {/* Progress Steps */}
        <div className="relative z-10 bg-slate-800/50 backdrop-blur-lg rounded-xl px-8 py-6 mb-8 border border-slate-700/50">
          <div className="flex items-center justify-between">
            {steps.map((step, index) => (
              <div key={step.id} className="flex items-center flex-1">
                <div className="flex flex-col items-center flex-1">
                  <div
                    className={`w-14 h-14 rounded-full flex items-center justify-center text-2xl transition-all duration-300 ${
                      step.id === currentStep
                        ? 'bg-blue-600 scale-110 shadow-lg shadow-blue-500/50'
                        : step.id < currentStep
                        ? 'bg-green-600'
                        : 'bg-slate-700'
                    }`}
                  >
                    {step.id < currentStep ? (
                      <CheckCircle className="w-7 h-7 text-white" />
                    ) : (
                      <span>{step.icon}</span>
                    )}
                  </div>
                  <span
                    className={`mt-2 text-sm font-medium ${
                      step.id === currentStep
                        ? 'text-blue-400'
                        : step.id < currentStep
                        ? 'text-green-400'
                        : 'text-gray-400'
                    }`}
                  >
                    {step.title}
                  </span>
                </div>
                {index < steps.length - 1 && (
                  <div
                    className={`h-1 flex-1 mx-2 rounded-full transition-all duration-300 ${
                      step.id < currentStep ? 'bg-green-600' : 'bg-slate-700'
                    }`}
                  />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Content Area */}
        <div className="relative z-10 bg-slate-800/50 backdrop-blur-lg rounded-xl p-8 mb-8 border border-slate-700/50 min-h-[500px]">
          {currentStep === 1 && (
            <GoalSelector
              value={wizardData.goal}
              onChange={(goal) => updateWizardData('goal', goal)}
            />
          )}
          {currentStep === 2 && (
            <FactorConfiguration
              nFactors={wizardData.nFactors}
              factorNames={wizardData.factorNames}
              factorLevels={wizardData.factorLevels}
              selectedInteractions={wizardData.selectedInteractions}
              goal={wizardData.goal}
              onFactorCountChange={(n) => updateWizardData('nFactors', n)}
              onFactorNamesChange={(names) => updateWizardData('factorNames', names)}
              onFactorLevelsChange={(levels) => updateWizardData('factorLevels', levels)}
              onInteractionsChange={(interactions) => updateWizardData('selectedInteractions', interactions)}
            />
          )}
          {currentStep === 3 && (
            <PowerAnalysis
              nFactors={wizardData.nFactors}
              powerAnalysis={wizardData.powerAnalysis}
              onPowerAnalysisChange={(pa) => updateWizardData('powerAnalysis', pa)}
            />
          )}
          {currentStep === 4 && (
            <ConstraintBuilder
              budget={wizardData.budget}
              timeConstraint={wizardData.timeConstraint}
              onBudgetChange={(budget) => updateWizardData('budget', budget)}
              onTimeConstraintChange={(time) => updateWizardData('timeConstraint', time)}
            />
          )}
          {currentStep === 5 && (
            <DesignRecommendationStep
              nFactors={wizardData.nFactors}
              budget={wizardData.budget}
              goal={wizardData.goal}
              minimumRuns={wizardData.powerAnalysis.minimumRuns}
              selectedInteractions={wizardData.selectedInteractions}
              selectedDesign={wizardData.selectedDesign}
              onSelectDesign={(design) => updateWizardData('selectedDesign', design)}
            />
          )}
          {currentStep === 6 && (
            <DesignSummary wizardData={wizardData} />
          )}
          {currentStep === 7 && generatedDesign && (
            <DesignResults design={generatedDesign} wizardData={wizardData} />
          )}
          {error && (
            <div className="bg-red-900/20 border border-red-700/50 rounded-lg p-4">
              <div className="flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-red-200 font-semibold">Error generating design</p>
                  <p className="text-red-300 text-sm mt-1">{error}</p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Navigation Footer - Steps 1-6 */}
        {currentStep !== 7 && (
          <div className="relative z-10 bg-slate-800/50 backdrop-blur-lg rounded-xl px-8 py-6 border border-slate-700/50 flex items-center justify-between">
            <button
              onClick={prevStep}
              disabled={currentStep === 1 || loading}
              className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all ${
                currentStep === 1 || loading
                  ? 'bg-slate-700/30 text-gray-500 cursor-not-allowed'
                  : 'bg-slate-700 text-gray-200 hover:bg-slate-600 hover:scale-105'
              }`}
            >
              <ChevronLeft className="w-5 h-5" />
              Back
            </button>

            <div className="text-gray-400 text-sm font-medium">
              Step {currentStep} of {steps.length}
            </div>

            {currentStep < steps.length && (
              <button
                onClick={nextStep}
                disabled={
                  (currentStep === 1 && !wizardData.goal) ||
                  (currentStep === 2 && wizardData.nFactors < 2) ||
                  (currentStep === 5 && !wizardData.selectedDesign)
                }
                className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all ${
                  (currentStep === 1 && !wizardData.goal) ||
                  (currentStep === 2 && wizardData.nFactors < 2) ||
                  (currentStep === 5 && !wizardData.selectedDesign)
                    ? 'bg-slate-700/30 text-gray-500 cursor-not-allowed'
                    : 'bg-blue-600 text-white hover:bg-blue-700 hover:scale-105'
                }`}
              >
                Next
                <ChevronRight className="w-5 h-5" />
              </button>
            )}

            {currentStep === steps.length && (
              <button
                onClick={handleComplete}
                disabled={loading}
                className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all ${
                  loading
                    ? 'bg-slate-700/30 text-gray-500 cursor-not-allowed'
                    : 'bg-green-600 text-white hover:bg-green-700 hover:scale-105'
                }`}
              >
                {loading ? (
                  <>
                    <div className="w-5 h-5 border-2 border-gray-400 border-t-transparent rounded-full animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <CheckCircle className="w-5 h-5" />
                    Generate Design
                  </>
                )}
              </button>
            )}
          </div>
        )}

        {/* Results Footer - Step 7 */}
        {currentStep === 7 && generatedDesign && (
          <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl px-8 py-6 border border-slate-700/50 flex items-center justify-center">
            <button
              onClick={() => {
                setCurrentStep(1)
                setGeneratedDesign(null)
                setError(null)
                setWizardData({
                  goal: '',
                  nFactors: 2,
                  factorNames: [],
                  factorLevels: [],
                  selectedInteractions: [],
                  powerAnalysis: {
                    effectSize: 'medium',
                    desiredPower: 0.80,
                    minimumRuns: null
                  },
                  budget: null,
                  timeConstraint: null,
                  selectedDesign: null
                })
              }}
              className="flex items-center gap-2 px-6 py-3 rounded-lg font-semibold bg-slate-700 text-gray-200 hover:bg-slate-600 hover:scale-105 transition-all"
            >
              Start New Design
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

// Step 1: Goal Selector
const GoalSelector = ({ value, onChange }) => {
  const goals = [
    {
      id: 'optimization',
      title: 'Optimization',
      tooltipTerm: 'optimization',
      icon: '🎯',
      description: 'Find the best factor settings to maximize or minimize your response',
      examples: 'Maximize yield, minimize cost, optimize quality'
    },
    {
      id: 'screening',
      title: 'Screening',
      tooltipTerm: 'screening',
      icon: '🔍',
      description: 'Identify which factors have the biggest impact on your response',
      examples: 'Which of 5+ factors matter most?'
    },
    {
      id: 'modeling',
      title: 'Response Surface Modeling',
      tooltipTerm: 'rsm',
      icon: '📈',
      description: 'Understand how factors interact and affect your response',
      examples: 'Map the response surface, study curvature'
    }
  ]

  return (
    <div>
      <h3 className="text-2xl font-bold text-gray-100 mb-2">What's your goal?</h3>
      <p className="text-gray-300 text-sm mb-6">
        Select the primary objective of your experiment
      </p>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {goals.map((goal) => (
          <button
            key={goal.id}
            onClick={() => onChange(goal.id)}
            className={`p-6 rounded-xl border-2 transition-all duration-200 text-left ${
              value === goal.id
                ? 'border-blue-500 bg-blue-900/30 scale-105 shadow-lg shadow-blue-500/20'
                : 'border-slate-600 bg-slate-800/30 hover:border-slate-500 hover:bg-slate-800/50'
            }`}
          >
            <div className="text-4xl mb-3">{goal.icon}</div>
            <h4 className="text-xl font-bold text-gray-100 mb-2 flex items-center gap-2">
              {goal.title}
              <InteractiveTooltip term={goal.tooltipTerm} mode="click" position="center" />
            </h4>
            <p className="text-gray-300 text-sm mb-3">{goal.description}</p>
            <p className="text-gray-400 text-xs italic">{goal.examples}</p>
          </button>
        ))}
      </div>
    </div>
  )
}

// Step 2: Factor Configuration
const FactorConfiguration = ({ nFactors, factorNames, factorLevels, selectedInteractions, goal, onFactorCountChange, onFactorNamesChange, onFactorLevelsChange, onInteractionsChange }) => {
  const handleFactorNameChange = (index, name) => {
    const newNames = [...factorNames]
    newNames[index] = name
    onFactorNamesChange(newNames)
  }

  const handleFactorLevelChange = (index, field, value) => {
    const newLevels = [...factorLevels]
    if (!newLevels[index]) {
      newLevels[index] = { min: '', max: '', units: '' }
    }
    newLevels[index] = { ...newLevels[index], [field]: value }
    onFactorLevelsChange(newLevels)
  }

  return (
    <div>
      <h3 className="text-2xl font-bold text-gray-100 mb-2">Configure Your Factors</h3>
      <p className="text-gray-300 text-sm mb-6">
        How many factors (variables) do you want to study?
      </p>

      {/* Factor Count Selector */}
      <div className="mb-8">
        <label className="block text-gray-200 font-semibold mb-3">Number of Factors</label>
        <div className="flex gap-3">
          {[2, 3, 4, 5, 6].map((n) => (
            <button
              key={n}
              onClick={() => onFactorCountChange(n)}
              className={`w-16 h-16 rounded-xl font-bold text-xl transition-all duration-200 ${
                nFactors === n
                  ? 'bg-blue-600 text-white scale-110 shadow-lg shadow-blue-500/50'
                  : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
              }`}
            >
              {n}
            </button>
          ))}
        </div>
        {nFactors >= 5 && (
          <div className="mt-3 bg-yellow-900/20 border border-yellow-700/50 rounded-lg p-3">
            <p className="text-yellow-300 text-sm">
              <strong>Note:</strong> With {nFactors} factors, we may recommend screening first to identify the most important factors.
            </p>
          </div>
        )}
      </div>

      {/* Factor Details */}
      <div>
        <label className="block text-gray-200 font-semibold mb-3">
          Factor Details <span className="text-gray-400 text-sm font-normal">(Specify ranges for real-world values)</span>
        </label>
        <div className="space-y-4">
          {Array.from({ length: nFactors }).map((_, i) => (
            <div key={i} className="bg-slate-800/50 rounded-lg p-4 border border-slate-600">
              <div className="flex items-center gap-2 mb-3">
                <span className="bg-blue-600 text-white w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm">
                  {i + 1}
                </span>
                <input
                  type="text"
                  placeholder={`Factor ${i + 1} (e.g., Temperature, Pressure)`}
                  value={factorNames[i] || ''}
                  onChange={(e) => handleFactorNameChange(i, e.target.value)}
                  className="flex-1 px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-gray-100 placeholder-gray-400 focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20"
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div>
                  <label className="block text-gray-400 text-xs mb-1">Low Level (-1)</label>
                  <input
                    type="number"
                    placeholder="Min value"
                    value={factorLevels[i]?.min || ''}
                    onChange={(e) => handleFactorLevelChange(i, 'min', e.target.value)}
                    className="w-full px-3 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-gray-100 placeholder-gray-400 focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20"
                  />
                </div>
                <div>
                  <label className="block text-gray-400 text-xs mb-1">High Level (+1)</label>
                  <input
                    type="number"
                    placeholder="Max value"
                    value={factorLevels[i]?.max || ''}
                    onChange={(e) => handleFactorLevelChange(i, 'max', e.target.value)}
                    className="w-full px-3 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-gray-100 placeholder-gray-400 focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20"
                  />
                </div>
                <div>
                  <label className="block text-gray-400 text-xs mb-1">Units (Optional)</label>
                  <input
                    type="text"
                    placeholder="e.g., °C, PSI"
                    value={factorLevels[i]?.units || ''}
                    onChange={(e) => handleFactorLevelChange(i, 'units', e.target.value)}
                    className="w-full px-3 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-gray-100 placeholder-gray-400 focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20"
                  />
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-4 bg-blue-900/20 border border-blue-700/50 rounded-lg p-4">
          <p className="text-blue-200 text-sm">
            <strong>Tip:</strong> Specifying factor levels allows the wizard to generate designs with real-world values.
            Leave blank to use standard coded values (-1, 0, +1).
          </p>
        </div>
      </div>

      {/* Factor Interaction Selector */}
      {nFactors >= 2 && (
        <div className="mt-8 pt-8 border-t border-slate-600">
          <FactorInteractionSelector
            nFactors={nFactors}
            factorNames={factorNames}
            goal={goal}
            selectedInteractions={selectedInteractions}
            onInteractionsChange={onInteractionsChange}
          />
        </div>
      )}
    </div>
  )
}

// Step 3: Power Analysis
const PowerAnalysis = ({ nFactors, powerAnalysis, onPowerAnalysisChange }) => {
  const effectSizes = [
    {
      id: 'small',
      label: 'Small',
      cohen_f: 0.10,
      description: 'Detecting subtle differences (hard to see without careful measurement)',
      example: '2-5% improvement in yield'
    },
    {
      id: 'medium',
      label: 'Medium',
      cohen_f: 0.25,
      description: 'Detecting moderate differences (noticeable with measurement)',
      example: '10-15% improvement in yield'
    },
    {
      id: 'large',
      label: 'Large',
      cohen_f: 0.40,
      description: 'Detecting substantial differences (obvious even without statistics)',
      example: '25%+ improvement in yield'
    }
  ]

  const powerLevels = [
    { value: 0.80, label: '80%', description: 'Standard (1 in 5 chance of missing real effect)' },
    { value: 0.90, label: '90%', description: 'Conservative (1 in 10 chance of missing)' },
    { value: 0.95, label: '95%', description: 'Very Conservative (1 in 20 chance of missing)' }
  ]

  // Calculate minimum runs based on power analysis
  // Practical approximation for Response Surface Methodology designs
  const calculateMinimumRuns = (effectSize, power, nFactors) => {
    // Effect size determines the multiplier for baseline design
    // small effects need more replication than large effects
    const effectMultiplier = {
      small: { base: 2.5, powerAdj: { 0.80: 1.0, 0.90: 1.3, 0.95: 1.6 } },
      medium: { base: 1.5, powerAdj: { 0.80: 1.0, 0.90: 1.2, 0.95: 1.4 } },
      large: { base: 1.0, powerAdj: { 0.80: 1.0, 0.90: 1.1, 0.95: 1.2 } }
    }

    const multiplier = effectMultiplier[effectSize]
    const powerAdj = multiplier.powerAdj[power]

    // Base runs for standard RSM designs (CCD or Box-Behnken)
    // These are the minimum unreplicated designs
    let baseRuns
    if (nFactors === 2) {
      baseRuns = 13 // 2^2 factorial (4) + 4 axial + 5 center points
    } else if (nFactors === 3) {
      baseRuns = 20 // 2^3 factorial (8) + 6 axial + 6 center points (CCD)
    } else if (nFactors === 4) {
      baseRuns = 31 // 2^4 factorial (16) + 8 axial + 7 center points
    } else if (nFactors === 5) {
      baseRuns = 52 // 2^5 fractional + 10 axial + ~10 center
    } else {
      baseRuns = 90 // For 6 factors, typically use fractional factorial + RSM
    }

    // Apply effect size and power adjustments
    const recommendedRuns = Math.ceil(baseRuns * multiplier.base * powerAdj)

    return recommendedRuns
  }

  const handleEffectSizeChange = (effectSize) => {
    const minimumRuns = calculateMinimumRuns(effectSize, powerAnalysis.desiredPower, nFactors)
    onPowerAnalysisChange({
      ...powerAnalysis,
      effectSize,
      minimumRuns
    })
  }

  const handlePowerChange = (power) => {
    const minimumRuns = calculateMinimumRuns(powerAnalysis.effectSize, power, nFactors)
    onPowerAnalysisChange({
      ...powerAnalysis,
      desiredPower: power,
      minimumRuns
    })
  }

  // Calculate on mount if not already calculated
  useEffect(() => {
    if (powerAnalysis.minimumRuns === null) {
      const minimumRuns = calculateMinimumRuns(powerAnalysis.effectSize, powerAnalysis.desiredPower, nFactors)
      onPowerAnalysisChange({
        ...powerAnalysis,
        minimumRuns
      })
    }
  }, []) // Run only on mount

  return (
    <div>
      <h3 className="text-2xl font-bold text-gray-100 mb-2 flex items-center gap-2">
        Power Analysis
        <InteractiveTooltip term="statistical-power" mode="both" position="bottom" />
      </h3>
      <p className="text-gray-300 text-sm mb-6">
        Ensure your experiment has enough runs to detect meaningful effects
      </p>

      {/* Educational Box */}
      <div className="bg-blue-900/20 border border-blue-700/50 rounded-lg p-4 mb-6">
        <h4 className="text-blue-200 font-semibold mb-2 flex items-center gap-2">
          What is <InlineTooltip term="statistical-power">Statistical Power</InlineTooltip>?
        </h4>
        <p className="text-blue-100 text-sm mb-2">
          Statistical power is the probability that your experiment will detect a real effect when it exists.
          Higher power means you're less likely to miss important discoveries (<InlineTooltip term="type-ii-error">Type II errors</InlineTooltip>)!
        </p>
        <p className="text-blue-100 text-sm">
          <strong>Rule of thumb:</strong> 80% power means if a real effect exists, you have an 80% chance of detecting it.
        </p>
      </div>

      {/* Effect Size Selection */}
      <div className="mb-6">
        <label className="block text-gray-200 font-semibold mb-3 flex items-center gap-2">
          What <InlineTooltip term="effect-size">effect size</InlineTooltip> do you want to detect?
        </label>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {effectSizes.map((effect) => (
            <button
              key={effect.id}
              onClick={() => handleEffectSizeChange(effect.id)}
              className={`p-4 rounded-xl border-2 transition-all duration-200 text-left ${
                powerAnalysis.effectSize === effect.id
                  ? 'border-blue-500 bg-blue-900/30 scale-105 shadow-lg shadow-blue-500/20'
                  : 'border-slate-600 bg-slate-800/30 hover:border-slate-500 hover:bg-slate-800/50'
              }`}
            >
              <h4 className="text-lg font-bold text-gray-100 mb-2">{effect.label}</h4>
              <p className="text-gray-300 text-sm mb-2">{effect.description}</p>
              <p className="text-gray-400 text-xs italic">e.g., {effect.example}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Power Level Selection */}
      <div className="mb-6">
        <label className="block text-gray-200 font-semibold mb-3">
          Desired Statistical Power
        </label>
        <div className="grid grid-cols-3 gap-3">
          {powerLevels.map((level) => (
            <button
              key={level.value}
              onClick={() => handlePowerChange(level.value)}
              className={`p-4 rounded-lg transition-all ${
                powerAnalysis.desiredPower === level.value
                  ? 'bg-blue-600 text-white scale-105 shadow-lg'
                  : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
              }`}
            >
              <div className="text-2xl font-bold mb-1">{level.label}</div>
              <div className="text-xs">{level.description}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Results Display */}
      {powerAnalysis.minimumRuns && (
        <div className="bg-gradient-to-r from-green-900/30 to-blue-900/30 border border-green-700/50 rounded-lg p-6">
          <div className="flex items-start gap-4">
            <div className="bg-green-600 rounded-full p-3">
              <CheckCircle className="w-8 h-8 text-white" />
            </div>
            <div className="flex-1">
              <h4 className="text-xl font-bold text-gray-100 mb-2">Recommended Minimum Runs</h4>
              <div className="flex items-baseline gap-2 mb-3">
                <span className="text-5xl font-bold text-green-300">{powerAnalysis.minimumRuns}</span>
                <span className="text-gray-300">experimental runs</span>
              </div>
              <p className="text-gray-300 text-sm mb-2">
                Based on detecting <strong>{powerAnalysis.effectSize}</strong> effects with{' '}
                <strong>{(powerAnalysis.desiredPower * 100).toFixed(0)}%</strong> power in a{' '}
                <strong>{nFactors}-factor</strong> experiment.
              </p>
              <div className="bg-slate-800/50 rounded-lg p-3 mt-3">
                <p className="text-yellow-200 text-sm">
                  <strong>⚠️ Note:</strong> This is a minimum recommendation. More runs will increase your power and precision.
                  The wizard will help you select an appropriate design that meets or exceeds this requirement.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Skip Option */}
      <div className="mt-6 bg-slate-800/30 border border-slate-600 rounded-lg p-4">
        <p className="text-gray-300 text-sm">
          <strong>Can I skip this?</strong> Yes, power analysis is optional but recommended. It helps ensure your experiment
          isn't underpowered (wasting resources) or overpowered (using unnecessary runs). Click "Next" to continue.
        </p>
      </div>
    </div>
  )
}

// Step 4: Constraint Builder
const ConstraintBuilder = ({ budget, timeConstraint, onBudgetChange, onTimeConstraintChange }) => {
  return (
    <div>
      <h3 className="text-2xl font-bold text-gray-100 mb-2">Budget & Constraints</h3>
      <p className="text-gray-300 text-sm mb-6">
        Set any limits on resources or time (all optional)
      </p>

      <div className="space-y-6">
        {/* Budget */}
        <div>
          <label className="block text-gray-200 font-semibold mb-3">
            Maximum Experimental Runs <span className="text-gray-400 text-sm font-normal">(Optional)</span>
          </label>
          <input
            type="number"
            min="5"
            max="100"
            placeholder="e.g., 20"
            value={budget || ''}
            onChange={(e) => onBudgetChange(e.target.value ? parseInt(e.target.value) : null)}
            className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600 rounded-lg text-gray-100 placeholder-gray-400 focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20"
          />
          <p className="mt-2 text-gray-400 text-sm">
            If you have a budget limit, we'll recommend designs that fit within it.
          </p>
        </div>

        {/* Time Constraint */}
        <div>
          <label className="block text-gray-200 font-semibold mb-3">
            Time Available <span className="text-gray-400 text-sm font-normal">(Optional)</span>
          </label>
          <div className="grid grid-cols-3 gap-3">
            {['Low (Days)', 'Medium (Weeks)', 'High (Months)'].map((option) => (
              <button
                key={option}
                onClick={() => onTimeConstraintChange(option.split(' ')[0].toLowerCase())}
                className={`px-4 py-3 rounded-lg font-medium transition-all ${
                  timeConstraint === option.split(' ')[0].toLowerCase()
                    ? 'bg-blue-600 text-white'
                    : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
                }`}
              >
                {option}
              </button>
            ))}
          </div>
        </div>

        <div className="bg-blue-900/20 border border-blue-700/50 rounded-lg p-4">
          <p className="text-blue-200 text-sm">
            <strong>Tip:</strong> Constraints help us recommend the most practical design for your situation. If unsure, you can skip this step.
          </p>
        </div>
      </div>
    </div>
  )
}

// Step 5: Design Summary
const DesignSummary = ({ wizardData }) => {
  const { goal, nFactors, factorNames, budget, timeConstraint, selectedDesign } = wizardData

  return (
    <div>
      <h3 className="text-2xl font-bold text-gray-100 mb-2">Review Your Experiment</h3>
      <p className="text-gray-300 text-sm mb-6">
        Confirm your experiment design before generation
      </p>

      <div className="space-y-4">
        {/* Goal */}
        <div className="bg-slate-800/50 rounded-lg p-4">
          <p className="text-gray-400 text-sm mb-1">Experiment Goal</p>
          <p className="text-gray-100 font-semibold text-lg capitalize">{goal || 'Not specified'}</p>
        </div>

        {/* Factors */}
        <div className="bg-slate-800/50 rounded-lg p-4">
          <p className="text-gray-400 text-sm mb-1">Number of Factors</p>
          <p className="text-gray-100 font-semibold text-lg">{nFactors} factors</p>
          {factorNames && factorNames.filter(n => n).length > 0 && (
            <div className="mt-2 flex flex-wrap gap-2">
              {factorNames.filter(n => n).map((name, i) => (
                <span key={i} className="px-2 py-1 bg-blue-900/30 border border-blue-700/50 rounded text-blue-200 text-sm">
                  {name}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Constraints */}
        {(budget || timeConstraint) && (
          <div className="bg-slate-800/50 rounded-lg p-4">
            <p className="text-gray-400 text-sm mb-2">Constraints</p>
            <div className="space-y-1">
              {budget && (
                <p className="text-gray-100">
                  <span className="text-gray-400">Max Runs:</span> <span className="font-semibold">{budget}</span>
                </p>
              )}
              {timeConstraint && (
                <p className="text-gray-100">
                  <span className="text-gray-400">Time Available:</span> <span className="font-semibold capitalize">{timeConstraint}</span>
                </p>
              )}
            </div>
          </div>
        )}

        {/* Selected Design */}
        {selectedDesign && (
          <div className="bg-gradient-to-r from-blue-900/30 to-purple-900/30 border border-blue-700/50 rounded-lg p-5">
            <p className="text-blue-300 text-sm font-semibold mb-2">Selected Design</p>
            <h4 className="text-2xl font-bold text-gray-100 mb-2">{selectedDesign.type}</h4>
            <p className="text-gray-300 mb-3">{selectedDesign.description}</p>
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-slate-800/50 rounded p-2">
                <p className="text-gray-400 text-xs">Experimental Runs</p>
                <p className="text-xl font-bold text-blue-300">{selectedDesign.runs}</p>
              </div>
              <div className="bg-slate-800/50 rounded p-2">
                <p className="text-gray-400 text-xs">Design Score</p>
                <p className="text-xl font-bold text-green-300">{selectedDesign.score}/100</p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Design Preview Visualization */}
      {selectedDesign && (
        <div className="mt-6">
          <DesignPreview wizardData={wizardData} />
        </div>
      )}

      {/* Smart Validation */}
      <div className="mt-6">
        <SmartValidation validations={validateWizardData(wizardData)} />
      </div>

      {/* Sequential Experimentation Guide */}
      <div className="mt-6">
        <SequentialExperimentGuide wizardData={wizardData} />
      </div>

      <div className="mt-6 bg-green-900/20 border border-green-700/50 rounded-lg p-4">
        <p className="text-green-200 text-sm">
          <strong>Ready to generate!</strong> Click "Generate Design" below to create your experimental design matrix.
        </p>
      </div>
    </div>
  )
}

// Design Results Component
const DesignResults = ({ design, wizardData }) => {
  const factorNames = design.factorNames || []
  const originalMatrix = design.design_matrix || []
  const factorLevels = design.factorLevels || []
  const useActualValues = design.useActualValues || false

  const [designMatrix, setDesignMatrix] = useState(originalMatrix)
  const [isRandomized, setIsRandomized] = useState(false)
  const [exportMenuOpen, setExportMenuOpen] = useState(false)

  // Fisher-Yates shuffle algorithm
  const shuffleArray = (array) => {
    const shuffled = [...array]
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]
    }
    return shuffled
  }

  const handleRandomize = () => {
    setDesignMatrix(shuffleArray(originalMatrix))
    setIsRandomized(true)
  }

  const handleRestore = () => {
    setDesignMatrix(originalMatrix)
    setIsRandomized(false)
  }

  // Create current design object with potentially randomized matrix
  const getCurrentDesign = () => ({
    ...design,
    design_matrix: designMatrix, // Use current (possibly randomized) matrix
    isRandomized
  })

  const exportOptions = [
    {
      label: 'PDF Report (Recommended)',
      description: 'Complete report with instructions',
      icon: <FileText className="w-5 h-5" />,
      color: 'text-red-400',
      handler: () => downloadPDF(getCurrentDesign(), wizardData)
    },
    {
      label: 'Excel Workbook',
      description: 'Multi-sheet with summary & instructions',
      icon: <FileSpreadsheet className="w-5 h-5" />,
      color: 'text-green-400',
      handler: () => downloadExcel(getCurrentDesign(), wizardData)
    },
    {
      label: 'CSV (Enhanced)',
      description: 'With metadata and notes column',
      icon: <Download className="w-5 h-5" />,
      color: 'text-blue-400',
      handler: () => downloadCSV(getCurrentDesign(), wizardData, true)
    },
    {
      label: 'JMP Format',
      description: 'Compatible with JMP software',
      icon: <FileCode className="w-5 h-5" />,
      color: 'text-purple-400',
      handler: () => downloadJMP(getCurrentDesign(), wizardData)
    },
    {
      label: 'Minitab Format',
      description: 'Compatible with Minitab software',
      icon: <FileCode className="w-5 h-5" />,
      color: 'text-orange-400',
      handler: () => downloadMinitab(getCurrentDesign(), wizardData)
    }
  ]

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-3xl font-bold text-gray-100 mb-2">Your Experiment Design</h3>
          <p className="text-gray-300">
            Generated {designMatrix.length} experimental runs for {factorNames.length} factors
          </p>
        </div>
        <div className="bg-green-900/30 border border-green-700/50 rounded-lg px-4 py-2">
          <CheckCircle className="w-6 h-6 text-green-400 inline mr-2" />
          <span className="text-green-200 font-semibold">Design Generated!</span>
        </div>
      </div>

      {/* Design Info */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-slate-800/50 rounded-lg p-4">
          <p className="text-gray-400 text-sm mb-1">Design Type</p>
          <p className="text-gray-100 font-semibold text-lg">{wizardData.selectedDesign?.type}</p>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-4">
          <p className="text-gray-400 text-sm mb-1">Total Runs</p>
          <p className="text-gray-100 font-semibold text-lg">{designMatrix.length}</p>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-4">
          <p className="text-gray-400 text-sm mb-1">Factors</p>
          <p className="text-gray-100 font-semibold text-lg">{factorNames.length}</p>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-4">
          <p className="text-gray-400 text-sm mb-1">Value Type</p>
          <p className="text-gray-100 font-semibold text-lg">
            {useActualValues ? 'Actual Values' : 'Coded (-1, 0, +1)'}
          </p>
        </div>
      </div>

      {/* Factor Ranges (if using actual values) */}
      {useActualValues && factorLevels.some(l => l && l.min && l.max) && (
        <div className="bg-blue-900/20 border border-blue-700/50 rounded-lg p-4 mb-6">
          <h4 className="text-blue-200 font-semibold mb-3">Factor Ranges</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {factorNames.map((name, idx) => {
              const level = factorLevels[idx]
              if (!level || !level.min || !level.max) return null
              return (
                <div key={idx} className="flex items-center justify-between bg-slate-800/50 rounded px-3 py-2">
                  <span className="text-gray-300 font-medium">{name}:</span>
                  <span className="text-blue-200">
                    {level.min} to {level.max} {level.units || ''}
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Design Matrix Table */}
      <div className="bg-slate-800/50 rounded-lg p-6 overflow-x-auto">
        <div className="flex items-start justify-between mb-4">
          <div className="flex-1">
            <h4 className="text-xl font-bold text-gray-100 mb-2">Experimental Design Matrix</h4>
            <p className="text-gray-300 text-sm">
              Run these experiments in the order shown below. Record your response values for each run.
            </p>
            {isRandomized && (
              <div className="mt-2 flex items-center gap-2 text-green-300 text-sm">
                <CheckCircle className="w-4 h-4" />
                <span className="font-semibold">Run order has been randomized</span>
              </div>
            )}
          </div>
          <div className="flex gap-2 ml-4">
            {/* Export Dropdown Menu */}
            <div className="relative">
              <button
                onClick={() => setExportMenuOpen(!exportMenuOpen)}
                className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white rounded-lg font-semibold transition-all hover:scale-105 shadow-lg"
                title="Export design in various formats"
              >
                <Download className="w-5 h-5" />
                Export Design
                <ChevronDown className={`w-4 h-4 transition-transform ${exportMenuOpen ? 'rotate-180' : ''}`} />
              </button>

              {exportMenuOpen && (
                <>
                  {/* Backdrop to close dropdown */}
                  <div
                    className="fixed inset-0 z-10"
                    onClick={() => setExportMenuOpen(false)}
                  />
                  {/* Dropdown Menu */}
                  <div className="absolute right-0 mt-2 w-80 bg-slate-800 border border-slate-700 rounded-lg shadow-2xl z-20 overflow-hidden">
                    <div className="bg-gradient-to-r from-blue-900/50 to-indigo-900/50 px-4 py-3 border-b border-slate-700">
                      <h3 className="text-white font-semibold text-sm">Export Options</h3>
                      <p className="text-gray-300 text-xs mt-0.5">
                        {isRandomized ? '✓ Randomized order included' : 'Design order (not randomized)'}
                      </p>
                    </div>
                    <div className="max-h-96 overflow-y-auto">
                      {exportOptions.map((option, idx) => (
                        <button
                          key={idx}
                          onClick={() => {
                            option.handler()
                            setExportMenuOpen(false)
                          }}
                          className="w-full px-4 py-3 text-left hover:bg-slate-700/50 transition-colors border-b border-slate-700/50 last:border-0"
                        >
                          <div className="flex items-start gap-3">
                            <div className={`${option.color} flex-shrink-0 mt-0.5`}>
                              {option.icon}
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="text-gray-100 font-semibold text-sm">
                                {option.label}
                              </div>
                              <div className="text-gray-400 text-xs mt-0.5">
                                {option.description}
                              </div>
                            </div>
                          </div>
                        </button>
                      ))}
                    </div>
                    <div className="bg-slate-900/50 px-4 py-2 border-t border-slate-700">
                      <p className="text-xs text-gray-400 italic">
                        💡 Tip: PDF Report recommended for complete documentation
                      </p>
                    </div>
                  </div>
                </>
              )}
            </div>

            <button
              onClick={handleRandomize}
              className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-semibold transition-all hover:scale-105 shadow-lg"
              title="Randomize the order of experimental runs to prevent systematic bias"
            >
              <Shuffle className="w-5 h-5" />
              Randomize
            </button>
            {isRandomized && (
              <button
                onClick={handleRestore}
                className="flex items-center gap-2 px-4 py-2 bg-slate-600 hover:bg-slate-700 text-white rounded-lg font-semibold transition-all hover:scale-105"
                title="Restore original run order"
              >
                <RotateCcw className="w-5 h-5" />
                Restore
              </button>
            )}
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-600">
                <th className="text-left py-3 px-4 text-gray-300 font-semibold">Run</th>
                {factorNames.map((name, idx) => {
                  const level = factorLevels[idx]
                  const units = level && level.units ? ` (${level.units})` : ''
                  return (
                    <th key={idx} className="text-left py-3 px-4 text-gray-300 font-semibold">
                      {name}{units}
                    </th>
                  )
                })}
                <th className="text-left py-3 px-4 text-gray-300 font-semibold">Response (Y)</th>
              </tr>
            </thead>
            <tbody>
              {designMatrix.map((row, runIdx) => (
                <tr key={runIdx} className="border-b border-slate-700/50 hover:bg-slate-700/30">
                  <td className="py-3 px-4 text-gray-200 font-medium">{runIdx + 1}</td>
                  {factorNames.map((name, factorIdx) => (
                    <td key={factorIdx} className="py-3 px-4 text-gray-100">
                      {typeof row[name] === 'number' ? row[name].toFixed(3) : row[name]}
                    </td>
                  ))}
                  <td className="py-3 px-4 text-gray-400 italic">Record here</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Randomization Importance */}
      {!isRandomized && (
        <div className="mt-6 bg-yellow-900/20 border border-yellow-700/50 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-yellow-300 font-semibold mb-1">Randomization Recommended</p>
              <p className="text-yellow-200 text-sm">
                For valid experimental results, randomize the run order to prevent systematic bias from uncontrolled variables
                (time trends, equipment drift, environmental changes). Click "Randomize Order" above before running your experiments.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Next Steps */}
      <div className="mt-6 bg-blue-900/20 border border-blue-700/50 rounded-lg p-6">
        <h4 className="text-lg font-bold text-blue-200 mb-3">Next Steps</h4>
        <ol className="space-y-2 text-blue-100">
          <li className="flex items-start gap-2">
            <span className="bg-blue-700/50 rounded-full w-6 h-6 flex items-center justify-center flex-shrink-0 mt-0.5 text-sm">1</span>
            <span><strong>Randomize the run order</strong> using the "Randomize" button above (critical for valid results)</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="bg-blue-700/50 rounded-full w-6 h-6 flex items-center justify-center flex-shrink-0 mt-0.5 text-sm">2</span>
            <span>Download the {isRandomized ? 'randomized' : ''} design matrix as CSV using the "Download CSV" button above</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="bg-blue-700/50 rounded-full w-6 h-6 flex items-center justify-center flex-shrink-0 mt-0.5 text-sm">3</span>
            <span>Run your experiments in the {isRandomized ? 'randomized' : 'specified'} order and record response values</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="bg-blue-700/50 rounded-full w-6 h-6 flex items-center justify-center flex-shrink-0 mt-0.5 text-sm">4</span>
            <span>Upload your completed data to the Response Surface page for analysis</span>
          </li>
        </ol>
      </div>
    </div>
  )
}

// Helper function to generate CSV
const generateCSV = (design) => {
  const factorNames = design.factorNames || []
  const designMatrix = design.design_matrix || []
  const factorLevels = design.factorLevels || []

  // Header row with units if available
  const headers = factorNames.map((name, idx) => {
    const level = factorLevels[idx]
    const units = level && level.units ? ` (${level.units})` : ''
    return `${name}${units}`
  })
  let csv = 'Run,' + headers.join(',') + ',Response\n'

  // Data rows
  designMatrix.forEach((row, idx) => {
    const values = factorNames.map(name => {
      const value = row[name]
      return typeof value === 'number' ? value.toFixed(3) : value
    })
    csv += `${idx + 1},${values.join(',')},\n`
  })

  return csv
}

export default ExperimentWizardPage
