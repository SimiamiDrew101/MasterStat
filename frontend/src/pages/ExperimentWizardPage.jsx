import { useState } from 'react'
import { ChevronRight, ChevronLeft, CheckCircle, Sparkles } from 'lucide-react'
import DesignRecommendationStep from '../components/DesignRecommendationStep'

const ExperimentWizardPage = () => {
  const [currentStep, setCurrentStep] = useState(1)
  const [wizardData, setWizardData] = useState({
    goal: '',
    nFactors: 2,
    factorNames: [],
    budget: null,
    timeConstraint: null,
    selectedDesign: null
  })

  const steps = [
    { id: 1, title: 'Goal', icon: '🎯' },
    { id: 2, title: 'Factors', icon: '🔬' },
    { id: 3, title: 'Constraints', icon: '📊' },
    { id: 4, title: 'Design', icon: '✨' },
    { id: 5, title: 'Review', icon: '✅' }
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

  const handleComplete = () => {
    console.log('Experiment design completed:', wizardData)
    // Here you can add logic to save the design or navigate to another page
    alert('Experiment design generated successfully! Check console for details.')
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
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl px-8 py-6 mb-8 border border-slate-700/50">
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
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl p-8 mb-8 border border-slate-700/50 min-h-[500px]">
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
              onFactorCountChange={(n) => updateWizardData('nFactors', n)}
              onFactorNamesChange={(names) => updateWizardData('factorNames', names)}
            />
          )}
          {currentStep === 3 && (
            <ConstraintBuilder
              budget={wizardData.budget}
              timeConstraint={wizardData.timeConstraint}
              onBudgetChange={(budget) => updateWizardData('budget', budget)}
              onTimeConstraintChange={(time) => updateWizardData('timeConstraint', time)}
            />
          )}
          {currentStep === 4 && (
            <DesignRecommendationStep
              nFactors={wizardData.nFactors}
              budget={wizardData.budget}
              goal={wizardData.goal}
              selectedDesign={wizardData.selectedDesign}
              onSelectDesign={(design) => updateWizardData('selectedDesign', design)}
            />
          )}
          {currentStep === 5 && (
            <DesignSummary wizardData={wizardData} />
          )}
        </div>

        {/* Navigation Footer */}
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl px-8 py-6 border border-slate-700/50 flex items-center justify-between">
          <button
            onClick={prevStep}
            disabled={currentStep === 1}
            className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all ${
              currentStep === 1
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

          {currentStep < steps.length ? (
            <button
              onClick={nextStep}
              disabled={
                (currentStep === 1 && !wizardData.goal) ||
                (currentStep === 2 && wizardData.nFactors < 2) ||
                (currentStep === 4 && !wizardData.selectedDesign)
              }
              className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all ${
                (currentStep === 1 && !wizardData.goal) ||
                (currentStep === 2 && wizardData.nFactors < 2) ||
                (currentStep === 4 && !wizardData.selectedDesign)
                  ? 'bg-slate-700/30 text-gray-500 cursor-not-allowed'
                  : 'bg-blue-600 text-white hover:bg-blue-700 hover:scale-105'
              }`}
            >
              Next
              <ChevronRight className="w-5 h-5" />
            </button>
          ) : (
            <button
              onClick={handleComplete}
              className="flex items-center gap-2 px-6 py-3 rounded-lg font-semibold bg-green-600 text-white hover:bg-green-700 hover:scale-105 transition-all"
            >
              <CheckCircle className="w-5 h-5" />
              Generate Design
            </button>
          )}
        </div>
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
      icon: '🎯',
      description: 'Find the best factor settings to maximize or minimize your response',
      examples: 'Maximize yield, minimize cost, optimize quality'
    },
    {
      id: 'screening',
      title: 'Screening',
      icon: '🔍',
      description: 'Identify which factors have the biggest impact on your response',
      examples: 'Which of 5+ factors matter most?'
    },
    {
      id: 'modeling',
      title: 'Response Surface Modeling',
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
            <h4 className="text-xl font-bold text-gray-100 mb-2">{goal.title}</h4>
            <p className="text-gray-300 text-sm mb-3">{goal.description}</p>
            <p className="text-gray-400 text-xs italic">{goal.examples}</p>
          </button>
        ))}
      </div>
    </div>
  )
}

// Step 2: Factor Configuration
const FactorConfiguration = ({ nFactors, factorNames, onFactorCountChange, onFactorNamesChange }) => {
  const handleFactorNameChange = (index, name) => {
    const newNames = [...factorNames]
    newNames[index] = name
    onFactorNamesChange(newNames)
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

      {/* Factor Names (Optional) */}
      <div>
        <label className="block text-gray-200 font-semibold mb-3">
          Factor Names <span className="text-gray-400 text-sm font-normal">(Optional)</span>
        </label>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {Array.from({ length: nFactors }).map((_, i) => (
            <input
              key={i}
              type="text"
              placeholder={`Factor ${i + 1} (e.g., Temperature, Pressure)`}
              value={factorNames[i] || ''}
              onChange={(e) => handleFactorNameChange(i, e.target.value)}
              className="px-4 py-3 bg-slate-700/50 border border-slate-600 rounded-lg text-gray-100 placeholder-gray-400 focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20"
            />
          ))}
        </div>
      </div>
    </div>
  )
}

// Step 3: Constraint Builder
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

      <div className="mt-6 bg-green-900/20 border border-green-700/50 rounded-lg p-4">
        <p className="text-green-200 text-sm">
          <strong>Ready to generate!</strong> Click "Generate Design" below to create your experimental design matrix.
        </p>
      </div>
    </div>
  )
}

export default ExperimentWizardPage
