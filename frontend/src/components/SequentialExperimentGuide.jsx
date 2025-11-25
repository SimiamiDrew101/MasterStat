import { useState, useEffect } from 'react'
import {
  GitBranch,
  ArrowRight,
  CheckCircle,
  Circle,
  AlertTriangle,
  Save,
  Clock,
  Lightbulb,
  Play,
  Target,
  TrendingUp,
  BookOpen,
  Info
} from 'lucide-react'
import {
  detectSequentialScenario,
  saveWizardState,
  loadWizardState,
  generateTimeline,
  PHASE_TYPE
} from '../utils/sequentialExperimentation'

const SequentialExperimentGuide = ({ wizardData }) => {
  const [guidance, setGuidance] = useState(null)
  const [timeline, setTimeline] = useState(null)
  const [savedState, setSavedState] = useState(null)
  const [showSaveConfirm, setShowSaveConfirm] = useState(false)

  useEffect(() => {
    const sequentialGuidance = detectSequentialScenario(wizardData)
    setGuidance(sequentialGuidance)

    if (sequentialGuidance) {
      const timelineData = generateTimeline(sequentialGuidance)
      setTimeline(timelineData)
    }

    // Check if there's a saved state
    const saved = loadWizardState()
    if (saved) {
      setSavedState(saved)
    }
  }, [wizardData])

  const handleSaveState = () => {
    const success = saveWizardState(wizardData, guidance?.currentPhaseGuidance?.title || 'Phase 1')
    if (success) {
      setShowSaveConfirm(true)
      setTimeout(() => setShowSaveConfirm(false), 3000)
    }
  }

  if (!guidance) {
    return null
  }

  // Don't show for simple single-phase experiments unless there's useful guidance
  if (!guidance.isSequential && guidance.scenario === 'no_guidance') {
    return null
  }

  const getPhaseIcon = (phaseType) => {
    switch (phaseType) {
      case PHASE_TYPE.SCREENING:
        return <Target className="w-5 h-5" />
      case PHASE_TYPE.OPTIMIZATION:
        return <TrendingUp className="w-5 h-5" />
      case PHASE_TYPE.CHARACTERIZATION:
        return <BookOpen className="w-5 h-5" />
      default:
        return <Circle className="w-5 h-5" />
    }
  }

  const getPhaseColor = (phaseType) => {
    switch (phaseType) {
      case PHASE_TYPE.SCREENING:
        return 'purple'
      case PHASE_TYPE.OPTIMIZATION:
        return 'green'
      case PHASE_TYPE.CHARACTERIZATION:
        return 'blue'
      default:
        return 'gray'
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="flex items-start gap-3">
          <GitBranch className="w-7 h-7 text-purple-400 mt-1" />
          <div>
            <h3 className="text-2xl font-bold text-gray-100">{guidance.title}</h3>
            <p className="text-gray-300 text-sm mt-1">{guidance.message}</p>
          </div>
        </div>

        {guidance.isSequential && (
          <button
            onClick={handleSaveState}
            className="flex items-center gap-2 px-4 py-2 bg-purple-700 hover:bg-purple-600 text-white rounded-lg text-sm font-medium transition-colors"
          >
            <Save className="w-4 h-4" />
            Save Progress
          </button>
        )}
      </div>

      {/* Save Confirmation */}
      {showSaveConfirm && (
        <div className="bg-green-900/30 border border-green-700/50 rounded-lg p-3 flex items-center gap-2">
          <CheckCircle className="w-5 h-5 text-green-400" />
          <p className="text-green-200 text-sm font-medium">
            Experiment state saved! You can return to continue Phase 2 later.
          </p>
        </div>
      )}

      {/* Warning Banner */}
      {guidance.warning && (
        <div className="bg-orange-900/20 border border-orange-700/50 rounded-lg p-4 flex items-start gap-3">
          <AlertTriangle className="w-6 h-6 text-orange-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-orange-200 font-semibold">Sequential Experimentation Recommended</p>
            <p className="text-orange-300 text-sm mt-1">
              Your current configuration will be more efficient with a two-phase approach.
              This saves experimental resources while providing better insights.
            </p>
          </div>
        </div>
      )}

      {/* Timeline Visualization */}
      {guidance.isSequential && timeline && (
        <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700/50">
          <div className="flex items-center gap-2 mb-4">
            <Clock className="w-5 h-5 text-blue-400" />
            <h4 className="text-lg font-bold text-gray-100">Experimental Timeline</h4>
          </div>

          <div className="flex items-center gap-4">
            {timeline.phases.map((phase, index) => (
              <div key={index} className="flex items-center gap-4 flex-1">
                <div
                  className={`flex-1 bg-gradient-to-br ${
                    index === 0
                      ? 'from-purple-900/30 to-purple-800/20 border-purple-700/50'
                      : 'from-green-900/30 to-green-800/20 border-green-700/50'
                  } border rounded-lg p-4`}
                >
                  <div className="flex items-center gap-2 mb-2">
                    {index === 0 ? (
                      <Target className={`w-5 h-5 text-purple-400`} />
                    ) : (
                      <TrendingUp className={`w-5 h-5 text-green-400`} />
                    )}
                    <p
                      className={`font-semibold ${
                        index === 0 ? 'text-purple-200' : 'text-green-200'
                      }`}
                    >
                      {phase.phase}
                    </p>
                  </div>
                  <p
                    className={`text-2xl font-bold mb-1 ${
                      index === 0 ? 'text-purple-100' : 'text-green-100'
                    }`}
                  >
                    {phase.runs} runs
                  </p>
                  <p className="text-gray-400 text-xs">{phase.description}</p>
                </div>

                {index < timeline.phases.length - 1 && (
                  <ArrowRight className="w-6 h-6 text-gray-500 flex-shrink-0" />
                )}
              </div>
            ))}
          </div>

          <div className="mt-4 pt-4 border-t border-slate-700/50">
            <p className="text-gray-300 text-sm">
              <strong>Total experimental budget:</strong> {timeline.totalRuns}
            </p>
          </div>
        </div>
      )}

      {/* Current Phase Guidance */}
      {guidance.currentPhaseGuidance && (
        <div className="bg-gradient-to-br from-purple-900/20 to-purple-800/10 border border-purple-700/50 rounded-lg p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="bg-purple-700/30 rounded-lg p-2">
              <Play className="w-6 h-6 text-purple-300" />
            </div>
            <div>
              <h4 className="text-xl font-bold text-purple-100">
                {guidance.currentPhaseGuidance.title}
              </h4>
              <p className="text-purple-300 text-sm">
                {guidance.currentPhaseGuidance.description}
              </p>
            </div>
          </div>

          {/* Design Recommendations */}
          {guidance.currentPhaseGuidance.designRecommendations && (
            <div className="mb-4">
              <p className="text-gray-300 text-sm font-semibold mb-2">Recommended Designs:</p>
              <div className="flex flex-wrap gap-2">
                {guidance.currentPhaseGuidance.designRecommendations.map((design, idx) => (
                  <span
                    key={idx}
                    className="bg-purple-900/30 border border-purple-700/30 text-purple-200 px-3 py-1 rounded-full text-sm"
                  >
                    {design}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Expected Runs */}
          <div className="mb-4 bg-slate-900/30 rounded-lg p-3">
            <p className="text-gray-300 text-sm">
              <strong>Expected runs:</strong>{' '}
              <span className="text-purple-200 font-bold">
                {guidance.currentPhaseGuidance.expectedRuns}
              </span>
            </p>
          </div>

          {/* Action Items */}
          {guidance.currentPhaseGuidance.actionItems && (
            <div>
              <div className="flex items-center gap-2 mb-3">
                <CheckCircle className="w-5 h-5 text-purple-400" />
                <p className="text-gray-200 font-semibold">Action Items:</p>
              </div>
              <div className="space-y-2">
                {guidance.currentPhaseGuidance.actionItems.map((item, idx) => (
                  <div key={idx} className="flex items-start gap-3">
                    <Circle className="w-4 h-4 text-purple-400 flex-shrink-0 mt-1" />
                    <p className="text-gray-300 text-sm">{item}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Tips */}
          {guidance.currentPhaseGuidance.tips && guidance.currentPhaseGuidance.tips.length > 0 && (
            <div className="mt-4 pt-4 border-t border-purple-700/30">
              <div className="flex items-center gap-2 mb-3">
                <Lightbulb className="w-5 h-5 text-yellow-400" />
                <p className="text-gray-200 font-semibold">Tips:</p>
              </div>
              <div className="space-y-2">
                {guidance.currentPhaseGuidance.tips.map((tip, idx) => (
                  <div key={idx} className="flex items-start gap-3">
                    <div className="w-1.5 h-1.5 rounded-full bg-yellow-400 flex-shrink-0 mt-2" />
                    <p className="text-gray-400 text-sm italic">{tip}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Next Phase Guidance */}
      {guidance.isSequential && guidance.nextPhaseGuidance && (
        <div className="bg-gradient-to-br from-green-900/20 to-green-800/10 border border-green-700/50 rounded-lg p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="bg-green-700/30 rounded-lg p-2">
              <ArrowRight className="w-6 h-6 text-green-300" />
            </div>
            <div>
              <h4 className="text-xl font-bold text-green-100">
                {guidance.nextPhaseGuidance.title}
              </h4>
              <p className="text-green-300 text-sm">{guidance.nextPhaseGuidance.description}</p>
            </div>
          </div>

          {/* Design Recommendations */}
          {guidance.nextPhaseGuidance.designRecommendations && (
            <div className="mb-4">
              <p className="text-gray-300 text-sm font-semibold mb-2">Recommended Designs:</p>
              <div className="flex flex-wrap gap-2">
                {guidance.nextPhaseGuidance.designRecommendations.map((design, idx) => (
                  <span
                    key={idx}
                    className="bg-green-900/30 border border-green-700/30 text-green-200 px-3 py-1 rounded-full text-sm"
                  >
                    {design}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Expected Runs */}
          <div className="mb-4 bg-slate-900/30 rounded-lg p-3">
            <p className="text-gray-300 text-sm">
              <strong>Expected runs:</strong>{' '}
              <span className="text-green-200 font-bold">
                {guidance.nextPhaseGuidance.expectedRuns}
              </span>
            </p>
          </div>

          {/* Action Items */}
          {guidance.nextPhaseGuidance.actionItems && (
            <div>
              <div className="flex items-center gap-2 mb-3">
                <CheckCircle className="w-5 h-5 text-green-400" />
                <p className="text-gray-200 font-semibold">What to do next:</p>
              </div>
              <div className="space-y-2">
                {guidance.nextPhaseGuidance.actionItems.map((item, idx) => (
                  <div key={idx} className="flex items-start gap-3">
                    <div
                      className={`flex-shrink-0 mt-0.5 w-6 h-6 rounded-full ${
                        idx === 0 ? 'bg-green-700/30' : 'bg-slate-700/30'
                      } flex items-center justify-center`}
                    >
                      <span
                        className={`text-xs font-bold ${
                          idx === 0 ? 'text-green-300' : 'text-gray-400'
                        }`}
                      >
                        {idx + 1}
                      </span>
                    </div>
                    <p className="text-gray-300 text-sm">{item}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Important Note */}
          <div className="mt-4 pt-4 border-t border-green-700/30 bg-green-900/10 rounded-lg p-3">
            <div className="flex items-start gap-2">
              <Info className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-green-200 font-semibold text-sm">Important:</p>
                <p className="text-green-300 text-sm mt-1">
                  After running and analyzing Phase 1, return to this Experiment Wizard to design
                  Phase 2. Your experimental context will be saved for continuity.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Single-Phase Success Message */}
      {!guidance.isSequential && guidance.scenario === 'single_phase_optimal' && (
        <div className="bg-green-900/20 border border-green-700/50 rounded-lg p-4 flex items-start gap-3">
          <CheckCircle className="w-6 h-6 text-green-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-green-200 font-semibold">Optimal Single-Phase Design</p>
            <p className="text-green-300 text-sm mt-1">
              Your experiment is well-suited for a single-phase approach. No sequential
              experimentation needed!
            </p>
          </div>
        </div>
      )}

      {/* Educational Note */}
      {guidance.isSequential && (
        <div className="bg-blue-900/20 border border-blue-700/50 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <BookOpen className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
            <div className="text-sm text-blue-200">
              <p className="font-semibold mb-2">Why Sequential Experimentation?</p>
              <ul className="space-y-1 list-disc list-inside text-blue-300">
                <li>
                  <strong>Efficiency:</strong> Screen many factors cheaply, then optimize only the
                  important ones
                </li>
                <li>
                  <strong>Learning:</strong> Phase 1 informs Phase 2 design, reducing uncertainty
                </li>
                <li>
                  <strong>Resource savings:</strong> Avoid wasting runs on unimportant factors
                </li>
                <li>
                  <strong>Better results:</strong> Focused optimization yields more precise answers
                </li>
                <li>
                  <strong>Risk management:</strong> Confirm direction before committing to full RSM
                </li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default SequentialExperimentGuide
