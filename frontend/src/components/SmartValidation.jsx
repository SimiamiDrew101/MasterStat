import { useState } from 'react'
import {
  AlertCircle,
  AlertTriangle,
  Info,
  CheckCircle,
  XCircle,
  ChevronDown,
  ChevronUp,
  Lightbulb,
  ShieldAlert,
  Target,
  Zap,
  Award,
  Filter,
  Network,
  TrendingUp,
  DollarSign
} from 'lucide-react'
import { SEVERITY } from '../utils/smartValidation'

const SmartValidation = ({ validations, compact = false }) => {
  const [expandedIds, setExpandedIds] = useState(new Set())

  if (!validations || validations.length === 0) {
    return null
  }

  // Group validations by severity
  const errors = validations.filter(v => v.severity === SEVERITY.ERROR)
  const warnings = validations.filter(v => v.severity === SEVERITY.WARNING)
  const info = validations.filter(v => v.severity === SEVERITY.INFO)
  const success = validations.filter(v => v.severity === SEVERITY.SUCCESS)

  const toggleExpanded = (id) => {
    setExpandedIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
  }

  const getIcon = (iconName) => {
    const icons = {
      AlertCircle,
      AlertTriangle,
      Info,
      CheckCircle,
      XCircle,
      Target,
      Zap,
      Award,
      Filter,
      Network,
      TrendingUp,
      DollarSign,
      ShieldAlert
    }
    const IconComponent = icons[iconName] || AlertCircle
    return <IconComponent className="w-5 h-5" />
  }

  const getSeverityConfig = (severity) => {
    const configs = {
      [SEVERITY.ERROR]: {
        bgColor: 'bg-red-900/20',
        borderColor: 'border-red-700/50',
        textColor: 'text-red-200',
        iconColor: 'text-red-400',
        badgeColor: 'bg-red-900/50',
        icon: <XCircle className="w-5 h-5" />
      },
      [SEVERITY.WARNING]: {
        bgColor: 'bg-yellow-900/20',
        borderColor: 'border-yellow-700/50',
        textColor: 'text-yellow-200',
        iconColor: 'text-yellow-400',
        badgeColor: 'bg-yellow-900/50',
        icon: <AlertTriangle className="w-5 h-5" />
      },
      [SEVERITY.INFO]: {
        bgColor: 'bg-blue-900/20',
        borderColor: 'border-blue-700/50',
        textColor: 'text-blue-200',
        iconColor: 'text-blue-400',
        badgeColor: 'bg-blue-900/50',
        icon: <Info className="w-5 h-5" />
      },
      [SEVERITY.SUCCESS]: {
        bgColor: 'bg-green-900/20',
        borderColor: 'border-green-700/50',
        textColor: 'text-green-200',
        iconColor: 'text-green-400',
        badgeColor: 'bg-green-900/50',
        icon: <CheckCircle className="w-5 h-5" />
      }
    }
    return configs[severity] || configs[SEVERITY.INFO]
  }

  const renderValidation = (validation, index) => {
    const id = `${validation.severity}-${index}`
    const isExpanded = expandedIds.has(id)
    const config = getSeverityConfig(validation.severity)

    return (
      <div
        key={id}
        className={`${config.bgColor} ${config.borderColor} border rounded-lg overflow-hidden transition-all`}
      >
        {/* Header - Always visible */}
        <button
          onClick={() => toggleExpanded(id)}
          className="w-full p-4 flex items-start gap-3 hover:bg-slate-700/20 transition-colors"
        >
          <div className={`flex-shrink-0 mt-0.5 ${config.iconColor}`}>
            {validation.icon ? getIcon(validation.icon) : config.icon}
          </div>
          <div className="flex-1 text-left">
            <div className="flex items-start justify-between gap-2">
              <h4 className={`font-semibold ${config.textColor}`}>
                {validation.title}
              </h4>
              <div className="flex items-center gap-2 flex-shrink-0">
                {validation.severity === SEVERITY.ERROR && (
                  <span className="text-xs font-semibold text-red-300 bg-red-900/50 px-2 py-0.5 rounded-full">
                    BLOCKS PROGRESS
                  </span>
                )}
                <div className={`${config.iconColor}`}>
                  {isExpanded ? (
                    <ChevronUp className="w-4 h-4" />
                  ) : (
                    <ChevronDown className="w-4 h-4" />
                  )}
                </div>
              </div>
            </div>
            {!isExpanded && (
              <p className={`text-sm mt-1 ${config.textColor} opacity-80`}>
                {validation.message.substring(0, 100)}
                {validation.message.length > 100 ? '...' : ''}
              </p>
            )}
          </div>
        </button>

        {/* Expanded content */}
        {isExpanded && (
          <div className="px-4 pb-4 space-y-3 border-t border-slate-700/50">
            <div className="pt-3">
              <p className={`text-sm ${config.textColor}`}>
                {validation.message}
              </p>
            </div>

            {validation.recommendation && (
              <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/30">
                <div className="flex items-start gap-2">
                  <Lightbulb className={`w-4 h-4 flex-shrink-0 mt-0.5 ${config.iconColor}`} />
                  <div>
                    <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-1">
                      Recommendation
                    </p>
                    <p className="text-sm text-gray-300">
                      {validation.recommendation}
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    )
  }

  const renderCompactSummary = () => {
    const hasErrors = errors.length > 0
    const hasWarnings = warnings.length > 0
    const hasSuccess = success.length > 0

    if (!hasErrors && !hasWarnings && hasSuccess) {
      return (
        <div className="bg-green-900/20 border border-green-700/50 rounded-lg p-3 flex items-center gap-3">
          <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0" />
          <p className="text-sm text-green-200">
            <strong>All validations passed!</strong> Your design configuration looks excellent.
          </p>
        </div>
      )
    }

    return (
      <div className="flex flex-wrap gap-2">
        {hasErrors && (
          <div className="bg-red-900/30 border border-red-700/50 rounded-lg px-3 py-2 flex items-center gap-2">
            <XCircle className="w-4 h-4 text-red-400" />
            <span className="text-sm font-semibold text-red-200">
              {errors.length} Error{errors.length !== 1 ? 's' : ''}
            </span>
          </div>
        )}
        {hasWarnings && (
          <div className="bg-yellow-900/30 border border-yellow-700/50 rounded-lg px-3 py-2 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-yellow-400" />
            <span className="text-sm font-semibold text-yellow-200">
              {warnings.length} Warning{warnings.length !== 1 ? 's' : ''}
            </span>
          </div>
        )}
        {info.length > 0 && (
          <div className="bg-blue-900/30 border border-blue-700/50 rounded-lg px-3 py-2 flex items-center gap-2">
            <Info className="w-4 h-4 text-blue-400" />
            <span className="text-sm font-semibold text-blue-200">
              {info.length} Tip{info.length !== 1 ? 's' : ''}
            </span>
          </div>
        )}
      </div>
    )
  }

  if (compact) {
    return (
      <div className="space-y-3">
        {renderCompactSummary()}
        {(errors.length > 0 || warnings.length > 0) && (
          <div className="space-y-2">
            {errors.slice(0, 2).map((v, i) => renderValidation(v, i))}
            {warnings.slice(0, 1).map((v, i) => renderValidation(v, i))}
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <ShieldAlert className="w-6 h-6 text-blue-400" />
          <h3 className="text-lg font-bold text-gray-100">Smart Validation</h3>
        </div>
        <div className="flex gap-2 text-xs">
          {errors.length > 0 && (
            <span className="bg-red-900/30 text-red-200 px-2 py-1 rounded-full font-semibold">
              {errors.length} Error{errors.length !== 1 ? 's' : ''}
            </span>
          )}
          {warnings.length > 0 && (
            <span className="bg-yellow-900/30 text-yellow-200 px-2 py-1 rounded-full font-semibold">
              {warnings.length} Warning{warnings.length !== 1 ? 's' : ''}
            </span>
          )}
          {info.length > 0 && (
            <span className="bg-blue-900/30 text-blue-200 px-2 py-1 rounded-full font-semibold">
              {info.length} Tip{info.length !== 1 ? 's' : ''}
            </span>
          )}
          {success.length > 0 && (
            <span className="bg-green-900/30 text-green-200 px-2 py-1 rounded-full font-semibold">
              {success.length} Success
            </span>
          )}
        </div>
      </div>

      {/* Errors (Critical) */}
      {errors.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-red-300 text-sm font-semibold">
            <XCircle className="w-4 h-4" />
            <span>Critical Issues ({errors.length})</span>
          </div>
          {errors.map((v, i) => renderValidation(v, i))}
        </div>
      )}

      {/* Warnings (Important) */}
      {warnings.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-yellow-300 text-sm font-semibold">
            <AlertTriangle className="w-4 h-4" />
            <span>Warnings ({warnings.length})</span>
          </div>
          {warnings.map((v, i) => renderValidation(v, i))}
        </div>
      )}

      {/* Info (Helpful) */}
      {info.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-blue-300 text-sm font-semibold">
            <Info className="w-4 h-4" />
            <span>Tips & Information ({info.length})</span>
          </div>
          {info.map((v, i) => renderValidation(v, i))}
        </div>
      )}

      {/* Success (Positive feedback) */}
      {success.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-green-300 text-sm font-semibold">
            <CheckCircle className="w-4 h-4" />
            <span>Validation Passed ({success.length})</span>
          </div>
          {success.map((v, i) => renderValidation(v, i))}
        </div>
      )}

      {/* Footer note for errors */}
      {errors.length > 0 && (
        <div className="bg-red-900/10 border border-red-700/30 rounded-lg p-3">
          <p className="text-xs text-red-300">
            <strong>Note:</strong> Critical issues must be resolved before generating your design.
            Warnings can be acknowledged and proceeded with caution.
          </p>
        </div>
      )}
    </div>
  )
}

export default SmartValidation
