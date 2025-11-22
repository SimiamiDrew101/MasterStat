import { useState, useEffect } from 'react'
import axios from 'axios'
import { Loader2, Check, X, AlertTriangle, TrendingUp } from 'lucide-react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const DesignRecommendationStep = ({ nFactors, budget, goal, selectedDesign, onSelectDesign }) => {
  const [recommendations, setRecommendations] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [warning, setWarning] = useState(null)

  useEffect(() => {
    fetchRecommendations()
  }, [nFactors, budget, goal])

  const fetchRecommendations = async () => {
    setLoading(true)
    setError(null)
    setWarning(null)

    try {
      const response = await axios.post(`${API_URL}/api/rsm/recommend-design`, {
        n_factors: nFactors,
        budget: budget,
        goal: goal || 'optimization'
      })

      setRecommendations(response.data.recommendations || [])

      // Check for warnings (e.g., budget too low)
      if (response.data.warning) {
        setWarning(response.data.warning)
      }

      // Auto-select the top recommendation
      if (response.data.recommendations && response.data.recommendations.length > 0) {
        if (!selectedDesign) {
          onSelectDesign(response.data.recommendations[0])
        }
      }

    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to load recommendations')
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <Loader2 className="w-12 h-12 text-blue-500 animate-spin mb-4" />
        <p className="text-gray-300">Analyzing best designs for {nFactors} factors...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-red-900/20 border border-red-700/50 rounded-lg p-6">
        <div className="flex items-center gap-3 mb-2">
          <X className="w-6 h-6 text-red-400" />
          <h4 className="text-xl font-bold text-red-300">Error Loading Recommendations</h4>
        </div>
        <p className="text-red-200">{error}</p>
      </div>
    )
  }

  return (
    <div>
      <h3 className="text-2xl font-bold text-gray-100 mb-2">Recommended Designs</h3>
      <p className="text-gray-300 text-sm mb-6">
        Based on {nFactors} factors{budget ? ` and a budget of ${budget} runs` : ''}, here are the best design options:
      </p>

      {/* Warning (if budget is insufficient) */}
      {warning && (
        <div className="mb-6 bg-yellow-900/20 border border-yellow-700/50 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-yellow-300 font-semibold mb-1">Budget Constraint</p>
              <p className="text-yellow-200 text-sm">{warning}</p>
            </div>
          </div>
        </div>
      )}

      {/* Recommendations Grid */}
      <div className="space-y-4">
        {recommendations.map((design, index) => (
          <DesignCard
            key={design.type}
            design={design}
            isRecommended={index === 0}
            isSelected={selectedDesign?.type === design.type}
            onSelect={() => onSelectDesign(design)}
          />
        ))}
      </div>

      {/* Help Text */}
      <div className="mt-6 bg-blue-900/20 border border-blue-700/50 rounded-lg p-4">
        <p className="text-blue-200 text-sm">
          <strong>How to choose:</strong> The top recommendation is usually the best choice for most situations.
          Consider the pros/cons if you have specific constraints or requirements.
        </p>
      </div>
    </div>
  )
}

const DesignCard = ({ design, isRecommended, isSelected, onSelect }) => {
  const [expanded, setExpanded] = useState(false)

  return (
    <div
      className={`border-2 rounded-xl transition-all duration-200 ${
        isSelected
          ? 'border-blue-500 bg-blue-900/30 shadow-lg shadow-blue-500/20'
          : 'border-slate-600 bg-slate-800/30 hover:border-slate-500'
      }`}
    >
      {/* Header */}
      <div
        className="p-5 cursor-pointer"
        onClick={onSelect}
      >
        <div className="flex items-start justify-between mb-3">
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-2">
              {isRecommended && (
                <span className="px-2 py-1 bg-green-600 text-white text-xs font-bold rounded">
                  RECOMMENDED
                </span>
              )}
              <h4 className="text-xl font-bold text-gray-100">{design.type}</h4>
              <div
                className={`w-6 h-6 rounded-full border-2 flex items-center justify-center transition-all ${
                  isSelected
                    ? 'border-blue-500 bg-blue-500'
                    : 'border-slate-500 bg-transparent'
                }`}
              >
                {isSelected && <Check className="w-4 h-4 text-white" />}
              </div>
            </div>
            <p className="text-gray-300 text-sm mb-2">{design.description}</p>
            <p className="text-gray-400 text-sm italic">{design.best_for}</p>
          </div>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-4 gap-3 mb-3">
          <div className="bg-slate-700/50 rounded-lg p-3">
            <p className="text-gray-400 text-xs">Runs Required</p>
            <p className="text-xl font-bold text-blue-300">{design.runs}</p>
          </div>
          <div className="bg-slate-700/50 rounded-lg p-3">
            <p className="text-gray-400 text-xs">Rotatable</p>
            <p className="text-lg font-bold text-gray-200">
              {design.properties.rotatable ? '✓' : '—'}
            </p>
          </div>
          <div className="bg-slate-700/50 rounded-lg p-3">
            <p className="text-gray-400 text-xs">Orthogonal</p>
            <p className="text-lg font-bold text-gray-200">
              {design.properties.orthogonal ? '✓' : '—'}
            </p>
          </div>
          <div className="bg-slate-700/50 rounded-lg p-3">
            <p className="text-gray-400 text-xs">Score</p>
            <div className="flex items-center gap-1">
              <TrendingUp className="w-4 h-4 text-green-400" />
              <p className="text-lg font-bold text-green-300">{design.score}</p>
            </div>
          </div>
        </div>

        {/* Toggle Details */}
        <button
          onClick={(e) => {
            e.stopPropagation()
            setExpanded(!expanded)
          }}
          className="text-blue-400 text-sm font-medium hover:text-blue-300 transition-colors"
        >
          {expanded ? 'Hide Details' : 'Show Details'}
        </button>
      </div>

      {/* Expanded Details */}
      {expanded && (
        <div className="px-5 pb-5 border-t border-slate-700/50 pt-4">
          <div className="grid grid-cols-2 gap-4">
            {/* Pros */}
            <div>
              <h5 className="font-semibold text-green-400 mb-2 flex items-center gap-2">
                <Check className="w-4 h-4" />
                Pros
              </h5>
              <ul className="space-y-1">
                {design.pros.map((pro, i) => (
                  <li key={i} className="text-gray-300 text-sm flex items-start gap-2">
                    <span className="text-green-500 mt-0.5">•</span>
                    <span>{pro}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Cons */}
            <div>
              <h5 className="font-semibold text-red-400 mb-2 flex items-center gap-2">
                <X className="w-4 h-4" />
                Cons
              </h5>
              <ul className="space-y-1">
                {design.cons.map((con, i) => (
                  <li key={i} className="text-gray-300 text-sm flex items-start gap-2">
                    <span className="text-red-500 mt-0.5">•</span>
                    <span>{con}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Design Properties */}
          {design.properties && (
            <div className="mt-4 bg-slate-700/30 rounded-lg p-3">
              <h5 className="font-semibold text-gray-300 mb-2">Design Properties</h5>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
                {Object.entries(design.properties).map(([key, value]) => (
                  <div key={key}>
                    <span className="text-gray-400">{formatKey(key)}:</span>
                    <span className="text-gray-200 ml-1 font-medium">
                      {formatValue(value)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// Helper functions
const formatKey = (key) => {
  return key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

const formatValue = (value) => {
  if (typeof value === 'boolean') return value ? 'Yes' : 'No'
  if (typeof value === 'number') return value
  return value
}

export default DesignRecommendationStep
