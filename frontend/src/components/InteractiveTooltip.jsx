import { useState, useRef, useEffect } from 'react'
import { Info, HelpCircle, BookOpen, Lightbulb, ArrowRight, X } from 'lucide-react'
import { getGlossaryTerm } from '../utils/doeGlossary'

/**
 * Interactive Tooltip Component
 * Displays educational DOE concept explanations on hover or click
 *
 * Usage:
 * <InteractiveTooltip term="ccd">Central Composite Design</InteractiveTooltip>
 * <InteractiveTooltip term="statistical-power" icon="help" />
 */
const InteractiveTooltip = ({
  term,
  children,
  icon = 'info',
  mode = 'hover', // 'hover', 'click', or 'both'
  position = 'top' // 'top', 'bottom', 'left', 'right'
}) => {
  const [isOpen, setIsOpen] = useState(false)
  const [tooltipPosition, setTooltipPosition] = useState({ top: 0, left: 0, opacity: 0 })
  const [isPositioned, setIsPositioned] = useState(false)
  const triggerRef = useRef(null)
  const tooltipRef = useRef(null)
  const hoverTimeoutRef = useRef(null)

  const glossaryEntry = getGlossaryTerm(term)

  useEffect(() => {
    if (isOpen && triggerRef.current && tooltipRef.current) {
      // Use requestAnimationFrame to ensure DOM is ready
      requestAnimationFrame(() => {
        if (!triggerRef.current || !tooltipRef.current) return

        const triggerRect = triggerRef.current.getBoundingClientRect()
        const tooltipRect = tooltipRef.current.getBoundingClientRect()
        const viewportWidth = window.innerWidth
        const viewportHeight = window.innerHeight
        const scrollY = window.scrollY
        const scrollX = window.scrollX

        let top = 0
        let left = 0
        const gap = 12 // Gap between trigger and tooltip
        const padding = 16 // Padding from viewport edges

        // Calculate position based on preference and available space
        switch (position) {
          case 'top':
            top = triggerRect.top - tooltipRect.height - gap
            left = triggerRect.left + (triggerRect.width / 2) - (tooltipRect.width / 2)

            // Fallback to bottom if not enough space above
            if (top < padding) {
              top = triggerRect.bottom + gap
            }
            break

          case 'bottom':
            top = triggerRect.bottom + gap
            left = triggerRect.left + (triggerRect.width / 2) - (tooltipRect.width / 2)

            // Fallback to top if not enough space below
            if (top + tooltipRect.height > viewportHeight - padding) {
              top = triggerRect.top - tooltipRect.height - gap
            }
            break

          case 'left':
            top = triggerRect.top + (triggerRect.height / 2) - (tooltipRect.height / 2)
            left = triggerRect.left - tooltipRect.width - gap

            // Fallback to right if not enough space on left
            if (left < padding) {
              left = triggerRect.right + gap
            }
            break

          case 'right':
            top = triggerRect.top + (triggerRect.height / 2) - (tooltipRect.height / 2)
            left = triggerRect.right + gap

            // Fallback to left if not enough space on right
            if (left + tooltipRect.width > viewportWidth - padding) {
              left = triggerRect.left - tooltipRect.width - gap
            }
            break

          default:
            top = triggerRect.bottom + gap
            left = triggerRect.left
        }

        // Constrain to viewport with padding
        top = Math.max(padding, Math.min(top, viewportHeight - tooltipRect.height - padding))
        left = Math.max(padding, Math.min(left, viewportWidth - tooltipRect.width - padding))

        // Ensure tooltip stays within viewport even after constraining
        if (left + tooltipRect.width > viewportWidth - padding) {
          left = viewportWidth - tooltipRect.width - padding
        }
        if (top + tooltipRect.height > viewportHeight - padding) {
          top = viewportHeight - tooltipRect.height - padding
        }

        setTooltipPosition({ top, left, opacity: 1 })
        setIsPositioned(true)
      })
    } else {
      setIsPositioned(false)
      setTooltipPosition({ top: 0, left: 0, opacity: 0 })
    }
  }, [isOpen, position])

  // Reposition on scroll or resize
  useEffect(() => {
    if (!isOpen) return

    const handleRepositioning = () => {
      setIsPositioned(false)
      // Trigger recalculation
      if (triggerRef.current && tooltipRef.current) {
        setIsOpen(true)
      }
    }

    window.addEventListener('scroll', handleRepositioning, true)
    window.addEventListener('resize', handleRepositioning)

    return () => {
      window.removeEventListener('scroll', handleRepositioning, true)
      window.removeEventListener('resize', handleRepositioning)
    }
  }, [isOpen])

  // Close on escape key
  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape' && isOpen) {
        setIsOpen(false)
      }
    }
    window.addEventListener('keydown', handleEscape)
    return () => window.removeEventListener('keydown', handleEscape)
  }, [isOpen])

  // Close when clicking outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (
        isOpen &&
        tooltipRef.current &&
        !tooltipRef.current.contains(e.target) &&
        triggerRef.current &&
        !triggerRef.current.contains(e.target)
      ) {
        setIsOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [isOpen])

  if (!glossaryEntry) {
    console.warn(`InteractiveTooltip: No glossary entry found for term "${term}"`)
    return children ? <span>{children}</span> : null
  }

  const getIcon = () => {
    const iconProps = { className: 'w-4 h-4' }
    switch (icon) {
      case 'help':
        return <HelpCircle {...iconProps} />
      case 'book':
        return <BookOpen {...iconProps} />
      case 'lightbulb':
        return <Lightbulb {...iconProps} />
      default:
        return <Info {...iconProps} />
    }
  }

  const handleMouseEnter = () => {
    if (mode === 'hover' || mode === 'both') {
      // Clear any pending close timeout
      if (hoverTimeoutRef.current) {
        clearTimeout(hoverTimeoutRef.current)
        hoverTimeoutRef.current = null
      }
      setIsOpen(true)
    }
  }

  const handleMouseLeave = () => {
    if (mode === 'hover') {
      // Delay closing to allow moving mouse to tooltip
      hoverTimeoutRef.current = setTimeout(() => {
        setIsOpen(false)
      }, 100)
    }
  }

  const handleTooltipMouseEnter = () => {
    if (mode === 'hover' || mode === 'both') {
      // Clear close timeout when hovering over tooltip
      if (hoverTimeoutRef.current) {
        clearTimeout(hoverTimeoutRef.current)
        hoverTimeoutRef.current = null
      }
    }
  }

  const handleTooltipMouseLeave = () => {
    if (mode === 'hover') {
      setIsOpen(false)
    }
  }

  const handleClick = (e) => {
    e.stopPropagation()
    if (mode === 'click' || mode === 'both') {
      setIsOpen(!isOpen)
    }
  }

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (hoverTimeoutRef.current) {
        clearTimeout(hoverTimeoutRef.current)
      }
    }
  }, [])

  return (
    <>
      <span
        ref={triggerRef}
        className="inline-flex items-center gap-1 cursor-help"
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
      >
        {children}
        <span className="text-blue-400 hover:text-blue-300 transition-colors">
          {getIcon()}
        </span>
      </span>

      {/* Tooltip Portal */}
      {isOpen && (
        <div
          ref={tooltipRef}
          className="fixed z-50 bg-slate-800 border-2 border-blue-500/50 rounded-lg shadow-2xl max-w-md transition-opacity duration-200"
          style={{
            top: `${tooltipPosition.top}px`,
            left: `${tooltipPosition.left}px`,
            opacity: tooltipPosition.opacity,
            pointerEvents: isPositioned ? 'auto' : 'none'
          }}
          onMouseEnter={handleTooltipMouseEnter}
          onMouseLeave={handleTooltipMouseLeave}
        >
          {/* Header */}
          <div className="bg-gradient-to-r from-blue-900/50 to-purple-900/50 px-4 py-3 rounded-t-lg border-b border-blue-700/30 flex items-start justify-between">
            <div className="flex items-start gap-2 flex-1">
              <BookOpen className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
              <div>
                <h4 className="font-bold text-gray-100 text-base">{glossaryEntry.term}</h4>
                <p className="text-blue-200 text-xs mt-0.5 italic">{glossaryEntry.shortDefinition}</p>
              </div>
            </div>
            {mode === 'click' || mode === 'both' ? (
              <button
                onClick={() => setIsOpen(false)}
                className="text-gray-400 hover:text-gray-200 transition-colors ml-2"
              >
                <X className="w-4 h-4" />
              </button>
            ) : null}
          </div>

          {/* Content */}
          <div className="p-4 overflow-y-auto custom-scrollbar" style={{ maxHeight: 'calc(80vh - 120px)' }}>
            {/* Full Explanation */}
            <div className="mb-4">
              <p className="text-gray-300 text-sm leading-relaxed whitespace-pre-line">
                {glossaryEntry.fullExplanation}
              </p>
            </div>

            {/* Practical Advice */}
            {glossaryEntry.practicalAdvice && (
              <div className="mb-4 bg-green-900/20 border border-green-700/30 rounded-lg p-3">
                <div className="flex items-start gap-2">
                  <Lightbulb className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="text-green-300 text-xs font-semibold mb-1">Practical Advice</p>
                    <p className="text-gray-300 text-xs leading-relaxed">
                      {glossaryEntry.practicalAdvice}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Examples */}
            {glossaryEntry.examples && glossaryEntry.examples.length > 0 && (
              <div className="mb-4">
                <p className="text-yellow-300 text-xs font-semibold mb-2 flex items-center gap-1">
                  <span className="text-yellow-400">📊</span> Examples
                </p>
                <ul className="space-y-1">
                  {glossaryEntry.examples.map((example, idx) => (
                    <li key={idx} className="text-gray-400 text-xs flex items-start gap-2">
                      <span className="text-yellow-400 flex-shrink-0">•</span>
                      <span>{example}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Related Terms */}
            {glossaryEntry.relatedTerms && glossaryEntry.relatedTerms.length > 0 && (
              <div>
                <p className="text-blue-300 text-xs font-semibold mb-2">Related Concepts</p>
                <div className="flex flex-wrap gap-2">
                  {glossaryEntry.relatedTerms.map((relatedTerm, idx) => {
                    const related = getGlossaryTerm(relatedTerm)
                    return related ? (
                      <InteractiveTooltip key={idx} term={relatedTerm} mode="click">
                        <span className="inline-flex items-center gap-1 px-2 py-1 bg-blue-900/30 border border-blue-700/30 rounded text-blue-200 text-xs hover:bg-blue-900/50 transition-colors cursor-pointer">
                          {related.term}
                          <ArrowRight className="w-3 h-3" />
                        </span>
                      </InteractiveTooltip>
                    ) : null
                  })}
                </div>
              </div>
            )}
          </div>

          {/* Footer hint */}
          {(mode === 'hover' || mode === 'both') && (
            <div className="px-4 py-2 bg-slate-900/50 rounded-b-lg border-t border-slate-700/30">
              <p className="text-gray-500 text-xs text-center">
                {mode === 'both' ? 'Click to pin • ' : ''}Hover to view • Press ESC to close
              </p>
            </div>
          )}
        </div>
      )}
    </>
  )
}

/**
 * Inline Tooltip - Simpler version for inline text with dotted underline
 */
export const InlineTooltip = ({ term, children, className = '' }) => {
  return (
    <InteractiveTooltip term={term} mode="hover" position="top">
      <span className={`border-b border-dotted border-blue-400 cursor-help ${className}`}>
        {children}
      </span>
    </InteractiveTooltip>
  )
}

/**
 * Tooltip Icon - Just the icon, no text
 */
export const TooltipIcon = ({ term, icon = 'info', size = 'sm' }) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-5 h-5',
    lg: 'w-6 h-6'
  }

  return (
    <InteractiveTooltip term={term} icon={icon} mode="both" position="top">
      <span className={`inline-block ${sizeClasses[size]}`} />
    </InteractiveTooltip>
  )
}

export default InteractiveTooltip
