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
  position = 'center' // 'center' = modal style (reliable), 'auto' = near trigger (complex)
}) => {
  const [isOpen, setIsOpen] = useState(false)
  const [tooltipPosition, setTooltipPosition] = useState({ top: '50%', left: '50%' })
  const [isDragging, setIsDragging] = useState(false)
  const [hasBeenDragged, setHasBeenDragged] = useState(false)
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 })
  const triggerRef = useRef(null)
  const tooltipRef = useRef(null)
  const hoverTimeoutRef = useRef(null)

  const glossaryEntry = getGlossaryTerm(term)

  // Handle drag functionality
  const handleMouseDown = (e) => {
    if (e.target.closest('.tooltip-drag-handle')) {
      e.preventDefault()

      if (!hasBeenDragged) {
        // First drag - convert from centered to absolute
        const rect = tooltipRef.current.getBoundingClientRect()
        setTooltipPosition({
          left: `${rect.left}px`,
          top: `${rect.top}px`
        })
        setHasBeenDragged(true)

        // Small delay to let position update, then start drag
        setTimeout(() => {
          const newRect = tooltipRef.current.getBoundingClientRect()
          setIsDragging(true)
          setDragOffset({
            x: e.clientX - newRect.left,
            y: e.clientY - newRect.top
          })
        }, 0)
      } else {
        // Already using absolute positioning
        const rect = tooltipRef.current.getBoundingClientRect()
        setIsDragging(true)
        setDragOffset({
          x: e.clientX - rect.left,
          y: e.clientY - rect.top
        })
      }
    }
  }

  const handleMouseMove = (e) => {
    if (isDragging && tooltipRef.current) {
      e.preventDefault()

      const newLeft = e.clientX - dragOffset.x
      const newTop = e.clientY - dragOffset.y

      // Get tooltip dimensions
      const rect = tooltipRef.current.getBoundingClientRect()

      // Calculate maximum positions (keep tooltip fully visible)
      const minLeft = 0
      const minTop = 0
      const maxLeft = window.innerWidth - rect.width
      const maxTop = window.innerHeight - rect.height

      // Constrain position to viewport
      const constrainedLeft = Math.max(minLeft, Math.min(newLeft, maxLeft))
      const constrainedTop = Math.max(minTop, Math.min(newTop, maxTop))

      setTooltipPosition({
        left: `${constrainedLeft}px`,
        top: `${constrainedTop}px`
      })
    }
  }

  const handleMouseUp = () => {
    setIsDragging(false)
  }

  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove)
      window.addEventListener('mouseup', handleMouseUp)
      return () => {
        window.removeEventListener('mousemove', handleMouseMove)
        window.removeEventListener('mouseup', handleMouseUp)
      }
    }
  }, [isDragging, dragOffset])

  // Reset position and drag state when opening/closing
  useEffect(() => {
    if (isOpen && position === 'center') {
      setTooltipPosition({ top: '50%', left: '50%' })
      setHasBeenDragged(false)
    } else if (!isOpen) {
      setHasBeenDragged(false)
      setIsDragging(false)
    }
  }, [isOpen, position])

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

  const handleClick = (e) => {
    e.stopPropagation()
    setIsOpen(!isOpen)
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
        className="inline-flex items-center gap-1 cursor-pointer"
        onClick={handleClick}
      >
        {children}
        <span className="text-blue-400 hover:text-blue-300 transition-colors">
          {getIcon()}
        </span>
      </span>

      {/* Backdrop */}
      {isOpen && position === 'center' && (
        <div
          className="fixed inset-0 bg-black/50 backdrop-blur-sm z-[9998] animate-fadeIn"
          onClick={() => setIsOpen(false)}
        />
      )}

      {/* Tooltip Portal */}
      {isOpen && (
        <div
          ref={tooltipRef}
          className="fixed z-[9999] bg-slate-800 border-2 border-blue-500/50 rounded-lg shadow-2xl max-w-md animate-fadeIn"
          style={{
            top: tooltipPosition.top,
            left: tooltipPosition.left,
            transform: position === 'center' && !hasBeenDragged ? 'translate(-50%, -50%)' : 'none',
            cursor: isDragging ? 'grabbing' : 'default',
            userSelect: isDragging ? 'none' : 'auto'
          }}
          onMouseDown={handleMouseDown}
        >
          {/* Header */}
          <div className="bg-gradient-to-r from-blue-900/50 to-purple-900/50 px-4 py-3 rounded-t-lg border-b border-blue-700/30 flex items-start justify-between tooltip-drag-handle cursor-move group">
            <div className="flex items-start gap-2 flex-1">
              <BookOpen className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <h4 className="font-bold text-gray-100 text-base flex items-center gap-2">
                  {glossaryEntry.term}
                  {position === 'center' && (
                    <span className="text-xs text-gray-400 font-normal opacity-0 group-hover:opacity-100 transition-opacity">
                      (Drag to move)
                    </span>
                  )}
                </h4>
                <p className="text-blue-200 text-xs mt-0.5 italic">{glossaryEntry.shortDefinition}</p>
              </div>
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation()
                setIsOpen(false)
              }}
              className="text-gray-400 hover:text-gray-200 transition-colors ml-2 flex-shrink-0"
            >
              <X className="w-4 h-4" />
            </button>
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
          {position === 'center' && (
            <div className="px-4 py-2 bg-slate-900/50 rounded-b-lg border-t border-slate-700/30">
              <p className="text-gray-500 text-xs text-center">
                💡 Drag header to reposition • Press ESC or click backdrop to close
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
export const InlineTooltip = ({ term, children, className = '', position = 'center' }) => {
  return (
    <InteractiveTooltip term={term} mode="click" position={position}>
      <span className={`border-b border-dotted border-blue-400 cursor-pointer ${className}`}>
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
    <InteractiveTooltip term={term} icon={icon} mode="click" position="center">
      <span className={`inline-block ${sizeClasses[size]}`} />
    </InteractiveTooltip>
  )
}

export default InteractiveTooltip
