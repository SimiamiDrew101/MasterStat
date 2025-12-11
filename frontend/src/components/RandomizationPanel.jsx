import { useState, useEffect } from 'react'
import { Shuffle, Lock, Unlock, RefreshCw, CheckCircle, AlertCircle, Info } from 'lucide-react'
import {
  completeRandomization,
  blockRandomization,
  restrictedRandomization,
  generateSeed,
  documentRandomization,
  verifyReproducibility,
} from '../utils/randomization'

const RandomizationPanel = ({ protocol, setProtocol }) => {
  const [previewRandomized, setPreviewRandomized] = useState(null)
  const [isRandomizing, setIsRandomizing] = useState(false)
  const [reproducibilityVerified, setReproducibilityVerified] = useState(false)

  // Check if design is available
  const hasDesign = protocol.materials?.factors && protocol.materials.factors.length > 0

  const handleRandomize = () => {
    if (!hasDesign) {
      alert('Please define factors in the Materials section first')
      return
    }

    const { method, seed, blockSize } = protocol.randomization

    if (!seed) {
      alert('Please set a random seed for reproducibility')
      return
    }

    setIsRandomizing(true)

    try {
      // Create a simple design matrix if none exists
      let designMatrix = protocol.randomization.randomizedDesign || []

      if (designMatrix.length === 0) {
        // Generate design from factors
        designMatrix = generateDesignFromFactors(protocol.materials.factors, protocol.materials.sampleSize)
      }

      let randomized

      switch (method) {
        case 'complete':
          randomized = completeRandomization(designMatrix, seed)
          break
        case 'block':
          if (!blockSize || blockSize < 2) {
            alert('Block size must be at least 2 for block randomization')
            setIsRandomizing(false)
            return
          }
          randomized = blockRandomization(designMatrix, blockSize, seed)
          break
        case 'restricted':
          // For restricted, use first factor as restriction
          const restrictionFactor = protocol.materials.factors[0]?.name || 'Factor1'
          randomized = restrictedRandomization(designMatrix, restrictionFactor, seed)
          break
        default:
          randomized = completeRandomization(designMatrix, seed)
      }

      setPreviewRandomized(randomized)
      setProtocol(prev => ({
        ...prev,
        randomization: {
          ...prev.randomization,
          randomizedDesign: randomized,
        },
      }))

      // Verify reproducibility
      const isReproducible = verifyReproducibility(designMatrix, seed, 3)
      setReproducibilityVerified(isReproducible)
    } catch (error) {
      console.error('Randomization error:', error)
      alert(`Randomization failed: ${error.message}`)
    } finally {
      setIsRandomizing(false)
    }
  }

  const regenerateSeed = () => {
    const newSeed = generateSeed()
    setProtocol(prev => ({
      ...prev,
      randomization: {
        ...prev.randomization,
        seed: newSeed,
      },
    }))
  }

  const copyDocumentation = () => {
    const doc = documentRandomization(protocol.randomization)
    navigator.clipboard.writeText(doc)
      .then(() => alert('Randomization documentation copied to clipboard'))
      .catch(err => console.error('Copy failed:', err))
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-100">Randomization</h2>
        {reproducibilityVerified && (
          <div className="flex items-center gap-2 text-green-400 text-sm">
            <CheckCircle className="w-4 h-4" />
            <span>Reproducibility Verified</span>
          </div>
        )}
      </div>

      {!hasDesign && (
        <div className="bg-orange-900/20 border border-orange-700 rounded-lg p-4">
          <div className="flex items-center gap-2 text-orange-400 mb-2">
            <AlertCircle className="w-5 h-5" />
            <span className="font-semibold">No Design Defined</span>
          </div>
          <p className="text-gray-300 text-sm">
            Please define factors in the Materials section before randomizing.
          </p>
        </div>
      )}

      {/* Randomization Method */}
      <div>
        <label className="block text-gray-200 font-medium mb-3">Randomization Method *</label>
        <div className="grid grid-cols-3 gap-3">
          {[
            { value: 'complete', label: 'Complete', description: 'Fully randomize all runs (CRD)' },
            { value: 'block', label: 'Block', description: 'Randomize within blocks (RBD)' },
            { value: 'restricted', label: 'Restricted', description: 'Balance throughout experiment' },
          ].map(method => (
            <button
              key={method.value}
              onClick={() => setProtocol(prev => ({
                ...prev,
                randomization: { ...prev.randomization, method: method.value }
              }))}
              className={`px-4 py-3 rounded-lg border-2 transition ${
                protocol.randomization.method === method.value
                  ? 'border-indigo-500 bg-indigo-600/20 text-indigo-300'
                  : 'border-slate-600 bg-slate-700/50 text-gray-300 hover:bg-slate-700'
              }`}
            >
              <div className="font-semibold mb-1">{method.label}</div>
              <div className="text-xs opacity-75">{method.description}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Random Seed - CRITICAL for reproducibility */}
      <div>
        <label className="block text-gray-200 font-medium mb-2">
          Random Seed (for reproducibility) *
        </label>
        <div className="flex gap-2">
          <input
            type="number"
            value={protocol.randomization.seed || ''}
            onChange={(e) => setProtocol(prev => ({
              ...prev,
              randomization: { ...prev.randomization, seed: parseInt(e.target.value) || null }
            }))}
            className="flex-1 px-4 py-3 bg-slate-700/50 border border-slate-600 rounded-lg text-gray-100 focus:ring-2 focus:ring-indigo-500"
            placeholder="123456"
          />
          <button
            onClick={regenerateSeed}
            className="px-4 py-3 bg-slate-700 hover:bg-slate-600 rounded-lg text-gray-300 transition"
            title="Generate new random seed"
          >
            <RefreshCw className="w-5 h-5" />
          </button>
          <button
            onClick={copyDocumentation}
            disabled={!protocol.randomization.seed}
            className="px-4 py-3 bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 disabled:cursor-not-allowed rounded-lg text-gray-300 transition"
            title="Copy randomization documentation"
          >
            <Lock className="w-5 h-5" />
          </button>
        </div>
        <div className="flex items-start gap-2 mt-2 text-sm text-blue-400">
          <Info className="w-4 h-4 mt-0.5 flex-shrink-0" />
          <p>
            Using the same seed will produce <strong>IDENTICAL</strong> randomization every time.
            This ensures your experiment is fully reproducible.
          </p>
        </div>
      </div>

      {/* Block Size (conditional) */}
      {protocol.randomization.method === 'block' && (
        <div>
          <label className="block text-gray-200 font-medium mb-2">Block Size *</label>
          <input
            type="number"
            value={protocol.randomization.blockSize || ''}
            onChange={(e) => setProtocol(prev => ({
              ...prev,
              randomization: { ...prev.randomization, blockSize: parseInt(e.target.value) || null }
            }))}
            className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600 rounded-lg text-gray-100 focus:ring-2 focus:ring-indigo-500"
            placeholder="4"
            min="2"
          />
          <p className="text-sm text-gray-400 mt-1">
            Number of runs to randomize within each block
          </p>
        </div>
      )}

      {/* Allocation Concealment */}
      <div>
        <label className="flex items-center gap-3 px-4 py-3 bg-slate-700/30 rounded-lg cursor-pointer hover:bg-slate-700/50 transition">
          <input
            type="checkbox"
            checked={protocol.randomization.allocationConcealment}
            onChange={(e) => setProtocol(prev => ({
              ...prev,
              randomization: { ...prev.randomization, allocationConcealment: e.target.checked }
            }))}
            className="w-5 h-5 rounded bg-slate-600 border-slate-500"
          />
          <div>
            <div className="text-gray-200 font-medium">Allocation Concealment</div>
            <div className="text-sm text-gray-400">Keep assignment sequence hidden until allocation</div>
          </div>
        </label>
      </div>

      {/* Randomize Button */}
      <button
        onClick={handleRandomize}
        disabled={!protocol.randomization.seed || !hasDesign || isRandomizing}
        className="w-full px-4 py-3 bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-600 disabled:cursor-not-allowed text-white rounded-lg font-semibold flex items-center justify-center gap-2 transition"
      >
        <Shuffle className="w-5 h-5" />
        {isRandomizing ? 'Randomizing...' : 'Apply Randomization'}
      </button>

      {/* Preview */}
      {previewRandomized && previewRandomized.length > 0 && (
        <div className="bg-slate-700/30 rounded-lg p-4 border border-slate-600">
          <div className="flex items-center justify-between mb-3">
            <h4 className="font-semibold text-gray-100">Randomized Run Order</h4>
            <span className="text-sm text-gray-400">
              {previewRandomized.length} runs total
            </span>
          </div>

          {/* Table Preview */}
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-600">
                  <th className="text-left py-2 px-2 text-gray-300">Run Order</th>
                  {protocol.randomization.method === 'block' && (
                    <th className="text-left py-2 px-2 text-gray-300">Block</th>
                  )}
                  <th className="text-left py-2 px-2 text-gray-300">Original</th>
                  {protocol.materials.factors.slice(0, 3).map((factor, idx) => (
                    <th key={idx} className="text-left py-2 px-2 text-gray-300">{factor.name}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {previewRandomized.slice(0, 15).map((run, idx) => (
                  <tr key={idx} className="border-b border-slate-700">
                    <td className="py-2 px-2 text-gray-100 font-mono">{run.runOrder}</td>
                    {protocol.randomization.method === 'block' && (
                      <td className="py-2 px-2 text-gray-400">{run.block || '-'}</td>
                    )}
                    <td className="py-2 px-2 text-gray-400">{run.originalOrder}</td>
                    {protocol.materials.factors.slice(0, 3).map((factor, fidx) => (
                      <td key={fidx} className="py-2 px-2 text-gray-300">
                        {run[factor.name] !== undefined ? run[factor.name] : '-'}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {previewRandomized.length > 15 && (
            <p className="text-gray-400 text-xs mt-2 text-center">
              ... and {previewRandomized.length - 15} more runs
            </p>
          )}

          {/* Reproducibility Badge */}
          {reproducibilityVerified && (
            <div className="mt-4 p-3 bg-green-900/20 border border-green-700 rounded-lg">
              <div className="flex items-center gap-2 text-green-400 text-sm">
                <CheckCircle className="w-4 h-4" />
                <span className="font-semibold">Reproducibility Confirmed</span>
              </div>
              <p className="text-green-300 text-xs mt-1">
                Using seed {protocol.randomization.seed} will always produce this exact run order
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

/**
 * Helper: Generate simple design matrix from factors
 */
const generateDesignFromFactors = (factors, sampleSize) => {
  if (!factors || factors.length === 0) {
    return []
  }

  // For demonstration, generate a simple full factorial or placeholder
  const design = []
  const runsNeeded = sampleSize || Math.pow(2, Math.min(factors.length, 3))

  for (let i = 0; i < runsNeeded; i++) {
    const run = {}
    factors.forEach((factor, idx) => {
      // Alternate between low and high
      if (factor.low !== undefined && factor.high !== undefined) {
        run[factor.name] = i % 2 === 0 ? factor.low : factor.high
      } else {
        run[factor.name] = i % 2 === 0 ? -1 : 1
      }
    })
    design.push(run)
  }

  return design
}

export default RandomizationPanel
