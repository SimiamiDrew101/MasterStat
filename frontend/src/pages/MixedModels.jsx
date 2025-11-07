import { Layers } from 'lucide-react'

const MixedModels = () => {
  return (
    <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-8 border border-slate-700/50">
      <div className="flex items-center space-x-3 mb-6">
        <Layers className="w-8 h-8 text-indigo-400" />
        <h2 className="text-3xl font-bold text-gray-100">Mixed Models</h2>
      </div>
      <p className="text-gray-300 text-lg mb-4">
        Analyze designs with both fixed and random effects.
      </p>
      <div className="bg-slate-700/50 rounded-lg p-6">
        <h3 className="text-gray-100 font-semibold mb-3">Features Coming Soon:</h3>
        <ul className="text-gray-300 space-y-2">
          <li>✓ Mixed model ANOVA</li>
          <li>✓ Split-plot designs</li>
          <li>✓ Nested designs</li>
          <li>✓ Expected Mean Squares (EMS)</li>
          <li>✓ Variance component estimation</li>
          <li>✓ Whole-plot and sub-plot analysis</li>
        </ul>
      </div>
    </div>
  )
}

export default MixedModels
