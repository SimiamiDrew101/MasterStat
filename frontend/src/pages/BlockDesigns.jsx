import { Grid } from 'lucide-react'

const BlockDesigns = () => {
  return (
    <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-8 border border-slate-700/50">
      <div className="flex items-center space-x-3 mb-6">
        <Grid className="w-8 h-8 text-pink-400" />
        <h2 className="text-3xl font-bold text-gray-100">Block Designs</h2>
      </div>
      <p className="text-gray-300 text-lg mb-4">
        Control for nuisance factors using blocking strategies.
      </p>
      <div className="bg-slate-700/50 rounded-lg p-6">
        <h3 className="text-gray-100 font-semibold mb-3">Features Coming Soon:</h3>
        <ul className="text-gray-300 space-y-2">
          <li>✓ Randomized Complete Block Design (RCBD)</li>
          <li>✓ Latin Square designs</li>
          <li>✓ Graeco-Latin squares</li>
          <li>✓ Fixed and random blocks</li>
          <li>✓ Blocking efficiency analysis</li>
          <li>✓ Design generation tools</li>
        </ul>
      </div>
    </div>
  )
}

export default BlockDesigns
