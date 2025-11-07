import { Mountain } from 'lucide-react'

const RSM = () => {
  return (
    <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-8 border border-slate-700/50">
      <div className="flex items-center space-x-3 mb-6">
        <Mountain className="w-8 h-8 text-orange-400" />
        <h2 className="text-3xl font-bold text-gray-100">Response Surface Methodology</h2>
      </div>
      <p className="text-gray-300 text-lg mb-4">
        Optimize processes and explore response surfaces.
      </p>
      <div className="bg-slate-700/50 rounded-lg p-6">
        <h3 className="text-gray-100 font-semibold mb-3">Features Coming Soon:</h3>
        <ul className="text-gray-300 space-y-2">
          <li>✓ Central Composite Design (CCD)</li>
          <li>✓ Box-Behnken designs</li>
          <li>✓ Second-order model fitting</li>
          <li>✓ Curvature detection</li>
          <li>✓ Steepest ascent/descent</li>
          <li>✓ Lack-of-fit testing</li>
          <li>✓ 3D response surface plots</li>
        </ul>
      </div>
    </div>
  )
}

export default RSM
