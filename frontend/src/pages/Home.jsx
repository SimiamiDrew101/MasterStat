import { Link } from 'react-router-dom'
import { TrendingUp, Database, Beaker, Grid, Layers, Mountain, Network, Target, Sparkles, TestTubes, ShieldCheck } from 'lucide-react'

const Home = () => {
  const features = [
    {
      icon: <Sparkles className="w-12 h-12" />,
      title: 'Experiment Wizard',
      description: 'Step-by-step guidance to design the perfect experiment',
      path: '/experiment-wizard',
      color: 'from-purple-400 to-purple-600'
    },
    {
      icon: <Target className="w-12 h-12" />,
      title: 'Experiment Planning',
      description: 'Sample size, power analysis, effect size calculations',
      path: '/planning',
      color: 'from-cyan-400 to-cyan-600'
    },
    {
      icon: <TrendingUp className="w-12 h-12" />,
      title: 'Hypothesis Testing',
      description: 't-tests, F-tests, Z-tests with confidence intervals',
      path: '/hypothesis',
      color: 'from-blue-400 to-blue-600'
    },
    {
      icon: <Database className="w-12 h-12" />,
      title: 'ANOVA',
      description: 'One-way, Two-way, and Post-hoc analysis',
      path: '/anova',
      color: 'from-green-400 to-green-600'
    },
    {
      icon: <Beaker className="w-12 h-12" />,
      title: 'Factorial Designs',
      description: 'Full and fractional factorial experiments',
      path: '/factorial',
      color: 'from-purple-400 to-purple-600'
    },
    {
      icon: <Grid className="w-12 h-12" />,
      title: 'Block Designs',
      description: 'RCBD, Latin Squares, and blocking strategies',
      path: '/blocks',
      color: 'from-pink-400 to-pink-600'
    },
    {
      icon: <Layers className="w-12 h-12" />,
      title: 'Mixed Models',
      description: 'Split-plot, nested designs, variance components',
      path: '/mixed',
      color: 'from-indigo-400 to-indigo-600'
    },
    {
      icon: <Mountain className="w-12 h-12" />,
      title: 'Response Surface',
      description: 'RSM, CCD, steepest ascent optimization',
      path: '/rsm',
      color: 'from-orange-400 to-orange-600'
    },
    {
      icon: <Network className="w-12 h-12" />,
      title: 'Bayesian DOE',
      description: 'Bayesian inference, sequential designs, optimal uncertainty',
      path: '/bayesian-doe',
      color: 'from-purple-400 to-purple-600'
    },
    {
      icon: <TestTubes className="w-12 h-12" />,
      title: 'Mixture Design',
      description: 'Simplex designs for mixture experiments and formulation optimization',
      path: '/mixture',
      color: 'from-teal-400 to-teal-600'
    },
    {
      icon: <ShieldCheck className="w-12 h-12" />,
      title: 'Robust Design',
      description: 'Taguchi methods, noise factors, and parameter design',
      path: '/robust',
      color: 'from-amber-400 to-amber-600'
    }
  ]

  return (
    <div className="space-y-8">
      {/* Features Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {features.map((feature, index) => (
          <Link
            key={index}
            to={feature.path}
            className="group bg-slate-800/50 backdrop-blur-lg rounded-xl p-6 border border-slate-700/50 hover:bg-slate-700/50 transition-all duration-300 hover:scale-105"
          >
            <div className={`w-16 h-16 rounded-lg bg-gradient-to-br ${feature.color} flex items-center justify-center text-white mb-4 group-hover:scale-110 transition-transform duration-300`}>
              {feature.icon}
            </div>
            <h3 className="text-xl font-bold text-gray-100 mb-2">{feature.title}</h3>
            <p className="text-gray-300">{feature.description}</p>
          </Link>
        ))}
      </div>

      {/* Footer */}
      <div className="text-center text-gray-400 text-sm">
        <p>Powered by Python, FastAPI, React, and Tailwind CSS</p>
        <p className="mt-1">Statistical computations using scipy, statsmodels, and pandas</p>
      </div>
    </div>
  )
}

export default Home
