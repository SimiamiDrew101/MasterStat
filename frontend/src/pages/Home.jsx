import { Link } from 'react-router-dom'
import { TrendingUp, Database, Beaker, Grid, Layers, Mountain, Network, Target, Sparkles, TestTubes, ShieldCheck, Heart, Coffee, Wand2, FileText, Activity, Gauge, SlidersHorizontal, Gem, GitBranch, BarChart2, Settings2, Cpu, LayoutGrid } from 'lucide-react'

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
      icon: <Wand2 className="w-12 h-12" />,
      title: 'Data Preprocessing',
      description: 'Transform, clean, impute missing values, and detect outliers',
      path: '/preprocessing',
      color: 'from-fuchsia-400 to-pink-600'
    },
    {
      icon: <FileText className="w-12 h-12" />,
      title: 'Protocol Generator',
      description: 'Create randomized, blinded experimental protocols with PDF export',
      path: '/protocol-generator',
      color: 'from-emerald-400 to-teal-600'
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
      description: 'Simplex, Extreme Vertices, ternary plots, trace plots for formulations',
      path: '/mixture',
      color: 'from-teal-400 to-teal-600'
    },
    {
      icon: <ShieldCheck className="w-12 h-12" />,
      title: 'Robust Design',
      description: 'Taguchi methods, noise factors, and parameter design',
      path: '/robust',
      color: 'from-amber-400 to-amber-600'
    },
    {
      icon: <Activity className="w-12 h-12" />,
      title: 'Reliability Analysis',
      description: 'Survival analysis, Weibull fitting, Kaplan-Meier, Cox regression',
      path: '/reliability',
      color: 'from-rose-400 to-red-600'
    },
    {
      icon: <Gauge className="w-12 h-12" />,
      title: 'Quality Control',
      description: 'Control charts, SPC, process capability analysis',
      path: '/quality-control',
      color: 'from-sky-400 to-blue-600'
    },
    {
      icon: <SlidersHorizontal className="w-12 h-12" />,
      title: 'Prediction Profiler',
      description: 'Interactive factor profiling and sensitivity analysis',
      path: '/prediction-profiler',
      color: 'from-violet-400 to-purple-600'
    },
    {
      icon: <Gem className="w-12 h-12" />,
      title: 'Optimal Designs',
      description: 'D-optimal, I-optimal, and A-optimal experimental designs',
      path: '/optimal-designs',
      color: 'from-yellow-400 to-orange-600'
    },
    {
      icon: <GitBranch className="w-12 h-12" />,
      title: 'Nonlinear Regression',
      description: 'Curve fitting for exponential, logistic, and custom models',
      path: '/nonlinear-regression',
      color: 'from-lime-400 to-green-600'
    },
    {
      icon: <BarChart2 className="w-12 h-12" />,
      title: 'Generalized Linear Models',
      description: 'Poisson, Binomial, Gamma, and Negative Binomial regression',
      path: '/glm',
      color: 'from-indigo-400 to-blue-600'
    },
    {
      icon: <Settings2 className="w-12 h-12" />,
      title: 'Custom Design',
      description: 'D-optimal, I-optimal, A-optimal designs with constraints',
      path: '/custom-design',
      color: 'from-cyan-400 to-teal-600'
    },
    {
      icon: <Cpu className="w-12 h-12" />,
      title: 'Predictive Modeling',
      description: 'Decision Trees, Random Forest, Gradient Boosting, Regularized Regression',
      path: '/predictive-modeling',
      color: 'from-pink-400 to-rose-600'
    },
    {
      icon: <LayoutGrid className="w-12 h-12" />,
      title: 'Graph Builder',
      description: 'Drag-and-drop visualization with scatter, bar, box, histogram, heatmap charts',
      path: '/graph-builder',
      color: 'from-indigo-400 to-purple-600'
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

      {/* Support Section */}
      <div className="bg-gradient-to-br from-pink-900/30 to-purple-900/30 backdrop-blur-lg rounded-xl p-8 border border-pink-700/30 text-center">
        <div className="flex items-center justify-center mb-4">
          <Heart className="w-8 h-8 text-pink-400 fill-pink-400 animate-pulse" />
        </div>
        <h3 className="text-2xl font-bold text-gray-100 mb-3">
          Support MasterStat
        </h3>
        <p className="text-gray-300 mb-6 max-w-2xl mx-auto">
          MasterStat is a free, open-source statistical analysis platform. If you find it useful for your research or work,
          consider supporting the project to help us continue developing new features and improvements.
        </p>
        <div className="flex items-center justify-center">
          <button
            onClick={() => window.open('https://ko-fi.com/MasterStat', '_blank')}
            className="group px-8 py-4 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 rounded-lg font-semibold text-white text-lg flex items-center gap-3 transition-all duration-300 hover:scale-105 shadow-lg"
          >
            <Coffee className="w-6 h-6 group-hover:scale-110 transition-transform" />
            Support us on Ko-fi
          </button>
        </div>
        <p className="text-gray-400 text-sm mt-4">
          Every contribution helps us improve MasterStat and keep it free for everyone
        </p>
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
