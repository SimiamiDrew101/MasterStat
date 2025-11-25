// DOE Glossary - Comprehensive educational content for interactive tooltips
// Provides clear, practical explanations of Design of Experiments concepts

/**
 * DOE Glossary Database
 * Each entry contains:
 * - term: The technical term
 * - shortDefinition: Brief one-liner
 * - fullExplanation: Detailed explanation with examples
 * - practicalAdvice: When/how to use this concept
 * - relatedTerms: Array of related glossary terms
 * - examples: Real-world examples (optional)
 */

export const DOE_GLOSSARY = {
  // Design Types
  'full-factorial': {
    term: 'Full Factorial Design',
    shortDefinition: 'Tests every combination of factor levels',
    fullExplanation: `A full factorial design tests all possible combinations of factor levels. For example, with 3 factors at 2 levels each (low/high), you get 2³ = 8 experimental runs. This provides complete information about main effects and all interactions, but run count grows exponentially with factors.`,
    practicalAdvice: 'Use full factorial when you have 2-4 factors and want complete information about interactions. Beyond 5 factors, consider fractional factorial or screening designs to reduce run count.',
    relatedTerms: ['fractional-factorial', 'interactions', 'main-effects'],
    examples: [
      '3 factors, 2 levels: 8 runs',
      '4 factors, 2 levels: 16 runs',
      '5 factors, 2 levels: 32 runs (often too many!)'
    ]
  },

  'fractional-factorial': {
    term: 'Fractional Factorial Design',
    shortDefinition: 'Tests a strategic subset of factor combinations',
    fullExplanation: `Fractional factorial designs test a carefully selected fraction (e.g., 1/2, 1/4) of all possible combinations. A 2⁵⁻¹ design tests 16 runs instead of 32, using confounding patterns to estimate main effects. This trades some information for efficiency, making it practical for screening many factors.`,
    practicalAdvice: 'Excellent for screening 5-8 factors. Choose Resolution IV or V designs when possible - they provide cleaner separation of main effects from interactions. Check the confounding pattern to understand which effects are aliased.',
    relatedTerms: ['confounding', 'resolution', 'screening', 'aliasing'],
    examples: [
      '2⁵⁻¹: 16 runs instead of 32 (half-fraction)',
      '2⁷⁻³: 16 runs for 7 factors (1/8 fraction)',
      'Resolution V: Main effects clear of 2-way interactions'
    ]
  },

  'ccd': {
    term: 'Central Composite Design (CCD)',
    shortDefinition: 'Gold standard for response surface methodology',
    fullExplanation: `Central Composite Design is the most popular RSM design. It combines: (1) factorial points at the corners, (2) center points at the middle, and (3) axial (star) points extending beyond the cube. This structure allows fitting quadratic models to find optimal factor settings. The axial points enable estimation of curvature.`,
    practicalAdvice: 'Use CCD when your goal is optimization with 2-5 factors. It requires more runs than screening but provides excellent quadratic modeling capability. The design can find optimal settings and generate response surface plots. Choose "face-centered" if you cannot go beyond the factor ranges.',
    relatedTerms: ['rsm', 'optimization', 'axial-points', 'center-points', 'quadratic'],
    examples: [
      '3 factors: 8 corner + 6 axial + 4 center = 18 runs',
      '4 factors: 16 corner + 8 axial + 5 center = 29 runs',
      'Can model Y = β₀ + β₁X₁ + β₂X₂ + β₁₁X₁² + β₁₂X₁X₂'
    ]
  },

  'box-behnken': {
    term: 'Box-Behnken Design',
    shortDefinition: 'RSM design that avoids extreme corners',
    fullExplanation: `Box-Behnken designs test edge midpoints and center points but deliberately avoid corner points (all factors at extreme values simultaneously). This makes them safer when extreme combinations might be dangerous, impractical, or expensive. However, if the optimum happens to be at a corner, you'll miss it.`,
    practicalAdvice: 'Choose Box-Behnken when: (1) extreme factor combinations are risky or impossible to test, (2) you need fewer runs than CCD (for 3 factors: 15 vs 20), or (3) corner points are outside your feasible region. Requires 3+ factors (cannot be used with only 2 factors).',
    relatedTerms: ['ccd', 'rsm', 'optimization', 'edge-points'],
    examples: [
      '3 factors: 12 edge + 3 center = 15 runs',
      'Avoids (High, High, High) and (Low, Low, Low) corners',
      'Good when extremes are dangerous (high temp + high pressure)'
    ]
  },

  'plackett-burman': {
    term: 'Plackett-Burman Design',
    shortDefinition: 'Ultra-efficient screening for many factors',
    fullExplanation: `Plackett-Burman designs are specially constructed screening designs that allow testing N-1 factors in N runs (where N is a multiple of 4). For example, screen 11 factors in just 12 runs! They assume all interactions are negligible and focus solely on identifying which main effects are significant.`,
    practicalAdvice: 'Perfect for initial screening of 5+ factors when you suspect only a few are important. Very efficient but assumes no interactions exist. Use Pareto charts and half-normal plots to identify active factors. Follow up with full or fractional factorial on the 2-4 most important factors.',
    relatedTerms: ['screening', 'main-effects', 'fractional-factorial'],
    examples: [
      '7 factors in 8 runs',
      '11 factors in 12 runs',
      'Use for "Which factors matter?" not "How do they interact?"'
    ]
  },

  // Core Concepts
  'main-effects': {
    term: 'Main Effects',
    shortDefinition: 'The individual impact of each factor on the response',
    fullExplanation: `A main effect is the average change in response when a factor moves from its low to high level, ignoring all other factors. For example, if increasing temperature from 150°C to 200°C increases yield by 10% on average (across all other factor settings), the main effect of temperature is +10%.`,
    practicalAdvice: 'Main effects tell you which factors matter and in what direction. Always analyze main effects first before looking at interactions. Use main effects plots to visualize the impact. Larger main effects indicate more important factors that should be included in optimization.',
    relatedTerms: ['interactions', 'pareto-chart', 'effect-size'],
    examples: [
      'Temperature main effect: +12% yield',
      'Pressure main effect: +3% yield',
      'Catalyst main effect: -2% yield'
    ]
  },

  'interactions': {
    term: 'Interaction Effects',
    shortDefinition: 'When the effect of one factor depends on another factor',
    fullExplanation: `An interaction occurs when the effect of one factor changes depending on the level of another factor. For example: at low temperature, pressure has no effect on yield; but at high temperature, increasing pressure boosts yield by 15%. This is a Temperature × Pressure interaction. The two factors work together synergistically.`,
    practicalAdvice: 'Interactions are crucial for optimization! They reveal factor combinations that work better together. Use interaction plots to visualize them - look for non-parallel lines. Full factorial and Resolution V designs can estimate 2-way interactions. Consider interactions when factors have strong scientific relationships.',
    relatedTerms: ['main-effects', 'full-factorial', 'synergy'],
    examples: [
      'Temp × Pressure: High-high gives 20% boost, but high-low gives 5%',
      'Non-parallel lines in interaction plot indicate interaction',
      'Synergistic: 1+1=3 (factors work together better than alone)'
    ]
  },

  'resolution': {
    term: 'Design Resolution',
    shortDefinition: 'Clarity of separation between effects (higher = better)',
    fullExplanation: `Resolution describes which effects are confounded (mixed together) in a fractional factorial design:

- Resolution III: Main effects confounded with 2-way interactions (avoid!)
- Resolution IV: Main effects clear, but 2-way interactions confounded with each other
- Resolution V: Main effects and 2-way interactions both clear (best for screening)

Higher resolution = better, but requires more runs.`,
    practicalAdvice: 'For screening, always choose Resolution IV or V if your budget allows. Resolution III designs are risky because main effects can be mistaken for interactions. Resolution V is ideal for important experiments where you need reliable results.',
    relatedTerms: ['fractional-factorial', 'confounding', 'aliasing'],
    examples: [
      'Resolution III: Avoid unless desperate',
      'Resolution IV: Good for screening main effects',
      'Resolution V: Gold standard for screening'
    ]
  },

  'confounding': {
    term: 'Confounding / Aliasing',
    shortDefinition: 'When two effects are mixed together and cannot be separated',
    fullExplanation: `Confounding (also called aliasing) happens in fractional factorial designs when the experiment cannot distinguish between two effects. For example, in a Resolution III design, the main effect A might be confounded with the BC interaction. You get one combined estimate, but cannot tell if it is due to A, BC, or both.`,
    practicalAdvice: 'Check the confounding pattern before running a fractional factorial! Most statistical software shows which effects are aliased. Design your experiment so important effects are not confounded with each other. Use follow-up experiments if you need to "break" critical confounding patterns.',
    relatedTerms: ['resolution', 'fractional-factorial', 'aliasing'],
    examples: [
      'A confounded with BC: A = A + BC (mixed)',
      'Resolution IV: AB confounded with CD',
      'Resolution V: Main effects never confounded with 2-way interactions'
    ]
  },

  'orthogonality': {
    term: 'Orthogonality',
    shortDefinition: 'Factors are independent - effects can be estimated separately',
    fullExplanation: `An orthogonal design means factors are statistically independent - changing one factor does not force you to change another. Mathematically, factor columns are perpendicular (dot product = 0). This ensures main effects and interactions can be estimated independently without correlation. Nearly all classical DOE designs are orthogonal.`,
    practicalAdvice: 'Orthogonality is a key strength of designed experiments over observational data! It ensures your effect estimates are unbiased and uncorrelated. Full factorials, fractional factorials, and CCD are all orthogonal. This means the estimate of Factor A is the same regardless of what Factor B does.',
    relatedTerms: ['independence', 'correlation', 'full-factorial'],
    examples: [
      'Full factorial at (-1, +1): Perfectly orthogonal',
      'Observational data: Usually NOT orthogonal',
      'Orthogonal columns → Independent effect estimates'
    ]
  },

  // Power and Statistics
  'statistical-power': {
    term: 'Statistical Power',
    shortDefinition: 'Probability of detecting a real effect (usually aim for 80%+)',
    fullExplanation: `Statistical power is the probability that your experiment will detect an effect if it truly exists. For example, 80% power means: "If temperature really increases yield by 5%, I have an 80% chance of proving it statistically significant." Higher power requires more runs or larger effect sizes. Low power risks missing important effects (Type II error).`,
    practicalAdvice: 'Aim for 80-90% power in experiments. Power increases with: (1) more experimental runs, (2) larger effect sizes, or (3) lower measurement noise. Use power analysis to determine minimum sample size before running expensive experiments. Underpowered experiments waste resources by missing real effects.',
    relatedTerms: ['effect-size', 'type-ii-error', 'sample-size'],
    examples: [
      '80% power = 20% risk of missing a real effect',
      '90% power = Better but requires ~30% more runs',
      'Doubling runs increases power but costs more'
    ]
  },

  'effect-size': {
    term: 'Effect Size',
    shortDefinition: 'Magnitude of the change you want to detect',
    fullExplanation: `Effect size measures how large a change you want to detect relative to noise (standard deviation). Cohen's d standardizes this:

- Small: d = 0.2 (subtle effect, hard to detect)
- Medium: d = 0.5 (moderate effect, common in practice)
- Large: d = 0.8 (obvious effect, easy to detect)

Larger effects require fewer runs to detect. Detecting small effects requires well-controlled experiments with many replicates.`,
    practicalAdvice: 'Be realistic about effect sizes! If you expect a 5% yield improvement (small), you need more runs than if you expect 20% (large). Use historical data or pilot studies to estimate expected effects. "Medium" effect size is a reasonable default if you\'re unsure.',
    relatedTerms: ['statistical-power', 'sample-size', 'noise'],
    examples: [
      'Large effect: 20% yield increase → Detect with 10 runs',
      'Small effect: 2% yield increase → Need 50+ runs',
      'Cohen\'s d = (mean difference) / (standard deviation)'
    ]
  },

  'type-i-error': {
    term: 'Type I Error (False Positive)',
    shortDefinition: 'Incorrectly concluding an effect exists when it doesn\'t',
    fullExplanation: `Type I error is declaring a factor significant when it actually has no effect - a false alarm. The significance level (α, usually 0.05 or 5%) controls this risk. With α = 0.05, you accept a 5% chance of false positives. Testing many factors without correction increases this risk (multiple testing problem).`,
    practicalAdvice: 'Use α = 0.05 (5% risk) as default. When screening many factors, consider Bonferroni correction or FDR adjustment to control overall false positive rate. False positives waste resources in follow-up experiments by pursuing factors that don\'t actually matter.',
    relatedTerms: ['type-ii-error', 'p-value', 'significance'],
    examples: [
      'α = 0.05: 5% chance of declaring a non-effect significant',
      'Testing 20 factors: Expect 1 false positive by chance alone',
      'p < 0.05 used to reject null hypothesis'
    ]
  },

  'type-ii-error': {
    term: 'Type II Error (False Negative)',
    shortDefinition: 'Missing a real effect - failing to detect something important',
    fullExplanation: `Type II error is failing to detect a factor that actually matters - you conclude it has no effect when it does. The risk is β, and power = 1 - β. With 80% power, β = 0.20 (20% risk of missing real effects). Type II errors waste opportunities by discarding important factors.`,
    practicalAdvice: 'Minimize Type II error by: (1) increasing sample size, (2) improving measurement precision, (3) using efficient designs. Type II errors are especially costly in screening - you might eliminate the most important factor! Balance against cost: more runs = more power = less risk.',
    relatedTerms: ['statistical-power', 'type-i-error', 'sample-size'],
    examples: [
      'β = 0.20: 20% risk of missing a real 5% effect',
      'Underpowered study → High Type II error risk',
      'Increasing runs from 16 to 32 → Better power → Less β'
    ]
  },

  // RSM Concepts
  'rsm': {
    term: 'Response Surface Methodology (RSM)',
    shortDefinition: 'Statistical method for finding optimal factor settings',
    fullExplanation: `RSM uses designed experiments and quadratic models to map the response surface and find optimal factor combinations. It fits models like Y = β₀ + β₁X₁ + β₂X₂ + β₁₁X₁² + β₁₂X₁X₂ to capture curvature. You can then use contour plots and 3D surfaces to visualize the optimum and generate predictions.`,
    practicalAdvice: 'Use RSM when your goal is optimization (finding best settings) rather than screening. Requires designs with center points and multiple levels (CCD, Box-Behnken). RSM is most effective with 2-5 factors after screening. Validates the model with ANOVA and checks assumptions (residual plots).',
    relatedTerms: ['ccd', 'box-behnken', 'optimization', 'quadratic', 'contour-plots'],
    examples: [
      'Find temperature and pressure that maximize yield',
      'Generate 3D surface plot showing "sweet spot"',
      'Predict response at any factor combination within the design space'
    ]
  },

  'center-points': {
    term: 'Center Points',
    shortDefinition: 'Runs at the middle of all factor ranges',
    fullExplanation: `Center points are experimental runs where all factors are set to their middle levels (0 in coded units). They serve two critical purposes: (1) detect curvature by comparing center average to factorial average, and (2) estimate pure error through replication. Most RSM designs include 3-6 center point replicates.`,
    practicalAdvice: 'Always include center points in optimization experiments! They are essential for fitting quadratic models and detecting non-linear effects. Run 3-6 center point replicates for pure error estimation. If the center average differs significantly from factorial average, curvature exists and you need RSM.',
    relatedTerms: ['curvature', 'quadratic', 'pure-error', 'ccd'],
    examples: [
      '3 factors: Center at (0, 0, 0) in coded units',
      'If 4 corners average 80% but center gives 90% → Curvature!',
      'Replicated centers estimate experimental error'
    ]
  },

  'axial-points': {
    term: 'Axial Points (Star Points)',
    shortDefinition: 'Points extending beyond the factorial cube in CCD',
    fullExplanation: `Axial points (also called star points) are runs where one factor is at an extreme value (±α) while all others are at center (0). They extend beyond the factorial cube to enable quadratic term estimation. The α value (typically 1.414 for 2 factors, 1.682 for 3) makes the design rotatable - equal prediction variance in all directions.`,
    practicalAdvice: 'Axial points are what make CCD capable of optimization! They allow estimation of pure quadratic terms (X²) separately from interactions. Face-centered CCD uses α=1 (axial points on cube faces) when you cannot exceed the original factor ranges. Rotatable CCD (α > 1) provides more uniform prediction.',
    relatedTerms: ['ccd', 'quadratic', 'rotatability'],
    examples: [
      '3 factors: 6 axial points like (1.682, 0, 0), (-1.682, 0, 0), ...',
      'Enable estimating β₁₁X₁² separately from β₁₂X₁X₂',
      'Face-centered: α=1 (on cube face), Rotatable: α=1.414 to 2'
    ]
  },

  'curvature': {
    term: 'Curvature',
    shortDefinition: 'Non-linear response - quadratic effects',
    fullExplanation: `Curvature means the response does not change linearly with factors - there's a "bend" in the relationship. For example, yield might increase with temperature up to 180°C (optimal), then decrease beyond that. This requires quadratic terms (X²) in the model. Two-level factorials cannot detect curvature; RSM designs are needed.`,
    practicalAdvice: 'Test for curvature by comparing center points to factorial average. If significantly different, curvature exists and you need RSM (CCD or Box-Behnken) to model it. Curvature is common in optimization - the best settings are usually in the middle of the design space, not at extremes.',
    relatedTerms: ['quadratic', 'center-points', 'rsm', 'ccd'],
    examples: [
      'Y = 50 + 10X + 5X² → Curved (parabola)',
      'Optimum often occurs at peak of curve (dY/dX = 0)',
      'Two-level designs miss curvature; need 3+ levels or axial points'
    ]
  },

  'quadratic': {
    term: 'Quadratic Model',
    shortDefinition: 'Second-order polynomial including squared terms',
    fullExplanation: `A quadratic model includes linear terms (X), squared terms (X²), and interactions (XY):

Y = β₀ + β₁X₁ + β₂X₂ + β₁₁X₁² + β₂₂X₂² + β₁₂X₁X₂

This captures curvature and finds optima. RSM designs (CCD, Box-Behnken) provide the structure to estimate all these terms. Quadratic models can predict maximum or minimum response.`,
    practicalAdvice: 'Use quadratic models for optimization. Fit the full model first, then remove non-significant terms. Check model adequacy with R², lack-of-fit test, and residual plots. The stationary point (where ∂Y/∂X = 0) gives the predicted optimum. Validate experimentally!',
    relatedTerms: ['rsm', 'curvature', 'ccd', 'optimization'],
    examples: [
      'Y = 75 + 5X₁ + 3X₂ - 2X₁² - 1X₂² + 4X₁X₂',
      'Find optimum: Take derivatives, set to zero, solve',
      'Requires 3+ levels or center + axial points'
    ]
  },

  // Practical Concepts
  'screening': {
    term: 'Screening Experiments',
    shortDefinition: 'Identify which factors matter from a large set',
    fullExplanation: `Screening experiments test many factors (5-15+) efficiently to identify the "vital few" that significantly affect the response. They use fractional factorial or Plackett-Burman designs with minimal runs. The goal is factor selection, not optimization. Non-significant factors are dropped; important ones move to follow-up optimization studies.`,
    practicalAdvice: 'Use screening for 5+ factors when you suspect only a few are important. Keep designs simple (2 levels, no center points) to maximize efficiency. Use Pareto charts to identify active factors visually. Plan on a 2-stage process: screening → optimization with important factors only.',
    relatedTerms: ['plackett-burman', 'fractional-factorial', 'pareto-chart'],
    examples: [
      'Screen 10 factors in 12 runs → Identify 3 important ones',
      'Follow up with CCD on 3 factors (20 runs)',
      'Total: 32 runs vs 1024 for full 10-factor design!'
    ]
  },

  'optimization': {
    term: 'Optimization Experiments',
    shortDefinition: 'Find the best factor settings to maximize/minimize response',
    fullExplanation: `Optimization experiments use RSM designs (CCD, Box-Behnken) to model the response surface and find optimal factor combinations. They require 2-5 factors (typically after screening), multiple levels, and designs that support quadratic models. The output is predicted optimal settings and expected response at the optimum.`,
    practicalAdvice: 'Run optimization after screening to focus on important factors only. Use CCD or Box-Behnken with 3-6 center points. Fit quadratic model, generate contour plots, find stationary point. Validate the predicted optimum with confirmation runs (typically 3-5 replicates). Watch for saddle points (not true optima)!',
    relatedTerms: ['rsm', 'ccd', 'box-behnken', 'quadratic', 'contour-plots'],
    examples: [
      'Find temperature and pressure that maximize yield',
      'Typical workflow: 12 runs screening → 20 runs RSM',
      'Confirm predicted optimum with replicate runs'
    ]
  },

  'blocking': {
    term: 'Blocking',
    shortDefinition: 'Grouping runs to control known sources of variation',
    fullExplanation: `Blocking divides experimental runs into groups (blocks) run under similar conditions to control nuisance factors. For example, if you cannot run all 16 experiments on the same day, use 2 blocks of 8 runs each. Differences between days are "blocked out" - removed from error term. This makes factor effects clearer.`,
    practicalAdvice: 'Use blocking when: (1) experiments span multiple days/batches/operators, (2) you can only run a few experiments at a time, or (3) known nuisance factors exist. Design blocks so main factors are not confounded with blocks. Include block terms in your statistical model to account for block-to-block variation.',
    relatedTerms: ['randomization', 'nuisance-factors', 'replication'],
    examples: [
      '16 runs over 2 days → 2 blocks of 8 runs each',
      'Block by: day, batch, operator, machine',
      'Removes day-to-day variation from experimental error'
    ]
  },

  'randomization': {
    term: 'Randomization',
    shortDefinition: 'Running experiments in random order to prevent bias',
    fullExplanation: `Randomization means conducting experimental runs in a random sequence rather than a systematic order. This prevents time trends, learning effects, and unknown confounders from biasing your results. For example, if you run all low-temperature experiments first and high-temperature last, any time trend will be confused with the temperature effect.`,
    practicalAdvice: 'ALWAYS randomize run order! Use a random number generator or statistical software to create the sequence. If complete randomization is impossible (e.g., temperature changes take hours), use restricted randomization or blocking. Randomization is your insurance against unknown confounding factors.',
    relatedTerms: ['blocking', 'confounding', 'run-order'],
    examples: [
      'Run experiments 5, 12, 3, 8, ... (random) not 1, 2, 3, 4 (sequential)',
      'Protects against time trends, operator learning, equipment drift',
      'Standard practice in all designed experiments'
    ]
  },

  'replication': {
    term: 'Replication',
    shortDefinition: 'Repeating experimental runs to estimate error',
    fullExplanation: `Replication means running the same factor combination multiple times independently (not just measuring the same sample twice - that is repeated measurement). Replicates estimate experimental error (noise) and allow statistical testing. True replicates reset everything: new material, new setup, different day ideally.`,
    practicalAdvice: 'Include replicates (3-6 center points minimum) to estimate pure error. This allows: (1) testing factor significance, (2) checking model lack-of-fit, and (3) calculating confidence intervals. Do not confuse replication (different experimental runs) with repeated measurements (same run measured multiple times).',
    relatedTerms: ['center-points', 'pure-error', 'experimental-error'],
    examples: [
      'Run (150°C, 30 PSI) three times on different days',
      'Center point replicates estimate error without using degrees of freedom',
      'More replicates → Better error estimate → More powerful tests'
    ]
  },

  // Design Properties
  'coded-values': {
    term: 'Coded Values',
    shortDefinition: 'Standardized scale (-1, 0, +1) used in design',
    fullExplanation: `Coded values transform real factor levels to a standardized -1 to +1 scale. For example, temperature 150-200°C becomes -1 to +1. Low level = -1, center = 0, high = +1. This standardization makes designs easier to construct, ensures orthogonality, and allows comparing effect magnitudes across factors with different units.`,
    practicalAdvice: 'Most DOE software works in coded units internally but shows you actual values. Understanding the coding helps interpret: axial points (±1.414), rotatability, and effect sizes. To decode: Actual = Center + (Coded × Range/2). For example: Coded 1 at 150-200°C → 175 + (1 × 25) = 200°C.',
    relatedTerms: ['actual-values', 'scaling', 'standardization'],
    examples: [
      'Temperature 150-200°C: -1=150, 0=175, +1=200',
      'Pressure 10-50 PSI: -1=10, 0=30, +1=50',
      'Axial α=1.414: 175 + (1.414 × 25) = 210.4°C (beyond range!)'
    ]
  },

  'actual-values': {
    term: 'Actual Values',
    shortDefinition: 'Real factor levels in original measurement units',
    fullExplanation: `Actual values are the factor levels in their original engineering units (°C, PSI, minutes, pH, etc.). Before running experiments, coded values (-1, 0, +1) must be converted to actual values. For example, if coded +1 means "high" and your design says X₁ = +1, you need to know this means "run at 200°C."`,
    practicalAdvice: 'Always create a conversion table between coded and actual values before starting experiments! Document factor ranges clearly (e.g., "Temperature: 150°C low, 200°C high"). This prevents costly setup errors. After analysis, convert predicted optimal settings back to actual units for implementation.',
    relatedTerms: ['coded-values', 'factor-levels', 'scaling'],
    examples: [
      'Coded -1 → Actual 150°C',
      'Coded 0 → Actual 175°C (center)',
      'Coded +1 → Actual 200°C'
    ]
  },

  'rotatability': {
    term: 'Rotatability',
    shortDefinition: 'Uniform prediction variance in all directions',
    fullExplanation: `A rotatable design has equal prediction variance at all points equidistant from the design center. This means predictions are equally precise whether you move north, south, east, or west from the center. Achieved by choosing the correct axial distance (α) in CCD. For example, α = 1.414 for 2 factors creates rotatability.`,
    practicalAdvice: 'Rotatability is desirable but not essential for most experiments. It ensures your response surface model is equally reliable in all directions. More important for exploring unknown regions than for finding a known optimum. Face-centered CCD (α=1) is not rotatable but still very practical.',
    relatedTerms: ['ccd', 'axial-points', 'prediction-variance'],
    examples: [
      '2 factors: α = √2 = 1.414 (rotatable)',
      '3 factors: α = 1.682 (rotatable)',
      'Uniform prediction quality regardless of direction'
    ]
  },

  'design-efficiency': {
    term: 'Design Efficiency',
    shortDefinition: 'How much information per experimental run',
    fullExplanation: `Design efficiency measures how well a design estimates model parameters relative to the "best possible" design. A-efficiency, D-efficiency, and G-efficiency assess different aspects. Higher efficiency means more precise parameter estimates with fewer runs. Optimal designs maximize efficiency for your specific model.`,
    practicalAdvice: 'Use efficiency metrics when comparing alternative designs with the same factors but different run counts. D-efficiency >80% is generally good. Do not obsess over perfect efficiency - classical designs (CCD, Box-Behnken) have well-studied properties and proven track records despite not always being optimal.',
    relatedTerms: ['optimal-design', 'd-efficiency', 'information'],
    examples: [
      'D-efficiency 95% → Almost as good as optimal design',
      'CCD with 20 runs vs optimal design with 18 runs',
      'Higher efficiency → Smaller confidence intervals'
    ]
  },

  // Analysis Concepts
  'pareto-chart': {
    term: 'Pareto Chart',
    shortDefinition: 'Bar chart showing factors ranked by importance',
    fullExplanation: `A Pareto chart displays factor effects as bars ordered from largest to smallest. A reference line shows the significance threshold (usually p=0.05). Effects exceeding this line are statistically significant. It's the fastest way to identify which factors matter in screening experiments - typically a few factors dominate.`,
    practicalAdvice: 'Use Pareto charts for screening analysis to quickly identify active factors. The chart directly implements the Pareto principle: often 20% of factors cause 80% of variation. Focus on factors above the significance line for follow-up optimization. Ignore factors below the line (within noise).',
    relatedTerms: ['screening', 'main-effects', 'significance'],
    examples: [
      'Temperature: 12% effect (significant)',
      'Pressure: 8% effect (significant)',
      'Catalyst: 1% effect (not significant)',
      'Focus on Temperature and Pressure for optimization'
    ]
  },

  'contour-plots': {
    term: 'Contour Plots',
    shortDefinition: '2D map showing response surface with iso-response lines',
    fullExplanation: `Contour plots display the response surface as a topographic map where lines connect points of equal response (like elevation contours on maps). They show how two factors jointly affect the response. Circles indicate a peak (optimum), ellipses show direction of steepest ascent, and saddles reveal complex surfaces.`,
    practicalAdvice: 'Generate contour plots after fitting RSM models to visualize factor interactions and locate optima. Look for: (1) circular contours centered on optimum, (2) direction of steepest improvement, (3) flat regions (factors do not matter), and (4) saddle points (not true optima). Overlay constraints to find the feasible optimum.',
    relatedTerms: ['rsm', 'response-surface', 'optimization'],
    examples: [
      'Concentric circles → Optimum at center',
      'Parallel lines → One factor does not matter',
      'Saddle → Neither maximum nor minimum (inflection)'
    ]
  },

  'anova': {
    term: 'ANOVA (Analysis of Variance)',
    shortDefinition: 'Statistical test comparing mean differences',
    fullExplanation: `ANOVA partitions total variation in the response into: (1) variation explained by model factors (signal), and (2) unexplained random error (noise). F-tests determine if factors significantly affect the response. R² shows percentage of variation explained. P-values indicate significance of each factor.`,
    practicalAdvice: 'Use ANOVA tables to: (1) test overall model significance (model F-test), (2) test individual factor significance (p-values), and (3) assess model quality (R², Adjusted R²). Check lack-of-fit test - if significant, your model is inadequate (missing terms or wrong form). Aim for p < 0.05 for important factors.',
    relatedTerms: ['f-test', 'p-value', 'r-squared'],
    examples: [
      'F = 25.3, p < 0.0001 → Model highly significant',
      'Temperature: p = 0.002 → Significant effect',
      'R² = 0.87 → Model explains 87% of variation'
    ]
  },

  'lack-of-fit': {
    term: 'Lack-of-Fit Test',
    shortDefinition: 'Tests if model is adequate or missing terms',
    fullExplanation: `Lack-of-fit test compares model error to pure error (from replicates). Significant lack-of-fit (p < 0.05) means your model does not capture the true relationship - you may need higher-order terms, transformations, or additional factors. Requires replicated runs (like center points) to estimate pure error.`,
    practicalAdvice: 'Always check lack-of-fit! If significant: (1) add quadratic terms (if using linear model), (2) try transformations (log, sqrt), or (3) add factors/interactions. If you cannot improve it, predictions may be unreliable. Non-significant lack-of-fit (p > 0.05) is good - means model is adequate.',
    relatedTerms: ['anova', 'pure-error', 'model-adequacy'],
    examples: [
      'Lack-of-fit p = 0.34 → Model adequate (good!)',
      'Lack-of-fit p = 0.003 → Model inadequate (bad!)',
      'Requires replicates to separate lack-of-fit from pure error'
    ]
  }
}

/**
 * Get glossary term by key
 */
export const getGlossaryTerm = (termKey) => {
  return DOE_GLOSSARY[termKey] || null
}

/**
 * Search glossary by keyword
 */
export const searchGlossary = (keyword) => {
  const lowerKeyword = keyword.toLowerCase()
  return Object.entries(DOE_GLOSSARY)
    .filter(([key, value]) =>
      value.term.toLowerCase().includes(lowerKeyword) ||
      value.shortDefinition.toLowerCase().includes(lowerKeyword)
    )
    .map(([key, value]) => ({ key, ...value }))
}

/**
 * Get all glossary terms
 */
export const getAllGlossaryTerms = () => {
  return Object.entries(DOE_GLOSSARY).map(([key, value]) => ({
    key,
    ...value
  }))
}

/**
 * Get terms by category
 */
export const getTermsByCategory = () => {
  return {
    'Design Types': [
      'full-factorial',
      'fractional-factorial',
      'ccd',
      'box-behnken',
      'plackett-burman'
    ],
    'Core Concepts': [
      'main-effects',
      'interactions',
      'resolution',
      'confounding',
      'orthogonality'
    ],
    'Power & Statistics': [
      'statistical-power',
      'effect-size',
      'type-i-error',
      'type-ii-error'
    ],
    'RSM Concepts': [
      'rsm',
      'center-points',
      'axial-points',
      'curvature',
      'quadratic'
    ],
    'Practical Concepts': [
      'screening',
      'optimization',
      'blocking',
      'randomization',
      'replication'
    ],
    'Design Properties': [
      'coded-values',
      'actual-values',
      'rotatability',
      'design-efficiency'
    ],
    'Analysis Concepts': [
      'pareto-chart',
      'contour-plots',
      'anova',
      'lack-of-fit'
    ]
  }
}
