/**
 * Protocol Templates and Data Structures
 * Defines the complete protocol state object and design-specific templates
 */

/**
 * Create an empty protocol with all required sections
 * @param {string} designType - Type of experimental design (factorial, rcbd, ccd, etc.)
 * @returns {Object} Empty protocol object with default values
 */
export const createEmptyProtocol = (designType = 'factorial') => ({
  // Metadata
  metadata: {
    title: '',
    investigator: '',
    institution: '',
    date: new Date().toISOString().split('T')[0],
    designType: designType,
    protocolVersion: '1.0',
    randomSeed: Math.floor(Math.random() * 1000000), // Default random seed
  },

  // Section 1: Objective
  objective: {
    researchQuestion: '',
    hypothesis: '',
    primaryOutcome: '',
    secondaryOutcomes: [],
    successCriteria: '',
  },

  // Section 2: Materials
  materials: {
    factors: [], // From design: [{ name, levels, units, type, low, high }]
    equipment: [],
    materials: [],
    sampleSize: null,
    samplingProcedure: '',
    experimentalUnits: '',
  },

  // Section 3: Experimental Procedure
  procedure: {
    preparation: '',
    executionSteps: [],
    measurementProtocol: '',
    safetyPrecautions: '',
    qualityControls: [],
    deviationProtocol: '',
  },

  // Section 4: Randomization
  randomization: {
    method: 'complete', // 'complete' | 'block' | 'restricted'
    seed: Math.floor(Math.random() * 1000000), // Reproducible seed
    blockSize: null, // For block randomization
    stratificationFactors: [],
    allocationConcealment: false,
    randomizedDesign: null, // Will hold the randomized run order
  },

  // Section 5: Blinding
  blinding: {
    type: 'none', // 'none' | 'single' | 'double'
    blindedParties: [], // ['experimenter', 'evaluator', 'subject']
    codeType: 'alphabetic', // 'alphabetic' | 'numeric' | 'custom'
    customCodes: {}, // { 'Treatment A': 'Code X' }
    generatedCodes: {}, // { 'Treatment A': 'T1' }
    unblindingCriteria: '',
  },

  // Section 6: Data Recording
  dataRecording: {
    responseVariables: [],
    dataCollectionForm: '',
    entryMethod: '',
    qualityAssurance: '',
    backupProcedure: '',
    dataStorage: '',
  },

  // Validation status
  validation: {
    isValid: false,
    errors: [],
    warnings: [],
  },
})

/**
 * Protocol templates for different design types
 * Provides default text and guidance for each design
 */
export const PROTOCOL_TEMPLATES = {
  factorial: {
    objective: {
      researchQuestion: 'What are the main effects and interactions of [factors] on [response]?',
      hypothesis: 'Factor [X] and Factor [Y] will have significant main effects and a significant interaction effect on [response].',
      successCriteria: 'Significant main effects (p < 0.05) and/or significant interactions detected',
    },
    procedure: {
      preparation: '1. Calibrate measurement equipment\n2. Prepare treatment combinations according to design matrix\n3. Label experimental units with run numbers\n4. Randomize run order',
      executionSteps: [
        'Randomly assign experimental units to treatment combinations',
        'Apply Factor A at specified levels',
        'Apply Factor B at specified levels',
        'Allow appropriate reaction/incubation time',
        'Measure response variable(s) after specified time',
        'Record observations and any deviations from protocol',
      ],
      measurementProtocol: 'Measure response variable [name] using [method/equipment]. Record to [precision]. Take [n] replicates per measurement.',
      safetyPrecautions: 'Follow standard laboratory safety protocols. Wear appropriate PPE.',
    },
    dataRecording: {
      entryMethod: 'Record data directly into electronic data capture system with automatic validation',
      qualityAssurance: 'Double-entry verification for all measurements',
      backupProcedure: 'Daily backup to secure cloud storage',
    },
  },

  rcbd: {
    objective: {
      researchQuestion: 'What is the effect of [treatment] on [response], controlling for [blocking factor]?',
      hypothesis: 'Treatment [X] will significantly affect [response], with blocking reducing experimental error.',
      successCriteria: 'Significant treatment effect (p < 0.05) with reduced MSE compared to CRD',
    },
    procedure: {
      preparation: '1. Identify and define blocks based on [blocking factor]\n2. Ensure blocking factor is consistent within blocks\n3. Randomize treatments within each block\n4. Label experimental units',
      executionSteps: [
        'Assign experimental units to blocks based on blocking factor',
        'Within each block, randomly assign treatments',
        'Apply treatments according to randomized design',
        'Measure response variable',
        'Record block identifier for each observation',
        'Note any block-specific conditions',
      ],
      measurementProtocol: 'Measure all units within a block before proceeding to next block to minimize time effects',
      safetyPrecautions: 'Follow standard laboratory safety protocols',
    },
    dataRecording: {
      entryMethod: 'Record block ID, treatment, and response for each experimental unit',
      qualityAssurance: 'Verify block assignments before starting experiment',
    },
  },

  ccd: {
    objective: {
      researchQuestion: 'What is the optimal combination of [factors] to maximize/minimize [response]?',
      hypothesis: 'A quadratic response surface model will adequately describe the relationship between factors and response.',
      successCriteria: 'Significant curvature detected, model R² > 0.80, lack-of-fit p > 0.05',
    },
    procedure: {
      preparation: '1. Calibrate measurement equipment\n2. Prepare factorial, axial, and center point runs\n3. Randomize run order\n4. Identify optimal operating range for factors',
      executionSteps: [
        'Run factorial points to estimate main effects and interactions',
        'Run axial points at α distance to estimate curvature',
        'Run center points to estimate pure error and check for curvature',
        'Measure response for each run',
        'Fit second-order model',
        'Locate stationary point (optimum)',
        'Perform confirmatory runs at predicted optimum',
      ],
      measurementProtocol: 'Use high-precision measurement method. Center points should be true replicates.',
      safetyPrecautions: 'Axial points may be at extreme factor levels - verify safety before execution',
    },
    dataRecording: {
      entryMethod: 'Record factor levels (coded and actual) and response for each run',
      qualityAssurance: 'Verify center point replicates show consistent results (CV < 10%)',
    },
  },

  'box-behnken': {
    objective: {
      researchQuestion: 'What is the optimal combination of [factors] to maximize/minimize [response]?',
      hypothesis: 'A quadratic response surface model will adequately describe the relationship within the experimental region.',
      successCriteria: 'Model R² > 0.80, significant lack-of-fit test p > 0.05',
    },
    procedure: {
      preparation: '1. Define three-level ranges for each factor\n2. Generate Box-Behnken design points\n3. Randomize run order\n4. Prepare treatment combinations',
      executionSteps: [
        'No extreme corner points - all runs at mid-level of at least one factor',
        'Execute runs according to design matrix',
        'Measure response for each combination',
        'Fit second-order model',
        'Optimize response using fitted model',
      ],
      measurementProtocol: 'Consistent measurement conditions across all runs',
      safetyPrecautions: 'Box-Behnken designs avoid extreme corners - safer for many applications',
    },
  },

  'latin-square': {
    objective: {
      researchQuestion: 'What is the effect of [treatment] controlling for [row factor] and [column factor]?',
      hypothesis: 'Treatment effect is significant after accounting for two blocking factors.',
      successCriteria: 'Significant treatment effect with both blocking factors reducing error',
    },
    procedure: {
      preparation: '1. Identify two blocking factors (rows and columns)\n2. Generate Latin Square design\n3. Randomize rows and columns\n4. Assign treatments',
      executionSteps: [
        'Set up experimental units in row × column arrangement',
        'Apply treatments according to Latin Square',
        'Each treatment appears exactly once in each row and column',
        'Measure response',
        'Record row, column, and treatment for each unit',
      ],
      measurementProtocol: 'Measure in a systematic pattern to avoid confounding with blocking factors',
    },
  },

  plackett_burman: {
    objective: {
      researchQuestion: 'Which factors have significant effects on [response] (screening)?',
      hypothesis: 'A subset of factors will show significant effects that warrant further investigation.',
      successCriteria: 'Identify 2-4 important factors with significant effects for follow-up experiments',
    },
    procedure: {
      preparation: '1. Define high and low levels for all factors\n2. Generate Plackett-Burman design\n3. Randomize run order',
      executionSteps: [
        'Execute runs with factors at high (+) or low (-) levels',
        'Focus on main effects only (interactions assumed negligible)',
        'Measure response for each run',
        'Calculate and plot main effects',
        'Identify factors with largest effects',
        'Plan follow-up factorial or RSM experiments with important factors',
      ],
      measurementProtocol: 'Efficient measurements - this is a screening design',
    },
  },
}

/**
 * Get template for specific design type
 * @param {string} designType - Design type key
 * @returns {Object} Template object or empty object if not found
 */
export const getTemplate = (designType) => {
  return PROTOCOL_TEMPLATES[designType] || PROTOCOL_TEMPLATES.factorial
}

/**
 * Apply template to protocol (fills in suggested text)
 * @param {Object} protocol - Current protocol object
 * @param {string} designType - Design type
 * @returns {Object} Protocol with template applied
 */
export const applyTemplate = (protocol, designType) => {
  const template = getTemplate(designType)

  return {
    ...protocol,
    metadata: {
      ...protocol.metadata,
      designType,
    },
    objective: {
      ...protocol.objective,
      researchQuestion: protocol.objective.researchQuestion || template.objective?.researchQuestion || '',
      hypothesis: protocol.objective.hypothesis || template.objective?.hypothesis || '',
      successCriteria: protocol.objective.successCriteria || template.objective?.successCriteria || '',
    },
    procedure: {
      ...protocol.procedure,
      preparation: protocol.procedure.preparation || template.procedure?.preparation || '',
      executionSteps: protocol.procedure.executionSteps.length > 0
        ? protocol.procedure.executionSteps
        : template.procedure?.executionSteps || [],
      measurementProtocol: protocol.procedure.measurementProtocol || template.procedure?.measurementProtocol || '',
      safetyPrecautions: protocol.procedure.safetyPrecautions || template.procedure?.safetyPrecautions || '',
    },
    dataRecording: {
      ...protocol.dataRecording,
      entryMethod: protocol.dataRecording.entryMethod || template.dataRecording?.entryMethod || '',
      qualityAssurance: protocol.dataRecording.qualityAssurance || template.dataRecording?.qualityAssurance || '',
      backupProcedure: protocol.dataRecording.backupProcedure || template.dataRecording?.backupProcedure || '',
    },
  }
}

/**
 * Design type display names
 */
export const DESIGN_TYPE_NAMES = {
  factorial: 'Factorial Design',
  rcbd: 'Randomized Complete Block Design',
  ccd: 'Central Composite Design',
  'box-behnken': 'Box-Behnken Design',
  'latin-square': 'Latin Square Design',
  'graeco-latin': 'Graeco-Latin Square',
  plackett_burman: 'Plackett-Burman Screening',
  'fractional-factorial': 'Fractional Factorial Design',
}

export default {
  createEmptyProtocol,
  PROTOCOL_TEMPLATES,
  getTemplate,
  applyTemplate,
  DESIGN_TYPE_NAMES,
}
