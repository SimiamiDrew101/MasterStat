/**
 * Protocol Section Components
 * Individual form sections for the Protocol Generator
 */

import { useState } from 'react'
import { Plus, Trash2, ChevronDown, ChevronUp } from 'lucide-react'

/**
 * Objective Section - Research question, hypothesis, outcomes
 */
export const ObjectiveSection = ({ protocol, updateField }) => {
  const [showAdvanced, setShowAdvanced] = useState(false)

  const addSecondaryOutcome = () => {
    const current = protocol.objective.secondaryOutcomes || []
    updateField('objective', 'secondaryOutcomes', [...current, ''])
  }

  const removeSecondaryOutcome = (index) => {
    const current = protocol.objective.secondaryOutcomes || []
    updateField('objective', 'secondaryOutcomes', current.filter((_, i) => i !== index))
  }

  const updateSecondaryOutcome = (index, value) => {
    const current = [...protocol.objective.secondaryOutcomes]
    current[index] = value
    updateField('objective', 'secondaryOutcomes', current)
  }

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-100">Objective & Hypothesis</h2>

      <div>
        <label className="block text-gray-200 font-medium mb-2">
          Research Question *
        </label>
        <textarea
          value={protocol.objective.researchQuestion}
          onChange={(e) => updateField('objective', 'researchQuestion', e.target.value)}
          placeholder="What scientific question are you trying to answer? Be specific and measurable."
          className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600 rounded-lg text-gray-100 focus:ring-2 focus:ring-indigo-500"
          rows={3}
        />
        <p className="text-sm text-gray-400 mt-1">
          Example: "What is the effect of temperature and pressure on reaction yield?"
        </p>
      </div>

      <div>
        <label className="block text-gray-200 font-medium mb-2">
          Hypothesis *
        </label>
        <textarea
          value={protocol.objective.hypothesis}
          onChange={(e) => updateField('objective', 'hypothesis', e.target.value)}
          placeholder="State your testable hypothesis"
          className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600 rounded-lg text-gray-100 focus:ring-2 focus:ring-indigo-500"
          rows={3}
        />
        <p className="text-sm text-gray-400 mt-1">
          Example: "Increasing temperature will increase yield, and the effect will be greater at higher pressures."
        </p>
      </div>

      <div>
        <label className="block text-gray-200 font-medium mb-2">
          Primary Outcome Variable *
        </label>
        <input
          type="text"
          value={protocol.objective.primaryOutcome}
          onChange={(e) => updateField('objective', 'primaryOutcome', e.target.value)}
          placeholder="Main response variable you will measure"
          className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600 rounded-lg text-gray-100 focus:ring-2 focus:ring-indigo-500"
        />
      </div>

      {/* Secondary Outcomes */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="text-gray-200 font-medium">Secondary Outcomes (Optional)</label>
          <button
            onClick={addSecondaryOutcome}
            className="flex items-center gap-1 px-3 py-1 bg-indigo-600 hover:bg-indigo-700 text-white rounded text-sm"
          >
            <Plus className="w-3 h-3" />
            Add Outcome
          </button>
        </div>
        <div className="space-y-2">
          {protocol.objective.secondaryOutcomes?.map((outcome, idx) => (
            <div key={idx} className="flex gap-2">
              <input
                type="text"
                value={outcome}
                onChange={(e) => updateSecondaryOutcome(idx, e.target.value)}
                placeholder={`Secondary outcome ${idx + 1}`}
                className="flex-1 px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-gray-100"
              />
              <button
                onClick={() => removeSecondaryOutcome(idx)}
                className="px-3 py-2 bg-red-600 hover:bg-red-700 text-white rounded"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Advanced Section */}
      <div>
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center gap-2 text-indigo-400 hover:text-indigo-300"
        >
          {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          <span className="font-medium">Advanced: Success Criteria</span>
        </button>
        {showAdvanced && (
          <div className="mt-3">
            <textarea
              value={protocol.objective.successCriteria}
              onChange={(e) => updateField('objective', 'successCriteria', e.target.value)}
              placeholder="Define specific criteria for successful experiment (e.g., p < 0.05, R² > 0.80)"
              className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600 rounded-lg text-gray-100 focus:ring-2 focus:ring-indigo-500"
              rows={2}
            />
          </div>
        )}
      </div>
    </div>
  )
}

/**
 * Materials Section - Factors, equipment, sample size
 */
export const MaterialsSection = ({ protocol, updateField }) => {
  const addEquipment = () => {
    const current = protocol.materials.equipment || []
    updateField('materials', 'equipment', [...current, ''])
  }

  const updateEquipment = (index, value) => {
    const current = [...protocol.materials.equipment]
    current[index] = value
    updateField('materials', 'equipment', current)
  }

  const removeEquipment = (index) => {
    const current = protocol.materials.equipment || []
    updateField('materials', 'equipment', current.filter((_, i) => i !== index))
  }

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-100">Materials & Experimental Design</h2>

      {/* Factors Summary */}
      <div>
        <label className="block text-gray-200 font-medium mb-2">Factors *</label>
        {protocol.materials.factors && protocol.materials.factors.length > 0 ? (
          <div className="bg-slate-700/30 rounded-lg p-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {protocol.materials.factors.map((factor, idx) => (
                <div key={idx} className="bg-slate-800/50 rounded p-3">
                  <div className="font-semibold text-gray-100">{factor.name || `Factor ${idx + 1}`}</div>
                  <div className="text-sm text-gray-400 mt-1">
                    {factor.low !== undefined && factor.high !== undefined
                      ? `Range: ${factor.low} to ${factor.high} ${factor.units || ''}`
                      : 'Range not specified'}
                  </div>
                  <div className="text-xs text-gray-500 mt-0.5">
                    Type: {factor.type || 'continuous'}
                  </div>
                </div>
              ))}
            </div>
            <p className="text-sm text-gray-400 mt-3">
              Factors are defined from the experimental design. To modify, return to the design configuration.
            </p>
          </div>
        ) : (
          <div className="bg-orange-900/20 border border-orange-700 rounded-lg p-4">
            <p className="text-orange-300 text-sm">
              No factors defined yet. Factors will be populated from the experimental design or can be added manually.
            </p>
          </div>
        )}
      </div>

      {/* Sample Size */}
      <div>
        <label className="block text-gray-200 font-medium mb-2">Sample Size / Number of Runs *</label>
        <input
          type="number"
          value={protocol.materials.sampleSize || ''}
          onChange={(e) => updateField('materials', 'sampleSize', parseInt(e.target.value) || null)}
          placeholder="Total number of experimental runs"
          className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600 rounded-lg text-gray-100 focus:ring-2 focus:ring-indigo-500"
          min="3"
        />
      </div>

      {/* Experimental Units */}
      <div>
        <label className="block text-gray-200 font-medium mb-2">Experimental Units</label>
        <textarea
          value={protocol.materials.experimentalUnits}
          onChange={(e) => updateField('materials', 'experimentalUnits', e.target.value)}
          placeholder="Describe the experimental units (e.g., 'Individual reactors', 'Test plots', 'Patients')"
          className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600 rounded-lg text-gray-100 focus:ring-2 focus:ring-indigo-500"
          rows={2}
        />
      </div>

      {/* Equipment List */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="text-gray-200 font-medium">Equipment & Materials</label>
          <button
            onClick={addEquipment}
            className="flex items-center gap-1 px-3 py-1 bg-indigo-600 hover:bg-indigo-700 text-white rounded text-sm"
          >
            <Plus className="w-3 h-3" />
            Add Item
          </button>
        </div>
        <div className="space-y-2">
          {protocol.materials.equipment?.map((item, idx) => (
            <div key={idx} className="flex gap-2">
              <input
                type="text"
                value={item}
                onChange={(e) => updateEquipment(idx, e.target.value)}
                placeholder={`Equipment/material ${idx + 1}`}
                className="flex-1 px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-gray-100"
              />
              <button
                onClick={() => removeEquipment(idx)}
                className="px-3 py-2 bg-red-600 hover:bg-red-700 text-white rounded"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Sampling Procedure */}
      <div>
        <label className="block text-gray-200 font-medium mb-2">Sampling Procedure</label>
        <textarea
          value={protocol.materials.samplingProcedure}
          onChange={(e) => updateField('materials', 'samplingProcedure', e.target.value)}
          placeholder="Describe how samples will be collected and prepared"
          className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600 rounded-lg text-gray-100 focus:ring-2 focus:ring-indigo-500"
          rows={3}
        />
      </div>
    </div>
  )
}

/**
 * Procedure Section - Preparation, execution, measurement
 */
export const ProcedureSection = ({ protocol, updateField }) => {
  const addExecutionStep = () => {
    const current = protocol.procedure.executionSteps || []
    updateField('procedure', 'executionSteps', [...current, ''])
  }

  const updateExecutionStep = (index, value) => {
    const current = [...protocol.procedure.executionSteps]
    current[index] = value
    updateField('procedure', 'executionSteps', current)
  }

  const removeExecutionStep = (index) => {
    const current = protocol.procedure.executionSteps || []
    updateField('procedure', 'executionSteps', current.filter((_, i) => i !== index))
  }

  const addQualityControl = () => {
    const current = protocol.procedure.qualityControls || []
    updateField('procedure', 'qualityControls', [...current, ''])
  }

  const updateQualityControl = (index, value) => {
    const current = [...protocol.procedure.qualityControls]
    current[index] = value
    updateField('procedure', 'qualityControls', current)
  }

  const removeQualityControl = (index) => {
    const current = protocol.procedure.qualityControls || []
    updateField('procedure', 'qualityControls', current.filter((_, i) => i !== index))
  }

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-100">Experimental Procedure</h2>

      {/* Preparation */}
      <div>
        <label className="block text-gray-200 font-medium mb-2">Preparation Steps</label>
        <textarea
          value={protocol.procedure.preparation}
          onChange={(e) => updateField('procedure', 'preparation', e.target.value)}
          placeholder="List all preparation steps before starting the experiment"
          className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600 rounded-lg text-gray-100 focus:ring-2 focus:ring-indigo-500"
          rows={4}
        />
      </div>

      {/* Execution Steps */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="text-gray-200 font-medium">Execution Steps *</label>
          <button
            onClick={addExecutionStep}
            className="flex items-center gap-1 px-3 py-1 bg-indigo-600 hover:bg-indigo-700 text-white rounded text-sm"
          >
            <Plus className="w-3 h-3" />
            Add Step
          </button>
        </div>
        <p className="text-sm text-gray-400 mb-3">
          Detailed step-by-step procedure for conducting the experiment
        </p>
        <div className="space-y-2">
          {protocol.procedure.executionSteps?.map((step, idx) => (
            <div key={idx} className="flex gap-2 items-start">
              <span className="text-gray-400 font-mono text-sm mt-3">{idx + 1}.</span>
              <textarea
                value={step}
                onChange={(e) => updateExecutionStep(idx, e.target.value)}
                placeholder={`Step ${idx + 1}`}
                className="flex-1 px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-gray-100"
                rows={2}
              />
              <button
                onClick={() => removeExecutionStep(idx)}
                className="px-3 py-2 bg-red-600 hover:bg-red-700 text-white rounded"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Measurement Protocol */}
      <div>
        <label className="block text-gray-200 font-medium mb-2">Measurement Protocol *</label>
        <textarea
          value={protocol.procedure.measurementProtocol}
          onChange={(e) => updateField('procedure', 'measurementProtocol', e.target.value)}
          placeholder="Describe how response variables will be measured (equipment, precision, replicates, timing)"
          className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600 rounded-lg text-gray-100 focus:ring-2 focus:ring-indigo-500"
          rows={3}
        />
      </div>

      {/* Safety Precautions */}
      <div>
        <label className="block text-gray-200 font-medium mb-2">Safety Precautions</label>
        <textarea
          value={protocol.procedure.safetyPrecautions}
          onChange={(e) => updateField('procedure', 'safetyPrecautions', e.target.value)}
          placeholder="List safety considerations and required PPE"
          className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600 rounded-lg text-gray-100 focus:ring-2 focus:ring-indigo-500"
          rows={2}
        />
      </div>

      {/* Quality Controls */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="text-gray-200 font-medium">Quality Control Measures</label>
          <button
            onClick={addQualityControl}
            className="flex items-center gap-1 px-3 py-1 bg-indigo-600 hover:bg-indigo-700 text-white rounded text-sm"
          >
            <Plus className="w-3 h-3" />
            Add QC
          </button>
        </div>
        <div className="space-y-2">
          {protocol.procedure.qualityControls?.map((qc, idx) => (
            <div key={idx} className="flex gap-2">
              <input
                type="text"
                value={qc}
                onChange={(e) => updateQualityControl(idx, e.target.value)}
                placeholder={`Quality control ${idx + 1}`}
                className="flex-1 px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-gray-100"
              />
              <button
                onClick={() => removeQualityControl(idx)}
                className="px-3 py-2 bg-red-600 hover:bg-red-700 text-white rounded"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Deviation Protocol */}
      <div>
        <label className="block text-gray-200 font-medium mb-2">Protocol Deviation Handling</label>
        <textarea
          value={protocol.procedure.deviationProtocol}
          onChange={(e) => updateField('procedure', 'deviationProtocol', e.target.value)}
          placeholder="How will protocol deviations be documented and addressed?"
          className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600 rounded-lg text-gray-100 focus:ring-2 focus:ring-indigo-500"
          rows={2}
        />
      </div>
    </div>
  )
}

/**
 * Data Recording Section - Response variables, data entry, QA
 */
export const DataRecordingSection = ({ protocol, updateField }) => {
  const addResponseVariable = () => {
    const current = protocol.dataRecording.responseVariables || []
    updateField('dataRecording', 'responseVariables', [...current, ''])
  }

  const updateResponseVariable = (index, value) => {
    const current = [...protocol.dataRecording.responseVariables]
    current[index] = value
    updateField('dataRecording', 'responseVariables', current)
  }

  const removeResponseVariable = (index) => {
    const current = protocol.dataRecording.responseVariables || []
    updateField('dataRecording', 'responseVariables', current.filter((_, i) => i !== index))
  }

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-100">Data Recording & Management</h2>

      {/* Response Variables */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="text-gray-200 font-medium">Response Variables *</label>
          <button
            onClick={addResponseVariable}
            className="flex items-center gap-1 px-3 py-1 bg-indigo-600 hover:bg-indigo-700 text-white rounded text-sm"
          >
            <Plus className="w-3 h-3" />
            Add Variable
          </button>
        </div>
        <p className="text-sm text-gray-400 mb-3">
          List all variables that will be measured and recorded
        </p>
        <div className="space-y-2">
          {protocol.dataRecording.responseVariables?.map((variable, idx) => (
            <div key={idx} className="flex gap-2">
              <input
                type="text"
                value={variable}
                onChange={(e) => updateResponseVariable(idx, e.target.value)}
                placeholder={`Response variable ${idx + 1} (e.g., Yield (%), Temperature (°C))`}
                className="flex-1 px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-gray-100"
              />
              <button
                onClick={() => removeResponseVariable(idx)}
                className="px-3 py-2 bg-red-600 hover:bg-red-700 text-white rounded"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Data Collection Form */}
      <div>
        <label className="block text-gray-200 font-medium mb-2">Data Collection Form</label>
        <textarea
          value={protocol.dataRecording.dataCollectionForm}
          onChange={(e) => updateField('dataRecording', 'dataCollectionForm', e.target.value)}
          placeholder="Describe the data collection form or template (paper or electronic)"
          className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600 rounded-lg text-gray-100 focus:ring-2 focus:ring-indigo-500"
          rows={3}
        />
      </div>

      {/* Data Entry Method */}
      <div>
        <label className="block text-gray-200 font-medium mb-2">Data Entry Method *</label>
        <textarea
          value={protocol.dataRecording.entryMethod}
          onChange={(e) => updateField('dataRecording', 'entryMethod', e.target.value)}
          placeholder="How will data be entered and stored? (e.g., Direct electronic entry, Paper then transcribed)"
          className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600 rounded-lg text-gray-100 focus:ring-2 focus:ring-indigo-500"
          rows={2}
        />
      </div>

      {/* Quality Assurance */}
      <div>
        <label className="block text-gray-200 font-medium mb-2">Quality Assurance Procedures</label>
        <textarea
          value={protocol.dataRecording.qualityAssurance}
          onChange={(e) => updateField('dataRecording', 'qualityAssurance', e.target.value)}
          placeholder="How will data quality be ensured? (e.g., Double entry, Range checks, Validation rules)"
          className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600 rounded-lg text-gray-100 focus:ring-2 focus:ring-indigo-500"
          rows={3}
        />
      </div>

      {/* Backup Procedure */}
      <div>
        <label className="block text-gray-200 font-medium mb-2">Data Backup Procedure</label>
        <textarea
          value={protocol.dataRecording.backupProcedure}
          onChange={(e) => updateField('dataRecording', 'backupProcedure', e.target.value)}
          placeholder="How often and where will data be backed up?"
          className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600 rounded-lg text-gray-100 focus:ring-2 focus:ring-indigo-500"
          rows={2}
        />
      </div>

      {/* Data Storage */}
      <div>
        <label className="block text-gray-200 font-medium mb-2">Long-term Data Storage</label>
        <textarea
          value={protocol.dataRecording.dataStorage}
          onChange={(e) => updateField('dataRecording', 'dataStorage', e.target.value)}
          placeholder="Where will data be stored long-term? How long will it be retained?"
          className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600 rounded-lg text-gray-100 focus:ring-2 focus:ring-indigo-500"
          rows={2}
        />
      </div>
    </div>
  )
}

export default {
  ObjectiveSection,
  MaterialsSection,
  ProcedureSection,
  DataRecordingSection,
}
