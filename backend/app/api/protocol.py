"""
Experimental Protocol API
Handles protocol generation, randomization, and PDF export
"""

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime
from app.utils.report_generator import PDFReportGenerator
from reportlab.lib.units import inch

router = APIRouter()

# ============================================================================
# Request/Response Models
# ============================================================================

class RandomizationRequest(BaseModel):
    """Request model for randomization endpoint"""
    design: List[Dict[str, Any]]  # Design matrix
    method: str  # 'complete' | 'block' | 'restricted'
    seed: int
    block_size: Optional[int] = None
    restriction_factor: Optional[str] = None

class ProtocolPDFRequest(BaseModel):
    """Request model for PDF generation"""
    metadata: Dict[str, Any]
    objective: Dict[str, Any]
    materials: Dict[str, Any]
    procedure: Dict[str, Any]
    randomization: Dict[str, Any]
    blinding: Dict[str, Any]
    dataRecording: Dict[str, Any]

# ============================================================================
# Randomization Endpoint - SEED-BASED for reproducibility
# ============================================================================

@router.post("/randomize")
async def randomize_design(request: RandomizationRequest):
    """
    Seed-based randomization endpoint for reproducible experimental designs

    CRITICAL: Using the same seed will produce IDENTICAL results every time
    """
    try:
        # Set numpy random seed for reproducibility
        np.random.seed(request.seed)

        n_runs = len(request.design)

        if n_runs == 0:
            raise HTTPException(status_code=400, detail="Design matrix is empty")

        if request.method == 'complete':
            # Complete Randomization (CRD) - shuffle all runs
            indices = np.arange(n_runs)
            np.random.shuffle(indices)

            randomized = [
                {
                    **request.design[i],
                    'runOrder': idx + 1,
                    'originalOrder': i + 1
                }
                for idx, i in enumerate(indices)
            ]

        elif request.method == 'block':
            # Block Randomization (RBD)
            if not request.block_size or request.block_size < 2:
                raise HTTPException(status_code=400, detail="Block size must be at least 2 for block randomization")

            # Split into blocks
            blocks = [request.design[i:i+request.block_size] for i in range(0, n_runs, request.block_size)]
            randomized_blocks = []

            for block_num, block in enumerate(blocks):
                block_indices = np.arange(len(block))
                np.random.shuffle(block_indices)

                randomized_block = [
                    {
                        **block[i],
                        'block': block_num + 1
                    }
                    for i in block_indices
                ]
                randomized_blocks.extend(randomized_block)

            # Add global run order
            randomized = [
                {**run, 'runOrder': idx + 1, 'originalOrder': idx + 1}
                for idx, run in enumerate(randomized_blocks)
            ]

        elif request.method == 'restricted':
            # Restricted Randomization - balance by restriction factor
            if not request.restriction_factor:
                raise HTTPException(status_code=400, detail="Restriction factor required for restricted randomization")

            # Group by restriction factor
            groups = {}
            for run in request.design:
                key = run.get(request.restriction_factor, 'unknown')
                if key not in groups:
                    groups[key] = []
                groups[key].append(run)

            # Randomize each group
            randomized_groups = []
            for group_runs in groups.values():
                indices = np.arange(len(group_runs))
                np.random.shuffle(indices)
                randomized_groups.append([group_runs[i] for i in indices])

            # Interleave groups
            max_length = max(len(g) for g in randomized_groups)
            interleaved = []

            for i in range(max_length):
                for group in randomized_groups:
                    if i < len(group):
                        interleaved.append(group[i])

            randomized = [
                {**run, 'runOrder': idx + 1, 'originalOrder': idx + 1}
                for idx, run in enumerate(interleaved)
            ]

        else:
            raise HTTPException(status_code=400, detail=f"Unknown randomization method: {request.method}")

        return {
            'success': True,
            'randomizedDesign': randomized,
            'seed': request.seed,
            'method': request.method,
            'timestamp': datetime.now().isoformat(),
            'message': f'Randomization completed with seed {request.seed} - results are reproducible'
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Main Protocol PDF Export
# ============================================================================

@router.post("/generate-pdf")
async def generate_protocol_pdf(request: ProtocolPDFRequest):
    """
    Generate comprehensive experimental protocol PDF
    """
    try:
        # Initialize PDF generator
        pdf = PDFReportGenerator(
            title=request.metadata.get('title', 'Experimental Protocol'),
            author=request.metadata.get('investigator', 'Unknown'),
        )

        # ===== COVER PAGE =====
        metadata_table = [
            ['Field', 'Value'],
            ['Protocol Title', request.metadata.get('title', '')],
            ['Principal Investigator', request.metadata.get('investigator', '')],
            ['Institution', request.metadata.get('institution', '')],
            ['Date', request.metadata.get('date', '')],
            ['Design Type', request.metadata.get('designType', '').upper()],
            ['Protocol Version', request.metadata.get('protocolVersion', '1.0')],
            ['Randomization Seed', str(request.randomization.get('seed', 'Not set'))],
        ]
        pdf.add_cover_page(metadata_table)

        # ===== SECTION 1: OBJECTIVE & HYPOTHESIS =====
        pdf.add_section('1. Research Objective & Hypothesis')

        pdf.add_subsection('Research Question')
        pdf.add_paragraph(request.objective.get('researchQuestion', 'Not specified'))

        pdf.add_subsection('Hypothesis')
        pdf.add_paragraph(request.objective.get('hypothesis', 'Not specified'))

        pdf.add_subsection('Primary Outcome')
        pdf.add_paragraph(request.objective.get('primaryOutcome', 'Not specified'))

        if request.objective.get('secondaryOutcomes'):
            pdf.add_subsection('Secondary Outcomes')
            for outcome in request.objective['secondaryOutcomes']:
                if outcome:
                    pdf.add_paragraph(f"• {outcome}")

        if request.objective.get('successCriteria'):
            pdf.add_subsection('Success Criteria')
            pdf.add_paragraph(request.objective['successCriteria'])

        # ===== SECTION 2: MATERIALS & DESIGN =====
        pdf.add_section('2. Materials & Experimental Design')

        # Factors table
        if request.materials.get('factors') and len(request.materials['factors']) > 0:
            pdf.add_subsection('Factors')
            factors_data = [['Factor', 'Low Level', 'High Level', 'Units', 'Type']]
            for factor in request.materials['factors']:
                factors_data.append([
                    factor.get('name', ''),
                    str(factor.get('low', '-')),
                    str(factor.get('high', '-')),
                    factor.get('units', ''),
                    factor.get('type', 'continuous'),
                ])
            pdf.add_table(factors_data, col_widths=[2*inch, 1.2*inch, 1.2*inch, 1*inch, 1*inch])

        pdf.add_subsection('Sample Size')
        pdf.add_paragraph(f"Total experimental runs: {request.materials.get('sampleSize', 'Not specified')}")

        if request.materials.get('experimentalUnits'):
            pdf.add_subsection('Experimental Units')
            pdf.add_paragraph(request.materials['experimentalUnits'])

        if request.materials.get('equipment') and len(request.materials['equipment']) > 0:
            pdf.add_subsection('Equipment & Materials')
            for item in request.materials['equipment']:
                if item:
                    pdf.add_paragraph(f"• {item}")

        if request.materials.get('samplingProcedure'):
            pdf.add_subsection('Sampling Procedure')
            pdf.add_paragraph(request.materials['samplingProcedure'])

        # ===== SECTION 3: EXPERIMENTAL PROCEDURE =====
        pdf.add_section('3. Experimental Procedure')

        if request.procedure.get('preparation'):
            pdf.add_subsection('Preparation')
            pdf.add_paragraph(request.procedure['preparation'])

        pdf.add_subsection('Execution Steps')
        if request.procedure.get('executionSteps'):
            for i, step in enumerate(request.procedure['executionSteps'], 1):
                if step:
                    pdf.add_paragraph(f"{i}. {step}")
        else:
            pdf.add_paragraph("No execution steps specified")

        if request.procedure.get('measurementProtocol'):
            pdf.add_subsection('Measurement Protocol')
            pdf.add_paragraph(request.procedure['measurementProtocol'])

        if request.procedure.get('safetyPrecautions'):
            pdf.add_subsection('Safety Precautions')
            pdf.add_paragraph(request.procedure['safetyPrecautions'])

        if request.procedure.get('qualityControls') and len(request.procedure['qualityControls']) > 0:
            pdf.add_subsection('Quality Control Measures')
            for qc in request.procedure['qualityControls']:
                if qc:
                    pdf.add_paragraph(f"• {qc}")

        # ===== SECTION 4: RANDOMIZATION =====
        pdf.add_section('4. Randomization Procedure')

        method = request.randomization.get('method', 'complete')
        pdf.add_paragraph(f"Randomization Method: {method.capitalize()}")

        seed = request.randomization.get('seed')
        if seed:
            pdf.add_paragraph(f"Random Seed: {seed}")
            pdf.add_paragraph("⚠️ IMPORTANT: Using this seed will produce IDENTICAL randomization results, ensuring full reproducibility.")
        else:
            pdf.add_paragraph("⚠️ WARNING: No random seed specified - randomization is NOT reproducible!")

        if method == 'block' and request.randomization.get('blockSize'):
            pdf.add_paragraph(f"Block Size: {request.randomization['blockSize']}")

        if request.randomization.get('allocationConcealment'):
            pdf.add_paragraph("Allocation Concealment: YES - assignment sequence hidden until allocation")

        # Randomized run order preview
        if request.randomization.get('randomizedDesign'):
            design = request.randomization['randomizedDesign']
            if len(design) > 0:
                pdf.add_subsection('Randomized Run Order (First 20 runs)')
                pdf.add_paragraph(f"Total runs: {len(design)}")
                # Could add a table here showing run order

        # ===== SECTION 5: BLINDING =====
        pdf.add_section('5. Blinding & Masking')

        blinding_type = request.blinding.get('type', 'none')

        if blinding_type == 'none':
            pdf.add_paragraph('This is an open-label study with no blinding.')
        else:
            pdf.add_paragraph(f"Blinding Type: {blinding_type.upper()}-BLIND")

            blinded_parties = request.blinding.get('blindedParties', [])
            if blinded_parties:
                pdf.add_paragraph(f"Blinded parties: {', '.join(blinded_parties)}")

            code_type = request.blinding.get('codeType', 'alphabetic')
            pdf.add_paragraph(f"Treatment Code Type: {code_type.capitalize()}")

            # IMPORTANT: Do NOT include actual codes in main protocol!
            pdf.add_paragraph("\n⚠️ CRITICAL SECURITY NOTICE:")
            pdf.add_paragraph("Treatment assignment codes are documented in a SEPARATE blinding key file.")
            pdf.add_paragraph("The blinding key must be:")
            pdf.add_paragraph("• Kept secure and confidential")
            pdf.add_paragraph("• Stored separately from this protocol")
            pdf.add_paragraph("• NOT shared with blinded parties")
            pdf.add_paragraph("• Only accessed by authorized personnel")

            if request.blinding.get('unblindingCriteria'):
                pdf.add_subsection('Unblinding Criteria')
                pdf.add_paragraph(request.blinding['unblindingCriteria'])

        # ===== SECTION 6: DATA RECORDING =====
        pdf.add_section('6. Data Recording & Management')

        pdf.add_subsection('Response Variables')
        if request.dataRecording.get('responseVariables'):
            for var in request.dataRecording['responseVariables']:
                if var:
                    pdf.add_paragraph(f"• {var}")
        else:
            pdf.add_paragraph("No response variables specified")

        if request.dataRecording.get('dataCollectionForm'):
            pdf.add_subsection('Data Collection Form')
            pdf.add_paragraph(request.dataRecording['dataCollectionForm'])

        if request.dataRecording.get('entryMethod'):
            pdf.add_subsection('Data Entry Method')
            pdf.add_paragraph(request.dataRecording['entryMethod'])

        if request.dataRecording.get('qualityAssurance'):
            pdf.add_subsection('Quality Assurance')
            pdf.add_paragraph(request.dataRecording['qualityAssurance'])

        if request.dataRecording.get('backupProcedure'):
            pdf.add_subsection('Data Backup')
            pdf.add_paragraph(request.dataRecording['backupProcedure'])

        if request.dataRecording.get('dataStorage'):
            pdf.add_subsection('Long-term Storage')
            pdf.add_paragraph(request.dataRecording['dataStorage'])

        # Build PDF
        pdf_bytes = pdf.build()

        # Return as downloadable file
        title_slug = request.metadata.get('title', 'protocol').replace(' ', '_')
        timestamp = datetime.now().strftime('%Y%m%d')
        filename = f"protocol_{title_slug}_{timestamp}.pdf"

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

# ============================================================================
# Blinding Key PDF Export (SEPARATE FILE)
# ============================================================================

@router.post("/generate-blinding-key")
async def generate_blinding_key_pdf(request: ProtocolPDFRequest):
    """
    Generate SEPARATE blinding key PDF

    CRITICAL: This file must NEVER be distributed with the main protocol!
    """
    try:
        pdf = PDFReportGenerator(
            title="BLINDING KEY - CONFIDENTIAL",
            author=request.metadata.get('investigator', 'Unknown'),
        )

        # Warning cover
        pdf.add_section('⚠️ CONFIDENTIAL BLINDING KEY ⚠️')
        pdf.add_paragraph('\nThis document contains treatment assignment codes.')
        pdf.add_paragraph('DO NOT share with blinded parties or store with the main protocol.')
        pdf.add_paragraph('Access restricted to authorized unblinded personnel only.')
        pdf.add_spacer(0.5*inch)

        # Protocol information
        pdf.add_subsection('Protocol Information')
        info_table = [
            ['Field', 'Value'],
            ['Protocol Title', request.metadata.get('title', '')],
            ['Investigator', request.metadata.get('investigator', '')],
            ['Date', request.metadata.get('date', '')],
            ['Blinding Type', request.blinding.get('type', 'none').upper()],
        ]
        pdf.add_table(info_table, col_widths=[2*inch, 4*inch])

        pdf.add_spacer(0.3*inch)

        # Blinding codes table
        if request.blinding.get('generatedCodes'):
            pdf.add_subsection('Treatment Assignment Codes')
            codes_data = [['Blinding Code', 'Actual Treatment']]

            for treatment, code in request.blinding['generatedCodes'].items():
                codes_data.append([str(code), str(treatment)])

            pdf.add_table(codes_data, col_widths=[2*inch, 4*inch])
        else:
            pdf.add_paragraph('No blinding codes generated.')

        pdf.add_spacer(0.5*inch)

        # Security reminder
        pdf.add_subsection('Security Reminders')
        pdf.add_paragraph('• Store in secure, access-controlled location')
        pdf.add_paragraph('• Do NOT email or share electronically without encryption')
        pdf.add_paragraph('• Log all access to this document')
        pdf.add_paragraph('• Unblind only according to protocol criteria')

        # Build PDF
        pdf_bytes = pdf.build()

        title_slug = request.metadata.get('title', 'protocol').replace(' ', '_')
        timestamp = datetime.now().strftime('%Y%m%d')
        filename = f"CONFIDENTIAL_blinding_key_{title_slug}_{timestamp}.pdf"

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Blinding key generation failed: {str(e)}")
