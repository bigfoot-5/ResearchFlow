# Reflection Agent for Cognition Engine

This enhancement adds reflection capabilities to the Cognition Engine's ICP (Ideal Customer Profile) analysis. Using the multi-agent reflection pattern, it enables the system to critique and improve its own outputs.

## Overview

The reflection agent pattern involves:

1. **Generator**: The existing ICP analysis engine that produces initial outputs (definitions, insights, reports)
2. **Reflector**: A second agent that critiques the initial outputs
3. **Improver**: A process that takes the critiques and produces enhanced outputs

This pattern has been implemented for three key features:
- ICP Definition generation
- Deal Pattern Insights
- Comprehensive ICP Reports

## New Components

### Backend

- **`src/core_logic/reflective_icp_engine.py`**: Implements the `ReflectionAgent` class and `ReflectiveICPEngine`.
- **New API endpoints** in `src/api/endpoints/icp_endpoints.py`:
  - `/icp/reflective/definition`: Enhanced ICP definition
  - `/icp/reflective/insights`: Enhanced deal pattern insights
  - `/icp/reflective/report`: Enhanced comprehensive report

### Frontend

The Streamlit frontend now includes toggles to enable AI reflection for:
- ICP Definition generation
- Deal Pattern Insights analysis 
- Vector Search Insights

Users can control the reflection depth (1-3 steps) to balance analysis quality and response time.

## How to Use

1. Make sure OpenAI API keys are properly configured in your environment.

2. Start the backend server:
   ```bash
   cd cognition_engine
   python -m src.main
   ```

3. Start the Streamlit frontend:
   ```bash
   cd cognition_frontend
   streamlit run app.py
   ```

4. In the UI, enable "Use AI Reflection" for any of the supported features.

5. Adjust the "Reflection Depth" slider to control the number of reflection cycles:
   - 1 step: Basic reflection (faster)
   - 2 steps: Balanced (recommended)
   - 3 steps: Deep reflection (slower, potentially higher quality)

## Technical Implementation

The reflection process works as follows:

1. The initial analysis is performed using the standard `ICPAnalysisEngine`.
2. The result is passed to the `ReflectionAgent` along with specialized prompts.
3. The agent generates a critique of the initial output.
4. The critique is used to generate an improved version of the output.
5. Steps 3-4 are repeated for the configured number of reflection steps.

This pattern leverages OpenAI's GPT models, but the implementation is modular and could be adapted to use other LLM providers.

## Benefits

- **Higher Quality Analysis**: The reflection process catches logical errors, improves clarity, and adds nuance to the ICP analysis.
- **More Actionable Insights**: Critiques focus on making insights more specific and actionable for business stakeholders.
- **Reduced Hallucinations**: The reflector can identify and correct speculative claims in the initial outputs.
- **Enhanced Explanations**: The improved outputs include clearer explanations connecting evidence to conclusions.

## Performance Considerations

Each reflection step adds approximately 10-20 seconds to the response time, depending on the complexity of the data and the LLM's response time. The default configuration of 2 reflection steps balances quality improvements with reasonable response times. 