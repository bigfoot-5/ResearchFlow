import json
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from sqlalchemy.orm import Session

from src.config import settings
from src.llm.llm_client import OllamaLLMClient
from src.core_logic.icp_analysis import (
    ICPAnalysisEngine, 
    ICPDefinition, 
    ICPScoredEntity,
    ICPCriterion,
    ICPInsight
)

logger = logging.getLogger(__name__)


class ReflectionAgent:
    """A reflection agent that can reflect on and improve outputs."""
    
    def __init__(
        self,
        generator_prompts: List[Dict[str, str]],
        reflection_prompts: List[Dict[str, str]],
        num_steps: int = 1
    ):
        """
        Initialize the reflection agent.
        
        Args:
            generator_prompts: Initial system prompts for the generator
            reflection_prompts: Initial system prompts for the reflector
            num_steps: Number of reflection iterations to perform
        """
        self.llm_client = OllamaLLMClient()
        self.generator_prompts = list(generator_prompts)
        self.reflection_prompts = list(reflection_prompts)
        self.generated_output = None
        self.reflection_feedback = None
        self.improved_output = None
        self.num_steps = num_steps
    
    def generate_initial(self, user_prompt: str) -> str:
        """Generate initial output from the generator."""
        system_prompt = self.generator_prompts[0]["content"] if self.generator_prompts else ""
        prompt = f"{system_prompt}\n\n{user_prompt}"
        self.generated_output = self.llm_client.generate_text(prompt=prompt)
        return self.generated_output
    
    def reflect(self) -> str:
        """Reflect on the generated output."""
        system_prompt = self.reflection_prompts[0]["content"] if self.reflection_prompts else ""
        prompt = f"{system_prompt}\n\nHere is the output to reflect on: {self.generated_output}"
        self.reflection_feedback = self.llm_client.generate_text(prompt=prompt)
        return self.reflection_feedback
    
    def improve(self) -> str:
        """Improve the output based on reflection feedback."""
        system_prompt = self.generator_prompts[0]["content"] if self.generator_prompts else ""
        prompt = f"{system_prompt}\n\nHere is the feedback to incorporate: {self.reflection_feedback}"
        self.improved_output = self.llm_client.generate_text(prompt=prompt)
        return self.improved_output
    
    def run(self, user_prompt: str, verbose: bool = False) -> str:
        """Run the complete reflection process."""
        output = self.generate_initial(user_prompt)
        if verbose:
            logger.info("Initial output generated")
            
        for step in range(self.num_steps):
            feedback = self.reflect()
            if verbose:
                logger.info(f"Reflection step {step+1}: Feedback generated")
                
            output = self.improve()
            if verbose:
                logger.info(f"Reflection step {step+1}: Output improved")
                
            # Prepare for next iteration if needed
            self.generated_output = self.improved_output
            
        return output


class ReflectiveICPEngine:
    """Enhanced ICP engine that uses reflection to improve analysis quality"""
    
    def __init__(self, reflection_steps: int = 2):
        """
        Initialize the reflective ICP engine.
        
        Args:
            reflection_steps: Number of reflection iterations to perform
        """
        self.core_engine = ICPAnalysisEngine()
        self.reflection_steps = reflection_steps
    
    def enhance_output(self, output: str, steps: int = None) -> str:
        """
        Enhance any output text using reflection.
        
        Args:
            output: The text to enhance with reflection
            steps: Optional override for number of reflection steps
            
        Returns:
            Enhanced output text
        """
        num_steps = steps if steps is not None else self.reflection_steps
        
        # Set up generator prompt
        generator_prompts = [
            {
                "role": "system",
                "content": """You are an expert at improving analysis and reports.
                Your task is to refine and improve the provided text, making it more accurate,
                insightful, and actionable. Focus on clarity, specificity, and business value.
                Maintain the same general structure and format of the original."""
            }
        ]
        
        # Set up reflection prompt
        reflection_prompts = [
            {
                "role": "system",
                "content": """You are a critical evaluator of business analysis.
                Your job is to provide constructive criticism on the analysis, including:
                1. Clarity and specificity issues
                2. Missing context or explanations
                3. Logical inconsistencies or gaps
                4. Actionability of insights
                5. Overall quality and usefulness
                
                Provide detailed, constructive feedback on how to improve the analysis."""
            }
        ]
        
        # Create reflection agent
        reflector = ReflectionAgent(
            generator_prompts=generator_prompts,
            reflection_prompts=reflection_prompts,
            num_steps=num_steps
        )
        
        # Run reflection process
        user_prompt = f"""Here is the original text to improve:
        
        {output}
        
        Your task is to enhance this text by:
        1. Making it more specific and actionable
        2. Ensuring the supporting evidence is relevant and compelling
        3. Improving clarity and readability
        4. Adding any missing insights or context
        
        Maintain the same general structure and format of the original.
        """
        
        try:
            logger.info("Starting output enhancement through reflection")
            enhanced_output = reflector.run(user_prompt, verbose=True)
            return enhanced_output
            
        except Exception as e:
            logger.error(f"Error in reflective enhancement: {e}")
            logger.info("Falling back to original output")
            return output
    
    def generate_icp_definition(self, data: Dict[str, Any], reflection_steps: int = None) -> Dict[str, Any]:
        """
        Generate an ICP definition based on provided data with reflection for enhancement.
        This method is useful for API endpoints that need to enhance definitions.
        
        Args:
            data: The ICP definition data
            reflection_steps: Optional override for number of reflection steps
            
        Returns:
            Enhanced ICP definition
        """
        # Convert the data to JSON
        data_json = json.dumps(data)
        
        # Use the enhance_output method
        enhanced_json = self.enhance_output(data_json, steps=reflection_steps)
        
        try:
            # Try to parse the enhanced output as JSON
            enhanced_data = json.loads(enhanced_json)
            
            # Add a flag indicating reflection was applied
            enhanced_data["reflection_applied"] = True
            enhanced_data["reflection_steps"] = reflection_steps if reflection_steps is not None else self.reflection_steps
            
            return enhanced_data
        except json.JSONDecodeError:
            # If parsing fails, return the original data with a flag
            logger.error("Enhanced output could not be parsed as JSON")
            data["reflection_applied"] = False
            return data
    
    def generate_improved_icp_definition(self, db: Session) -> ICPDefinition:
        """
        Generate an improved ICP definition using reflection.
        
        First generates a standard ICP definition, then uses a reflection agent
        to critique and improve it.
        """
        # Get initial ICP definition
        initial_icp = self.core_engine.generate_icp_definition_from_data(db)
        
        # Set up generator prompt
        generator_prompts = [
            {
                "role": "system",
                "content": """You are an expert at defining Ideal Customer Profiles based on CRM data analysis.
                Your task is to refine and improve the ICP definition provided, making it more accurate,
                insightful, and actionable. Respond with the improved ICP definition in valid JSON format
                that matches the exact structure of the input."""
            }
        ]
        
        # Set up reflection prompt
        reflection_prompts = [
            {
                "role": "system",
                "content": """You are a senior sales strategist specializing in ICP definition critique.
                Your job is to critically evaluate the provided ICP definition, looking for:
                1. Missing important factors or criteria
                2. Inappropriate weighting of criteria
                3. Inaccurate or overly broad descriptions
                4. Factors that might not actually be predictive of success
                5. Overall actionability and business value
                
                Provide detailed, constructive feedback on how to improve the ICP definition."""
            }
        ]
        
        # Create reflection agent
        reflector = ReflectionAgent(
            generator_prompts=generator_prompts,
            reflection_prompts=reflection_prompts,
            num_steps=self.reflection_steps
        )
        
        # Run reflection process
        initial_icp_json = json.dumps(initial_icp.dict())
        user_prompt = f"""This is an automatically generated ICP definition based on CRM data:
        
        {initial_icp_json}
        
        Your task is to analyze this definition and improve it by:
        1. Refining the criteria and weights based on business logic
        2. Making the description more actionable and specific
        3. Ensuring all patterns are significant and relevant
        
        Respond with the improved ICP definition in JSON format that exactly matches the structure of the input.
        """
        
        try:
            logger.info("Starting ICP definition improvement through reflection")
            improved_icp_json = reflector.run(user_prompt, verbose=True)
            
            # Parse the improved definition
            improved_icp_data = json.loads(improved_icp_json)
            return ICPDefinition(**improved_icp_data)
            
        except Exception as e:
            logger.error(f"Error in reflective ICP generation: {e}")
            logger.info("Falling back to original ICP definition")
            return initial_icp
    
    def analyze_deal_patterns_with_reflection(self, db: Session) -> List[ICPInsight]:
        """Generate improved deal pattern insights using reflection."""
        # Get initial insights
        initial_insights = self.core_engine.analyze_deal_patterns(db)
        
        # Set up generator prompt
        generator_prompts = [
            {
                "role": "system",
                "content": """You are an expert CRM data analyst specializing in identifying actionable patterns
                in deals data. Your task is to improve the provided insights, making them more accurate, 
                insightful, and actionable. Respond with the improved insights in valid JSON format that
                matches the exact structure of the input."""
            }
        ]
        
        # Set up reflection prompt
        reflection_prompts = [
            {
                "role": "system",
                "content": """You are a critical evaluator of CRM insights. Analyze the provided insights for:
                1. Statistical significance and validity
                2. Business relevance and actionability
                3. Clarity and specificity
                4. Supporting evidence quality
                5. Potential confounding factors or alternative explanations
                
                Provide detailed, constructive feedback on how to improve these insights."""
            }
        ]
        
        # Create reflection agent
        reflector = ReflectionAgent(
            generator_prompts=generator_prompts,
            reflection_prompts=reflection_prompts,
            num_steps=self.reflection_steps
        )
        
        # Run reflection process
        initial_insights_json = json.dumps([insight.dict() for insight in initial_insights])
        user_prompt = f"""These are automatically generated insights about deal patterns from CRM data:
        
        {initial_insights_json}
        
        Your task is to analyze these insights and improve them by:
        1. Making them more specific and actionable
        2. Ensuring the supporting evidence is relevant and compelling
        3. Adjusting confidence levels based on evidence quality
        4. Adding missing insights that might be evident from the patterns
        
        Respond with the improved insights in JSON format that exactly matches the structure of the input.
        """
        
        try:
            logger.info("Starting deal pattern insights improvement through reflection")
            improved_insights_json = reflector.run(user_prompt, verbose=True)
            
            # Parse the improved insights
            improved_insights_data = json.loads(improved_insights_json)
            return [ICPInsight(**data) for data in improved_insights_data]
            
        except Exception as e:
            logger.error(f"Error in reflective deal pattern analysis: {e}")
            logger.info("Falling back to original insights")
            return initial_insights
    
    def generate_unified_icp_report_with_reflection(self, db: Session) -> Dict[str, Any]:
        """Generate an improved unified ICP report using reflection."""
        # Get initial report
        initial_report = self.core_engine.generate_unified_icp_report(db)
        
        # Set up generator prompt
        generator_prompts = [
            {
                "role": "system",
                "content": """You are an expert at creating comprehensive ICP reports for business stakeholders.
                Your task is to refine and improve the provided report, making it more insightful, coherent,
                and actionable. Respond with the improved report in valid JSON format that matches the
                exact structure of the input."""
            }
        ]
        
        # Set up reflection prompt
        reflection_prompts = [
            {
                "role": "system",
                "content": """You are a senior business consultant specialized in evaluating CRM analysis reports.
                Critically evaluate the provided ICP report for:
                1. Clarity and coherence of the overall narrative
                2. Quality and actionability of insights
                3. Consistency between different sections
                4. Missing important perspectives or analysis angles
                5. Executive-level value and strategic guidance
                
                Provide detailed, constructive feedback on how to improve the report."""
            }
        ]
        
        # Create reflection agent
        reflector = ReflectionAgent(
            generator_prompts=generator_prompts,
            reflection_prompts=reflection_prompts,
            num_steps=self.reflection_steps
        )
        
        # Run reflection process
        initial_report_json = json.dumps(initial_report)
        user_prompt = f"""This is an automatically generated ICP report based on CRM data:
        
        {initial_report_json}
        
        Your task is to analyze this report and improve it by:
        1. Enhancing the strategic narrative and key takeaways
        2. Making recommendations more specific and actionable
        3. Ensuring all sections are cohesive and tell a clear story
        4. Adding executive-level insights that might be missing
        
        Respond with the improved report in JSON format that exactly matches the structure of the input.
        """
        
        try:
            logger.info("Starting unified ICP report improvement through reflection")
            improved_report_json = reflector.run(user_prompt, verbose=True)
            
            # Parse the improved report
            improved_report = json.loads(improved_report_json)
            return improved_report
            
        except Exception as e:
            logger.error(f"Error in reflective unified report generation: {e}")
            logger.info("Falling back to original report")
            return initial_report


# Simple test function to verify the implementation
def test_reflective_icp():
    """Basic test of the reflective ICP engine."""
    from src.database.database_setup import get_db
    
    db = next(get_db())
    engine = ReflectiveICPEngine(reflection_steps=1)
    
    try:
        # Test improved ICP definition
        improved_icp = engine.generate_improved_icp_definition(db)
        print(f"Improved ICP definition: {improved_icp.description}")
        
        # Test improved deal patterns
        improved_insights = engine.analyze_deal_patterns_with_reflection(db)
        print(f"Generated {len(improved_insights)} improved insights")
        
        return True
    except Exception as e:
        print(f"Error in test: {e}")
        return False


if __name__ == "__main__":
    test_reflective_icp() 