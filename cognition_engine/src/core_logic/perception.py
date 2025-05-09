import sys
import os
from typing import List, Dict, Optional

# Add project root to Python path for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # From src/core_logic back to cognition_engine
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.llm.llm_client import OllamaLLMClient # Import our LLM client
from thefuzz import process as fuzzy_process # For fuzzy string matching

# Standard buckets for Closed Lost Reason classification
CLOSED_LOST_REASON_BUCKETS = [
    "Pricing/Budget",
    "Lost to Competitor",
    "Product Fit/Feature Gap",
    "No Decision/Project Shelved",
    "Lack of Engagement/Champion",
    "Implementation Concerns (Timeline, Complexity)",
    "Company Strategy/Internal Changes",
    "Other"
]

DEFAULT_FUZZY_MATCH_THRESHOLD = 75 # Score out of 100

class PerceptionService:
    def __init__(self, llm_client: Optional[OllamaLLMClient] = None):
        if llm_client:
            self.llm_client = llm_client
        else:
            # Initialize client with a low temperature suitable for classification
            self.llm_client = OllamaLLMClient(temperature=0.2) 

    def classify_closed_lost_reason(self, closed_lost_reason_text: str) -> str:
        """
        Classifies a free-text closed lost reason into one of the predefined buckets using an LLM.
        Returns one of the CLOSED_LOST_REASON_BUCKETS or 'Other' if no clear match.
        """
        if not self.llm_client or not self.llm_client.llm:
            print("LLM client not available, cannot classify. Defaulting to 'Other'.")
            return "Other"
        
        if not closed_lost_reason_text or not closed_lost_reason_text.strip():
            return "Other" # Or perhaps a "Not Specified" category

        # Prepare the list of buckets for the prompt
        bucket_list_str = "\n".join([f"- {bucket}" for bucket in CLOSED_LOST_REASON_BUCKETS])

        prompt = f"""
        You are an expert sales operations analyst. Your task is to classify a given 'Closed Lost Reason' into one of the following predefined categories.
        Please respond with ONLY the category name that best fits the reason. Do not add any extra explanation or punctuation.

        Categories:
        {bucket_list_str}

        Closed Lost Reason: "{closed_lost_reason_text}"
        
        Category:
        """

        # Temperature is now set at the client level
        # raw_response = self.llm_client.generate_text(prompt, temperature=0.2)
        raw_response = self.llm_client.generate_text(prompt)

        if raw_response:
            cleaned_response = raw_response.strip().replace('"', '')
            if not cleaned_response: # Handle empty string from LLM after cleaning
                print(f"LLM returned an empty or whitespace-only response for: {closed_lost_reason_text}. Defaulting to 'Other'.")
                return "Other"

            # Use fuzzy matching to find the best bucket
            # extractOne returns a tuple: (choice, score, index_if_provided_sequence_else_key)
            best_match = fuzzy_process.extractOne(cleaned_response, CLOSED_LOST_REASON_BUCKETS)
            
            if best_match and best_match[1] >= DEFAULT_FUZZY_MATCH_THRESHOLD:
                print(f"Fuzzy match for '{cleaned_response}': '{best_match[0]}' (Score: {best_match[1]})")
                return best_match[0]
            else:
                print(f"LLM response '{cleaned_response}' did not meet fuzzy match threshold ({DEFAULT_FUZZY_MATCH_THRESHOLD}). Best attempt: {best_match}. Defaulting to 'Other'.")
                return "Other"
        else:
            print(f"LLM did not return a response for: {closed_lost_reason_text}. Defaulting to 'Other'.")
            return "Other"

    # Placeholder for NLP summarization of qualitative data (e.g., Use Case field normalization)
    def normalize_use_case(self, use_case_text: str) -> str:
        """
        Placeholder for normalizing/summarizing use case text.
        For MVP, this might just return the text or do simple keyword extraction.
        """
        # TODO: Implement actual LLM-based normalization/summarization if needed
        print(f"normalize_use_case called with: {use_case_text} (Not implemented, returning as is)")
        return use_case_text

# Example Usage:
if __name__ == "__main__":
    print("Testing PerceptionService with Fuzzy Matching...")
    # PerceptionService now initializes its own LLM client with appropriate temperature
    perception_service = PerceptionService()

    if not perception_service.llm_client or not perception_service.llm_client.llm:
        print("LLM client not initialized in PerceptionService. Aborting tests.")
    else:
        test_reasons = [
            "Customer went with Competitor B because their pricing was 20% lower.",
            "The project was put on hold indefinitely due to internal restructuring.",
            "We couldn't get the main stakeholder to attend the final demo call.",
            "Product lacked critical feature X that they needed for their workflow.",
            "Too expensive for their current financial year.",
            "No response after initial quote was sent.", # Expected to match 'Lack of Engagement/Champion'
            "They loved the product but the timeline for integration was too long for their go-live.", # Expected to match 'Implementation Concerns'
            "Our UI was considered clunky compared to Acme Corp.",
            "Budget was reallocated to another department.",
            "Just decided not to move forward at this time.", # Should be 'No Decision/Project Shelved'
            "Feature set wasn't comprehensive enough for our needs"
        ]

        for reason in test_reasons:
            classified_bucket = perception_service.classify_closed_lost_reason(reason)
            print(f"Reason: \"{reason}\" -> Classified Bucket: {classified_bucket}\n")

        print("\nTesting Use Case Normalization (placeholder)...")
        normalized_uc = perception_service.normalize_use_case("Client wants to improve sales team efficiency and reduce manual data entry for their field agents.")
        print(f"Normalized Use Case: {normalized_uc}") 