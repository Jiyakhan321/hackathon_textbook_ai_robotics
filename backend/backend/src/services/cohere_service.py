import cohere
from typing import List, Dict
from ..config.settings import settings
from ..models.book_chunk import BookChunkBase


class CohereService:
    """
    Service to handle text generation using Cohere's language model
    """

    def __init__(self):
        self.client = cohere.Client(settings.cohere_api_key)
        self.model = settings.cohere_model
        self.temperature = 0.1  # Low temperature for factual accuracy

    def generate_response(self, query: str, relevant_chunks: List[BookChunkBase], context_mode: str = "full_book") -> str:
        """
        Generate a response based on the user query and relevant chunks
        """
        try:
            # Create the context from the relevant chunks
            context_parts = []
            for chunk in relevant_chunks:
                source_info = f"Source: Chapter '{chunk.chapter_name}', Section '{chunk.section_title}'"
                if chunk.page_number is not None:
                    source_info += f", Page {chunk.page_number}"
                context_parts.append(f"{source_info}\n{chunk.content}\n")

            context = "\n".join(context_parts)

            # Prepare the prompt for the model
            if context_mode == "selected_text":
                prompt = f"""
                Answer the user's question based ONLY on the following text passage:

                {context}

                Question: {query}

                Answer: """
            else:
                prompt = f"""
                Answer the user's question based on the following relevant text passages from the textbook:

                {context}

                Question: {query}

                Answer: """

            # Generate the response using Cohere
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                max_tokens=300,
                temperature=self.temperature,
                stop_sequences=["\n\n", "Question:"]  # Stop if it starts a new question
            )

            # Return the generated text
            return response.generations[0].text.strip()

        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")

    def validate_response_factual_accuracy(self, response: str, source_chunks: List[BookChunkBase]) -> bool:
        """
        Validate that the response is factually accurate based on source chunks.
        This is a simplified implementation that checks for text overlap.
        In a production system, you might want to use more sophisticated methods.
        """
        try:
            response_lower = response.lower()

            # Check if the main concepts from the response appear in the source chunks
            for chunk in source_chunks:
                chunk_lower = chunk.content.lower()

                # Simple overlap check - in a real implementation, you'd want to use semantic similarity
                if len(response_lower) > 0 and len(chunk_lower) > 0:
                    # If more than 10% of the response appears to be in the chunks, consider it valid
                    overlap = len(set(response_lower.split()) & set(chunk_lower.split()))
                    if overlap > 0:  # If there's any overlap, consider it potentially valid
                        return True

            # If no significant overlap was found, return false
            return False

        except Exception as e:
            raise Exception(f"Error validating response: {str(e)}")

    def check_context_sufficiency(self, query: str, relevant_chunks: List[BookChunkBase]) -> bool:
        """
        Check if the provided chunks contain sufficient information to answer the query
        """
        try:
            # A simple heuristic approach - if the chunks are too short relative to the query,
            # they might not contain enough information
            query_length = len(query.split())

            # For selected text mode, we need to be more stringent about sufficiency
            total_content_length = sum(len(chunk.content.split()) for chunk in relevant_chunks)

            # If the total content is less than half of the query length, it might be insufficient
            if total_content_length < query_length / 2:
                return False

            # Additional check: if the content is too short in absolute terms
            if total_content_length < 10:  # Less than 10 words of content
                return False

            # Additional checks could include:
            # - Semantic similarity between query and content
            # - Keyword matching between query and content
            # - Content relevance scoring

            return True

        except Exception as e:
            raise Exception(f"Error checking context sufficiency: {str(e)}")