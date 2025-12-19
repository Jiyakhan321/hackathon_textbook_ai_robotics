import re
from typing import List, Tuple
from ..config.settings import settings
from ..models.book_chunk import BookChunkCreate


class ChunkingService:
    """
    Service to handle the semantic chunking of book content based on document structure
    """
    
    @staticmethod
    def chunk_text_by_structure(text: str, chapter_name: str, section_title: str) -> List[BookChunkCreate]:
        """
        Chunk the text semantically based on document structure (chapters, sections, paragraphs)
        """
        chunks = []
        
        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunk_id_counter = 1
        
        for para in paragraphs:
            # Clean up the paragraph
            para = para.strip()
            if not para:
                continue
                
            # If the paragraph is too long, split it further by sentences
            if len(para) > settings.max_chunk_length:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk + " " + sentence) <= settings.max_chunk_length:
                        if current_chunk:
                            current_chunk += " " + sentence
                        else:
                            current_chunk = sentence
                    else:
                        # Save the current chunk
                        if current_chunk.strip():
                            chunk_id = f"{chapter_name.replace(' ', '_').lower()}_chunk_{chunk_id_counter}"
                            chunk = BookChunkCreate(
                                content=current_chunk.strip(),
                                chapter_name=chapter_name,
                                section_title=section_title,
                                source_reference=f"Chapter {chapter_name}, Section {section_title}"
                            )
                            chunks.append(chunk)
                            chunk_id_counter += 1
                        # Start a new chunk with the current sentence
                        current_chunk = sentence
                
                # Add the last chunk if there's remaining content
                if current_chunk.strip():
                    chunk_id = f"{chapter_name.replace(' ', '_').lower()}_chunk_{chunk_id_counter}"
                    chunk = BookChunkCreate(
                        content=current_chunk.strip(),
                        chapter_name=chapter_name,
                        section_title=section_title,
                        source_reference=f"Chapter {chapter_name}, Section {section_title}"
                    )
                    chunks.append(chunk)
                    chunk_id_counter += 1
            else:
                # Add the paragraph as a single chunk
                chunk_id = f"{chapter_name.replace(' ', '_').lower()}_chunk_{chunk_id_counter}"
                chunk = BookChunkCreate(
                    content=para,
                    chapter_name=chapter_name,
                    section_title=section_title,
                    source_reference=f"Chapter {chapter_name}, Section {section_title}"
                )
                chunks.append(chunk)
                chunk_id_counter += 1

        return chunks

    @staticmethod
    def chunk_selected_text(selected_text: str) -> List[BookChunkCreate]:
        """
        Chunk the selected text for temporary use in selected-text mode
        """
        # For selected text, we'll create a single chunk or split if too long
        chunks = []
        chunk_id_counter = 1

        # If the selected text is too long, split it by paragraphs or sentences
        if len(selected_text) > settings.max_chunk_length:
            paragraphs = re.split(r'\n\s*\n', selected_text)
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                    
                if len(para) > settings.max_chunk_length:
                    # Split by sentences if paragraph is still too long
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    current_chunk = ""
                    
                    for sentence in sentences:
                        if len(current_chunk + " " + sentence) <= settings.max_chunk_length:
                            if current_chunk:
                                current_chunk += " " + sentence
                            else:
                                current_chunk = sentence
                        else:
                            # Save the current chunk
                            if current_chunk.strip():
                                chunk_id = f"selected_text_chunk_{chunk_id_counter}"
                                chunk = BookChunkCreate(
                                    content=current_chunk.strip(),
                                    chapter_name="Selected Text",
                                    section_title="User Selection",
                                    source_reference="User selected text"
                                )
                                chunks.append(chunk)
                                chunk_id_counter += 1
                            # Start a new chunk with the current sentence
                            current_chunk = sentence
                    
                    # Add the last chunk if there's remaining content
                    if current_chunk.strip():
                        chunk_id = f"selected_text_chunk_{chunk_id_counter}"
                        chunk = BookChunkCreate(
                            content=current_chunk.strip(),
                            chapter_name="Selected Text",
                            section_title="User Selection",
                            source_reference="User selected text"
                        )
                        chunks.append(chunk)
                        chunk_id_counter += 1
                else:
                    # Add the paragraph as a single chunk
                    chunk_id = f"selected_text_chunk_{chunk_id_counter}"
                    chunk = BookChunkCreate(
                        content=para,
                        chapter_name="Selected Text",
                        section_title="User Selection",
                        source_reference="User selected text"
                    )
                    chunks.append(chunk)
                    chunk_id_counter += 1
        else:
            # Add the selected text as a single chunk
            chunk_id = f"selected_text_chunk_{chunk_id_counter}"
            chunk = BookChunkCreate(
                content=selected_text,
                chapter_name="Selected Text",
                section_title="User Selection",
                source_reference="User selected text"
            )
            chunks.append(chunk)

        return chunks