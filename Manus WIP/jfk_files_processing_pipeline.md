# Budget-Friendly Processing Pipeline for JFK Files

## Overview

This document outlines a budget-friendly approach to process the large JFK Files collection (65.9 GB, 73,056 files) and make it queryable through an LLM system. The pipeline is designed for technical novices with limited budgets, focusing on open-source tools and efficient processing strategies.

## Pipeline Architecture

The processing pipeline consists of four main stages:

1. **OCR Processing**: Convert poor quality scanned documents to machine-readable text
2. **Text Extraction**: Extract and clean text from all document types
3. **Chunking & Embedding**: Split documents into appropriate chunks and create vector embeddings
4. **RAG Implementation**: Set up a retrieval system to query the processed documents

### Stage 1: OCR Processing

**Recommended Tool**: OCRmyPDF
- Open-source command-line tool for adding OCR layer to PDFs
- Supports batch processing with GNU Parallel
- Can process multiple files concurrently
- Automatically repairs PDFs before processing

**Implementation Approach**:
- Process files in batches of manageable size (e.g., 1000 files per batch)
- Use GNU Parallel to utilize multiple CPU cores
- Skip files that already contain text layers

**Sample Command**:
```bash
find /path/to/jfk_files -name "*.pdf" | parallel --tag -j 2 ocrmypdf '{}' '{}'
```

### Stage 2: Text Extraction

**Recommended Tools**:
- PyPDF2/pdfplumber: For extracting text from PDFs with text layers
- Unstructured.io: Open-source library for extracting content from various document types

**Implementation Approach**:
- Extract text from OCR-processed PDFs
- Preserve document metadata (title, date, source)
- Clean extracted text (remove headers/footers, fix common OCR errors)
- Store extracted text in plain text files for easier processing

### Stage 3: Chunking & Embedding

**Recommended Tools**:
- LangChain: Open-source framework for document processing
- Sentence-Transformers: Open-source embedding models

**Chunking Strategy**:
- Use recursive text splitting with appropriate separators
- Target chunk size: 500-1000 tokens (balances context and specificity)
- Include overlap between chunks (100-200 tokens) to maintain context
- Preserve document structure where possible

**Embedding Approach**:
- Use lightweight embedding models (e.g., all-MiniLM-L6-v2)
- Process in batches to manage memory usage
- Store embeddings in a vector database

### Stage 4: RAG Implementation

**Recommended Tools**:
- Chroma DB: Lightweight, open-source vector database
- LlamaIndex: Open-source framework for building RAG applications
- Ollama: Run open-source LLMs locally

**Implementation Approach**:
- Set up Chroma DB to store document chunks and embeddings
- Implement retrieval mechanism using similarity search
- Connect to an LLM for query processing
- Create a simple interface for querying the knowledge base

## Computational Requirements

### Hardware Recommendations:
- **Minimum**: 16GB RAM, 4-core CPU, 100GB free storage
- **Recommended**: 32GB RAM, 8-core CPU, 200GB free storage
- **Storage**: External hard drive for original files backup

### Processing Time Estimates:
- OCR Processing: 2-5 minutes per file (depending on complexity)
- Full Pipeline: Several days to weeks for the entire collection
- Recommendation: Process in batches of 1000-5000 files

### Cost Considerations:
- All recommended tools are open-source and free to use
- Primary cost is computational resources and time
- Optional: Cloud computing resources for faster processing (~$50-200 depending on processing time)

## Incremental Processing Strategy

Given the large collection size, an incremental approach is recommended:

1. **Start Small**: Begin with a subset (e.g., 1000 files) to test the pipeline
2. **Prioritize Content**: Process the most important or relevant documents first
3. **Batch Processing**: Organize processing in manageable batches
4. **Iterative Improvement**: Refine the pipeline based on initial results

This approach allows for:
- Early validation of the processing pipeline
- Ability to query initial results while processing continues
- Opportunity to adjust parameters based on quality assessment
- Manageable resource utilization

## Next Steps

The next section will provide detailed step-by-step implementation instructions for each stage of the pipeline.
